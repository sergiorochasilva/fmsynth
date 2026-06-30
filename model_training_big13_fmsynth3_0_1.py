"""Multi-task FM predictor for `dataset_big13`, version 0_1.

Architecture:
- Raw waveform input
- Differentiable multi-resolution log-mel front-end inside the model, with extra temporal derivatives
- Residual 1D CNN backbone over the time axis of the stacked spectrogram features
- Shared dense trunk with a direct `algorithm` classifier
- Auxiliary regression heads for ratio, base frequency, FM indices, detune, feedback, LFO, key scaling, and ADSR envelopes

Data flow:
- Input: `dataset_big13/parameters.csv` plus `audio_big13_manifest.json` shards or `sample_*.wav`
- Output: trained weights, preprocessing artifacts, predictions, plots, and `results.json`
"""

from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (
    BatchNormalization,
    Add,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
    LayerNormalization,
    MaxPooling1D,
    Lambda,
)
from keras.models import Model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import mixed_precision

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_PATH = Path(os.getenv("DATASET_PATH", "dataset_big13"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "model_training_big13_fmsynth3_0_1"))
MODEL_NAME = "model_training_big13_fmsynth3_0_1"

MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "0"))
RANDOM_STATE = int(os.getenv("SEED", "42"))
TRAIN_FRAC = float(os.getenv("TRAIN_FRAC", "0.75"))
VAL_FRAC = float(os.getenv("VAL_FRAC", "0.20"))

CNN_BATCH_DEFAULT = os.getenv("TRAIN_BATCH_SIZE", "8")
BATCH_SIZE = int(CNN_BATCH_DEFAULT)
PRED_BATCH_SIZE = int(os.getenv("PRED_BATCH_SIZE", "12"))
EPOCHS = int(os.getenv("EPOCHS", "120"))
PATIENCE = int(os.getenv("PATIENCE", "6"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1.8e-4"))
DROPOUT = float(os.getenv("DROPOUT", "0.08"))
BASE_FILTERS = int(os.getenv("BASE_FILTERS", "80"))
CNN_BLOCKS = int(os.getenv("CNN_BLOCKS", "6"))
DENSE_UNITS = int(os.getenv("DENSE_UNITS", "512"))
USE_MIXED_PRECISION = os.getenv("MIXED_PRECISION", "0") == "1"
ENABLE_XLA = os.getenv("ENABLE_XLA", "0") == "1"
RESUME_TRAINING = os.getenv("RESUME_TRAINING", "1") == "1"
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", "0.02"))
PREFIX_SUFFIX_LABEL_SMOOTHING = float(os.getenv("PREFIX_SUFFIX_LABEL_SMOOTHING", "0.01"))
FIT_VERBOSE = int(os.getenv("FIT_VERBOSE", "1"))
FORCE_WAV_AUDIO = os.getenv("FORCE_WAV_AUDIO", "0") == "1"

N_FFT_SMALL = int(os.getenv("N_FFT_SMALL", "512"))
N_FFT_LARGE = int(os.getenv("N_FFT_LARGE", "2048"))
HOP_LENGTH = int(os.getenv("HOP_LENGTH", "256"))
N_MELS = int(os.getenv("N_MELS", "64"))
MEL_FMIN = float(os.getenv("MEL_FMIN", "30.0"))
MEL_FMAX = float(os.getenv("MEL_FMAX", "7600.0"))
EPS = 1e-6
MEL_MATRIX_SMALL = tf.constant(
    tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=N_FFT_SMALL // 2 + 1,
        sample_rate=16000,
        lower_edge_hertz=MEL_FMIN,
        upper_edge_hertz=MEL_FMAX,
    ),
    dtype=tf.float32,
)
MEL_MATRIX_LARGE = tf.constant(
    tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=N_FFT_LARGE // 2 + 1,
        sample_rate=16000,
        lower_edge_hertz=MEL_FMIN,
        upper_edge_hertz=MEL_FMAX,
    ),
    dtype=tf.float32,
)

AUDIO_MANIFEST_PATH = BASE_PATH / "audio_big13_manifest.json"
AUDIO_LEGACY_CACHE_PATH = BASE_PATH / "audio_big13_int16.npy"
PARAMS_PATH = BASE_PATH / "parameters.csv"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_LATEST_WEIGHTS = CHECKPOINT_DIR / "latest.weights.h5"
CHECKPOINT_BEST_WEIGHTS = CHECKPOINT_DIR / "best.weights.h5"
TRAIN_STATE_PATH = CHECKPOINT_DIR / "training_state.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

if USE_MIXED_PRECISION:
    mixed_precision.set_global_policy("mixed_float16")
if not ENABLE_XLA:
    tf.config.optimizer.set_jit(False)


TARGET_SPECS = [
    {"head": "ratio_log2_head", "column": "ratio_carrier", "transform": "log2", "loss_weight": 0.75, "group": "ratio"},
    {"head": "freq_log2_head", "column": "frequencia_base", "transform": "log2", "loss_weight": 0.75, "group": "ratio"},
    {"head": "index_12_head", "column": "index_12", "transform": "identity", "loss_weight": 0.55, "group": "struct"},
    {"head": "index_23_head", "column": "index_23", "transform": "identity", "loss_weight": 0.55, "group": "struct"},
    {"head": "index_3c_head", "column": "index_3c", "transform": "identity", "loss_weight": 0.55, "group": "struct"},
    {"head": "index_4c_head", "column": "index_4c", "transform": "identity", "loss_weight": 0.55, "group": "struct"},
    {"head": "index_5c_head", "column": "index_5c", "transform": "identity", "loss_weight": 0.55, "group": "struct"},
    {"head": "detune_carrier_head", "column": "detune_carrier", "transform": "identity", "loss_weight": 0.35, "group": "struct"},
    {"head": "feedback_head", "column": "feedback", "transform": "identity", "loss_weight": 0.35, "group": "struct"},
    {"head": "lfo_rate_head", "column": "lfo_rate", "transform": "identity", "loss_weight": 0.35, "group": "struct"},
    {"head": "lfo_depth_cents_head", "column": "lfo_depth_cents", "transform": "identity", "loss_weight": 0.35, "group": "struct"},
    {"head": "key_scaling_head", "column": "key_scaling", "transform": "identity", "loss_weight": 0.35, "group": "struct"},
    {"head": "env_mod_attack_head", "column": "env_mod_attack", "transform": "log2", "loss_weight": 0.45, "group": "env"},
    {"head": "env_mod_decay_head", "column": "env_mod_decay", "transform": "log2", "loss_weight": 0.45, "group": "env"},
    {"head": "env_mod_sustain_head", "column": "env_mod_sustain", "transform": "identity", "loss_weight": 0.35, "group": "env"},
    {"head": "env_mod_release_head", "column": "env_mod_release", "transform": "log2", "loss_weight": 0.45, "group": "env"},
    {"head": "env_car_attack_head", "column": "env_car_attack", "transform": "log2", "loss_weight": 0.45, "group": "env"},
    {"head": "env_car_decay_head", "column": "env_car_decay", "transform": "log2", "loss_weight": 0.45, "group": "env"},
    {"head": "env_car_sustain_head", "column": "env_car_sustain", "transform": "identity", "loss_weight": 0.35, "group": "env"},
    {"head": "env_car_release_head", "column": "env_car_release", "transform": "log2", "loss_weight": 0.45, "group": "env"},
]

REGRESSION_HEADS = [spec["head"] for spec in TARGET_SPECS]
CONTINUOUS_GROUPS = {spec["head"]: spec["group"] for spec in TARGET_SPECS}


def to_json_scalar(value):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(value) for value in obj]
    return to_json_scalar(obj)


def atomic_json_dump(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(payload), f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_json_file(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_metric(history_df: pd.DataFrame, train_key: str, val_key: str, ylabel: str, filename: str) -> None:
    if train_key not in history_df.columns or val_key not in history_df.columns:
        return
    plt.figure(dpi=300)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.plot(history_df["epoch"], history_df[train_key], label=f"{ylabel} Training")
    plt.plot(history_df["epoch"], history_df[val_key], label=f"{ylabel} Validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def load_manifest() -> dict | None:
    if not AUDIO_MANIFEST_PATH.exists():
        return None
    try:
        with open(AUDIO_MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def normalize_audio_batch(batch: np.ndarray) -> np.ndarray:
    batch = np.asarray(batch, dtype=np.float32) / 32768.0
    return batch.astype(np.float32, copy=False)


class DenseAudioStore:
    def __init__(self, audio: np.ndarray):
        audio = np.asarray(audio)
        if audio.ndim != 2:
            raise ValueError(f"Unexpected dense audio shape: {audio.shape}")
        self.audio = audio.astype(np.int16, copy=False)
        self.sample_len = int(self.audio.shape[1])

    def get_batch(self, sample_indices: np.ndarray) -> np.ndarray:
        sample_indices = np.asarray(sample_indices, dtype=np.int32)
        return normalize_audio_batch(self.audio[sample_indices]).reshape(
            sample_indices.shape[0], self.sample_len, 1
        )


class ShardedAudioStore:
    def __init__(self, base_path: Path, manifest: dict):
        self.base_path = base_path
        self.manifest = manifest
        self.sample_len = int(manifest["audio_sample_len"])
        self.shard_size = int(manifest["audio_shard_size"])
        self.shards = list(manifest.get("shards", []))
        self.max_cached_shards = int(os.getenv("AUDIO_SHARD_CACHE", "4"))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def _load_shard(self, shard_idx: int) -> np.ndarray:
        shard_idx = int(shard_idx)
        if shard_idx in self._cache:
            self._cache.move_to_end(shard_idx)
            return self._cache[shard_idx]

        shard_meta = self.shards[shard_idx]
        shard_path = self.base_path / shard_meta["file"]
        shard = np.load(shard_path, mmap_mode="r")
        self._cache[shard_idx] = shard
        self._cache.move_to_end(shard_idx)
        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)
        return shard

    def get_batch(self, sample_indices: np.ndarray) -> np.ndarray:
        sample_indices = np.asarray(sample_indices, dtype=np.int32)
        batch = np.empty((sample_indices.shape[0], self.sample_len), dtype=np.int16)
        shard_ids = sample_indices // self.shard_size
        offsets = sample_indices % self.shard_size

        for shard_idx in np.unique(shard_ids):
            shard_mask = shard_ids == shard_idx
            shard = self._load_shard(int(shard_idx))
            batch[shard_mask] = shard[offsets[shard_mask]]

        return normalize_audio_batch(batch).reshape(sample_indices.shape[0], self.sample_len, 1)


def load_audio_store(base_path: Path, sample_ids: list[int]):
    if FORCE_WAV_AUDIO:
        samples = []
        for sample_id in sample_ids:
            wav_path = base_path / f"sample_{sample_id}.wav"
            signal, _ = sf.read(wav_path)
            samples.append(np.round(np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16))
        return DenseAudioStore(np.asarray(samples, dtype=np.int16))

    manifest = load_manifest()
    if manifest is not None:
        return ShardedAudioStore(base_path, manifest)

    if AUDIO_LEGACY_CACHE_PATH.exists():
        audio = np.load(AUDIO_LEGACY_CACHE_PATH, mmap_mode="r")
        if audio.ndim != 2:
            raise ValueError(f"Unexpected legacy cache shape: {audio.shape}")
        return DenseAudioStore(audio)

    samples = []
    for sample_id in sample_ids:
        wav_path = base_path / f"sample_{sample_id}.wav"
        signal, _ = sf.read(wav_path)
        samples.append(np.round(np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16))
    return DenseAudioStore(np.asarray(samples, dtype=np.int16))


def make_stratify_key(frame: pd.DataFrame) -> pd.Series | None:
    if "algorithm" not in frame.columns:
        return None
    if "ratio_carrier" not in frame.columns:
        return frame["algorithm"].astype(str).reset_index(drop=True)
    ratio_key = frame["ratio_carrier"].astype(np.float32).round(4).astype(str).reset_index(drop=True)
    return frame["algorithm"].astype(str).reset_index(drop=True) + "|" + ratio_key


def stratified_split_indices(indices: np.ndarray, strata: pd.Series | None, test_size: float, random_state: int):
    if strata is None:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(indices)
        cut = int(round((1.0 - test_size) * len(perm)))
        return perm[:cut], perm[cut:]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_pos, test_pos = next(splitter.split(np.zeros(len(indices)), strata))
    return indices[train_pos], indices[test_pos]


def build_ratio_classes(train_ratio_carrier: np.ndarray) -> np.ndarray:
    classes = np.unique(np.round(np.asarray(train_ratio_carrier, dtype=np.float32), 4))
    classes.sort()
    return classes.astype(np.float32, copy=False)


def assign_ratio_classes(ratio_carrier: np.ndarray, classes: np.ndarray) -> np.ndarray:
    values = np.round(np.asarray(ratio_carrier, dtype=np.float32), 4)
    bins = np.searchsorted(classes, values)
    bins = np.clip(bins, 0, len(classes) - 1)
    return bins.astype(np.int32)


def parse_algorithm_prefix(algorithm_name: str) -> str:
    if algorithm_name == "parallel5":
        return "parallel"
    if algorithm_name.startswith("series2x2"):
        return "series2x2"
    if algorithm_name.startswith("series3"):
        return "series3"
    raise ValueError(f"Unsupported algorithm label: {algorithm_name}")


def parse_algorithm_suffix(algorithm_name: str) -> int:
    if algorithm_name == "parallel5":
        return 5
    match = re.search(r"parallel(\d+)", algorithm_name)
    return int(match.group(1)) if match else 0


def sparse_categorical_focal_loss(gamma: float = 1.4):
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, EPS, 1.0 - EPS)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1], dtype=tf.float32)
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        ce = -tf.math.log(p_t)
        return tf.pow(1.0 - p_t, gamma) * ce

    return loss


def transform_series(values: pd.Series | np.ndarray, transform: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if transform == "identity":
        return arr
    if transform == "log2":
        return np.log2(np.clip(arr, 1e-8, None)).astype(np.float32)
    if transform == "log1p":
        return np.log1p(np.clip(arr, 0.0, None)).astype(np.float32)
    raise ValueError(f"Unknown transform: {transform}")


def inverse_transform_series(values: np.ndarray, transform: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if transform == "identity":
        return arr
    if transform == "log2":
        return np.power(2.0, arr)
    if transform == "log1p":
        return np.expm1(arr)
    raise ValueError(f"Unknown transform: {transform}")


class MultiTaskSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        audio_store,
        sample_indices: np.ndarray,
        y: dict[str, np.ndarray],
        batch_size: int,
        shuffle: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_store = audio_store
        self.sample_indices = np.asarray(sample_indices, dtype=np.int32)
        self.y = y
        self.batch_size = int(max(batch_size, 1))
        self.shuffle = bool(shuffle)
        self.indices = np.arange(self.sample_indices.shape[0], dtype=np.int32)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        batch_ids = self.indices[start:end]
        audio_ids = self.sample_indices[batch_ids]
        x_batch = self.audio_store.get_batch(audio_ids)
        y_batch = {name: values[batch_ids] for name, values in self.y.items()}
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def compute_logmel(x, n_fft: int, mel_matrix: tf.Tensor):
    x = tf.cast(x, tf.float32)
    x = tf.squeeze(x, axis=-1)
    x = x - tf.reduce_mean(x, axis=1, keepdims=True)
    stft = tf.signal.stft(
        x,
        frame_length=n_fft,
        frame_step=HOP_LENGTH,
        fft_length=n_fft,
        window_fn=tf.signal.hann_window,
        pad_end=True,
    )
    power = tf.square(tf.abs(stft))
    mel = tf.tensordot(power, mel_matrix, axes=[[-1], [0]])
    log_mel = tf.math.log(mel + EPS)
    mean = tf.reduce_mean(log_mel, axis=[1, 2], keepdims=True)
    std = tf.math.reduce_std(log_mel, axis=[1, 2], keepdims=True)
    return (log_mel - mean) / (std + EPS)


def normalize_time_feature(feature: tf.Tensor) -> tf.Tensor:
    feature = feature - tf.reduce_mean(feature, axis=[1, 2], keepdims=True)
    return feature / (tf.math.reduce_std(feature, axis=[1, 2], keepdims=True) + EPS)


def temporal_delta(feature: tf.Tensor) -> tf.Tensor:
    delta = feature[:, 1:, :] - feature[:, :-1, :]
    delta = tf.pad(delta, paddings=[[0, 0], [1, 0], [0, 0]])
    return normalize_time_feature(delta)


def logmel_frontend(x):
    log_mel_small = compute_logmel(x, N_FFT_SMALL, MEL_MATRIX_SMALL)
    log_mel_large = compute_logmel(x, N_FFT_LARGE, MEL_MATRIX_LARGE)
    delta_small = temporal_delta(log_mel_small)
    delta_large = temporal_delta(log_mel_large)
    delta2_large = temporal_delta(delta_large)
    return tf.concat([log_mel_small, log_mel_large, delta_small, delta_large, delta2_large], axis=-1)


def build_model(
    input_len: int,
    n_algorithm_classes: int,
) -> Model:
    inputs = Input(shape=(input_len, 1), name="audio_input")
    x = Lambda(logmel_frontend, output_shape=(None, N_MELS * 5), name="logmel_frontend")(inputs)
    x = BatchNormalization(name="mel_bn")(x)

    filters = BASE_FILTERS
    for block_idx in range(CNN_BLOCKS):
        residual = x
        kernel_a = 7 if block_idx < 2 else 5 if block_idx < 4 else 3
        dilation = 1 if block_idx < 2 else 2 if block_idx < 4 else 4 if block_idx < 5 else 8
        x = Conv1D(
            filters,
            kernel_size=kernel_a,
            dilation_rate=dilation,
            padding="same",
            activation="swish",
            name=f"conv_{block_idx + 1}_a",
        )(x)
        x = BatchNormalization(name=f"bn_{block_idx + 1}_a")(x)
        x = Conv1D(filters, kernel_size=3, padding="same", activation="swish", name=f"conv_{block_idx + 1}_b")(x)
        x = BatchNormalization(name=f"bn_{block_idx + 1}_b")(x)
        x = Conv1D(filters, kernel_size=1, padding="same", name=f"proj_{block_idx + 1}")(x)
        if residual.shape[-1] != filters:
            residual = Conv1D(filters, kernel_size=1, padding="same", name=f"res_proj_{block_idx + 1}")(residual)
        x = Add(name=f"res_add_{block_idx + 1}")([x, residual])
        x = BatchNormalization(name=f"bn_{block_idx + 1}_c")(x)
        if block_idx in {1, 3}:
            x = MaxPooling1D(pool_size=2, name=f"pool_{block_idx + 1}")(x)
        x = Dropout(DROPOUT, name=f"drop_{block_idx + 1}")(x)
        filters = min(filters * 2, BASE_FILTERS * 3)

    gap = GlobalAveragePooling1D(name="gap")(x)
    gmp = GlobalMaxPooling1D(name="gmp")(x)
    x = Concatenate(name="global_concat")([gap, gmp])
    x = LayerNormalization(name="shared_ln")(x)
    x = Dense(DENSE_UNITS, activation="swish", name="shared_dense")(x)
    x = Dropout(DROPOUT, name="shared_drop")(x)
    x = Dense(max(DENSE_UNITS // 2, 256), activation="swish", name="shared_dense_2")(x)
    x = Dropout(DROPOUT, name="shared_drop_2")(x)

    def branch_block(base: tf.Tensor, name: str, first_units: int, second_units: int) -> tf.Tensor:
        branch = Dense(first_units, activation="swish", name=f"{name}_dense_1")(base)
        branch = LayerNormalization(name=f"{name}_ln_1")(branch)
        branch = Dropout(DROPOUT, name=f"{name}_drop_1")(branch)
        branch = Dense(second_units, activation="swish", name=f"{name}_dense_2")(branch)
        branch = LayerNormalization(name=f"{name}_ln_2")(branch)
        branch = Dropout(DROPOUT, name=f"{name}_drop_2")(branch)
        return branch

    algo_branch = branch_block(x, "algo_branch", max(DENSE_UNITS // 2, 256), max(DENSE_UNITS // 3, 192))
    ratio_branch = branch_block(x, "ratio_branch", max(DENSE_UNITS // 3, 160), max(DENSE_UNITS // 4, 128))
    struct_branch = branch_block(x, "struct_branch", max(DENSE_UNITS // 2, 256), max(DENSE_UNITS // 3, 192))
    env_branch = branch_block(x, "env_branch", max(DENSE_UNITS // 2, 256), max(DENSE_UNITS // 3, 192))

    algo_joint = Concatenate(name="algo_joint_concat")([algo_branch, ratio_branch, struct_branch, env_branch, x])
    algo_joint = Dense(max(DENSE_UNITS, 384), activation="swish", name="algo_joint_dense")(algo_joint)
    algo_joint = LayerNormalization(name="algo_joint_ln")(algo_joint)
    algo_joint = Dropout(DROPOUT, name="algo_joint_drop")(algo_joint)
    algo_joint = Dense(max(DENSE_UNITS // 2, 256), activation="swish", name="algo_joint_dense_2")(algo_joint)
    algo_joint = LayerNormalization(name="algo_joint_ln_2")(algo_joint)
    algo_joint = Dropout(DROPOUT, name="algo_joint_drop_2")(algo_joint)

    def scalar_head(base: tf.Tensor, name: str) -> tf.Tensor:
        head = Dense(max(DENSE_UNITS // 4, 128), activation="swish", name=f"{name}_dense_1")(base)
        head = LayerNormalization(name=f"{name}_ln_1")(head)
        head = Dropout(DROPOUT, name=f"{name}_drop_1")(head)
        head = Dense(max(DENSE_UNITS // 5, 96), activation="swish", name=f"{name}_dense_2")(head)
        head = Dropout(DROPOUT, name=f"{name}_drop_2")(head)
        return Dense(1, activation=None, dtype="float32", name=name)(head)

    outputs = [Dense(n_algorithm_classes, activation="softmax", name="algorithm_head")(algo_joint)]
    for spec in TARGET_SPECS:
        branch = ratio_branch if spec["group"] == "ratio" else struct_branch if spec["group"] == "struct" else env_branch
        outputs.append(scalar_head(branch, spec["head"]))

    return Model(
        inputs=inputs,
        outputs=outputs,
        name="multitask_fm_big13_0_1",
    )


def main() -> None:
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"Missing parameters CSV: {PARAMS_PATH}")

    target_raw = pd.read_csv(PARAMS_PATH)
    if MAX_SAMPLES > 0 and MAX_SAMPLES < len(target_raw):
        target_raw = target_raw.iloc[:MAX_SAMPLES].copy()

    if "algorithm" not in target_raw.columns:
        raise ValueError("dataset_big13 must contain an `algorithm` column.")

    required_columns = {spec["column"] for spec in TARGET_SPECS}
    missing_columns = sorted(required_columns.difference(target_raw.columns))
    if missing_columns:
        raise ValueError(f"dataset_big13 is missing required columns: {missing_columns}")

    target_raw = target_raw.copy()
    target_raw["algorithm"] = target_raw["algorithm"].astype(str)
    target_raw["algorithm_idx"] = 0
    algorithm_classes = sorted(target_raw["algorithm"].unique().tolist())
    algorithm_map = {name: idx for idx, name in enumerate(algorithm_classes)}
    target_raw["algorithm_idx"] = target_raw["algorithm"].map(algorithm_map).astype(np.int32)

    for spec in TARGET_SPECS:
        target_raw[spec["head"]] = transform_series(target_raw[spec["column"]], spec["transform"])

    if "id" in target_raw.columns:
        sample_ids = target_raw["id"].astype(int).tolist()
    else:
        sample_ids = list(range(len(target_raw)))

    audio_store = load_audio_store(BASE_PATH, sample_ids)
    audio_len = int(audio_store.sample_len)

    dataset_indices = target_raw.index.to_numpy(dtype=np.int32)
    train_strata = make_stratify_key(target_raw)
    train_idx, test_idx = stratified_split_indices(
        dataset_indices,
        train_strata,
        test_size=1.0 - TRAIN_FRAC,
        random_state=RANDOM_STATE,
    )

    train_audio_indices = np.asarray(train_idx, dtype=np.int32)
    test_audio_indices = np.asarray(test_idx, dtype=np.int32)

    y_train_full = target_raw.loc[train_idx].reset_index(drop=True)
    y_test_full = target_raw.loc[test_idx].reset_index(drop=True)

    y_train_model = y_train_full.copy()
    fit_idx, val_idx = stratified_split_indices(
        np.arange(len(y_train_model), dtype=np.int32),
        make_stratify_key(y_train_model),
        test_size=VAL_FRAC,
        random_state=RANDOM_STATE,
    )

    fit_audio_indices = train_audio_indices[fit_idx]
    val_audio_indices = train_audio_indices[val_idx]

    scalers: dict[str, StandardScaler] = {}
    y_fit: dict[str, np.ndarray] = {"algorithm_head": y_train_model["algorithm_idx"].to_numpy(dtype=np.int32)[fit_idx]}
    y_val: dict[str, np.ndarray] = {"algorithm_head": y_train_model["algorithm_idx"].to_numpy(dtype=np.int32)[val_idx]}
    y_test: dict[str, np.ndarray] = {"algorithm_head": y_test_full["algorithm_idx"].to_numpy(dtype=np.int32)}
    transformed_test: dict[str, np.ndarray] = {}

    for spec in TARGET_SPECS:
        scaler = StandardScaler()
        scalers[spec["head"]] = scaler
        train_values = y_train_model[spec["head"]].to_numpy(dtype=np.float32).reshape(-1, 1)
        fit_values = scaler.fit_transform(train_values).astype(np.float32)
        val_values = scaler.transform(y_train_model[spec["head"]].to_numpy(dtype=np.float32)[val_idx].reshape(-1, 1)).astype(np.float32)
        test_values = scaler.transform(y_test_full[spec["head"]].to_numpy(dtype=np.float32).reshape(-1, 1)).astype(np.float32)
        y_fit[spec["head"]] = fit_values[fit_idx]
        y_val[spec["head"]] = val_values
        y_test[spec["head"]] = test_values
        transformed_test[spec["head"]] = test_values

    np.save(OUTPUT_DIR / "train_audio_indices.npy", train_audio_indices)
    np.save(OUTPUT_DIR / "val_audio_indices.npy", val_audio_indices)
    np.save(OUTPUT_DIR / "test_audio_indices.npy", test_audio_indices)
    y_train_full.to_csv(OUTPUT_DIR / "y_train_big13.csv", index=False)
    y_test_full.to_csv(OUTPUT_DIR / "y_test_big13.csv", index=False)

    train_seq = MultiTaskSequence(audio_store, fit_audio_indices, y_fit, BATCH_SIZE, shuffle=True)
    val_seq = MultiTaskSequence(audio_store, val_audio_indices, y_val, PRED_BATCH_SIZE, shuffle=False)
    test_seq = MultiTaskSequence(audio_store, test_audio_indices, y_test, PRED_BATCH_SIZE, shuffle=False)

    model = build_model(audio_len, len(algorithm_classes))
    losses = {"algorithm_head": sparse_categorical_focal_loss(gamma=1.15)}
    metrics = {"algorithm_head": ["sparse_categorical_accuracy"]}
    loss_weights = {"algorithm_head": 3.0}
    for spec in TARGET_SPECS:
        losses[spec["head"]] = tf.keras.losses.Huber(delta=1.0)
        metrics[spec["head"]] = ["mae"]
        loss_weights[spec["head"]] = float(spec["loss_weight"])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=losses,
        metrics=metrics,
        loss_weights=loss_weights,
    )

    resume_state = load_json_file(TRAIN_STATE_PATH) if RESUME_TRAINING else None
    resume_epoch = 0
    if RESUME_TRAINING and resume_state and CHECKPOINT_LATEST_WEIGHTS.exists():
        try:
            model.load_weights(str(CHECKPOINT_LATEST_WEIGHTS))
            resume_epoch = int(resume_state.get("last_completed_epoch", -1)) + 1
        except Exception as exc:
            print(f"Resume disabled: {exc}")
            resume_epoch = 0

    class ResumableCheckpointCallback(tf.keras.callbacks.Callback):
        def __init__(self, latest_weights_path: Path, best_weights_path: Path, state_path: Path, initial_state: dict | None = None):
            super().__init__()
            self.latest_weights_path = latest_weights_path
            self.best_weights_path = best_weights_path
            self.state_path = state_path
            self.last_completed_epoch = int(initial_state.get("last_completed_epoch", -1)) if initial_state else -1
            self.best_val_loss = float(initial_state.get("best_val_loss")) if initial_state and initial_state.get("best_val_loss") is not None else None
            self.best_val_loss_epoch = int(initial_state.get("best_val_loss_epoch")) if initial_state and initial_state.get("best_val_loss_epoch") is not None else None

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.last_completed_epoch = int(epoch)
            self.model.save_weights(self.latest_weights_path)
            val_loss = logs.get("val_loss")
            if val_loss is not None:
                try:
                    val_loss = float(val_loss)
                except (TypeError, ValueError):
                    val_loss = None
            if val_loss is not None and np.isfinite(val_loss):
                if self.best_val_loss is None or val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_val_loss_epoch = int(epoch)
                    self.model.save_weights(self.best_weights_path)
            atomic_json_dump(
                self.state_path,
                {
                    "model_name": MODEL_NAME,
                    "dataset": str(BASE_PATH),
                    "last_completed_epoch": self.last_completed_epoch,
                    "total_epochs_target": EPOCHS,
                    "best_val_loss": self.best_val_loss,
                    "best_val_loss_epoch": self.best_val_loss_epoch,
                },
            )

        def on_train_end(self, logs=None):
            self.model.save_weights(self.latest_weights_path)

    callbacks = [
        ResumableCheckpointCallback(CHECKPOINT_LATEST_WEIGHTS, CHECKPOINT_BEST_WEIGHTS, TRAIN_STATE_PATH, resume_state),
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(PATIENCE // 2, 2), min_lr=1e-6, verbose=1),
    ]

    print(
        "Runtime config: "
        f"dataset={BASE_PATH}, epochs={EPOCHS}, batch_size={BATCH_SIZE}, "
        f"base_filters={BASE_FILTERS}, blocks={CNN_BLOCKS}, mel_bins={N_MELS}, "
        f"resume_epoch={resume_epoch}"
    )

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS,
        initial_epoch=resume_epoch,
        callbacks=callbacks,
        verbose=FIT_VERBOSE,
    )

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(resume_epoch, resume_epoch + len(history_df), dtype=np.int32))
    history_df.to_csv(OUTPUT_DIR / "history.csv", index=False)

    if CHECKPOINT_BEST_WEIGHTS.exists():
        try:
            model.load_weights(str(CHECKPOINT_BEST_WEIGHTS))
        except Exception as exc:
            print(f"Best-weight reload skipped: {exc}")

    model_path = OUTPUT_DIR / f"{MODEL_NAME}.keras"
    try:
        model.save(model_path)
        print(f"Model saved to {model_path}")
    except Exception as exc:
        print(f"Model save skipped: {exc}")

    preds = model.predict(test_seq, verbose=0)
    algo_pred = np.asarray(preds[0], dtype=np.float32)
    algo_pred_idx = np.argmax(algo_pred, axis=1)
    algo_true = y_test_full["algorithm_idx"].to_numpy(dtype=np.int32)

    final_predictions = {
        "sample_id": test_audio_indices.astype(int),
        "algorithm_true": [algorithm_classes[i] for i in algo_true],
        "algorithm_pred": [algorithm_classes[i] for i in algo_pred_idx],
        "algorithm_prob": np.max(algo_pred, axis=1),
    }

    final_metrics = {
        "algorithm_accuracy": float(np.mean(algo_pred_idx == algo_true)),
        "algorithm_crossentropy": float(tf.keras.losses.sparse_categorical_crossentropy(algo_true, algo_pred).numpy().mean()),
    }

    pred_offset = 1
    for spec in TARGET_SPECS:
        scaler = scalers[spec["head"]]
        pred_scaled = np.asarray(preds[pred_offset], dtype=np.float32)
        pred_offset += 1
        pred_transformed = scaler.inverse_transform(pred_scaled).reshape(-1)
        pred_raw = inverse_transform_series(pred_transformed, spec["transform"])
        true_transformed = y_test_full[spec["head"]].to_numpy(dtype=np.float32)
        true_raw = y_test_full[spec["column"]].to_numpy(dtype=np.float32)
        true_raw_eval = np.asarray(true_raw, dtype=np.float32)

        final_predictions[f"{spec['head']}_true"] = true_raw_eval
        final_predictions[f"{spec['head']}_pred"] = pred_raw
        final_metrics[f"{spec['head']}_mae"] = float(mean_absolute_error(true_raw_eval, pred_raw))
        final_metrics[f"{spec['head']}_rmse"] = float(np.sqrt(mean_squared_error(true_raw_eval, pred_raw)))
        if spec["transform"] == "log2":
            final_metrics[f"{spec['head']}_log2_mae"] = float(mean_absolute_error(true_transformed, pred_transformed))

    predictions = pd.DataFrame(final_predictions)
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    plot_metric(history_df, "loss", "val_loss", "Loss", "train_loss.png")
    plot_metric(history_df, "algorithm_head_sparse_categorical_accuracy", "val_algorithm_head_sparse_categorical_accuracy", "Algorithm Accuracy", "train_algorithm_accuracy.png")
    if "ratio_log2_head_mae" in history_df.columns:
        plot_metric(history_df, "ratio_log2_head_mae", "val_ratio_log2_head_mae", "Ratio Log2 MAE", "train_ratio_log2_mae.png")
    if "freq_log2_head_mae" in history_df.columns:
        plot_metric(history_df, "freq_log2_head_mae", "val_freq_log2_head_mae", "Frequency Log2 MAE", "train_freq_log2_mae.png")
    if "env_mod_attack_head_mae" in history_df.columns:
        plot_metric(history_df, "env_mod_attack_head_mae", "val_env_mod_attack_head_mae", "Env Mod Attack MAE", "train_env_mod_attack_mae.png")
    if "env_car_attack_head_mae" in history_df.columns:
        plot_metric(history_df, "env_car_attack_head_mae", "val_env_car_attack_head_mae", "Env Car Attack MAE", "train_env_car_attack_mae.png")

    results = {
        "model_name": MODEL_NAME,
        "dataset": str(BASE_PATH),
        "runtime_config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "pred_batch_size": PRED_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "dropout": DROPOUT,
            "base_filters": BASE_FILTERS,
            "cnn_blocks": CNN_BLOCKS,
            "dense_units": DENSE_UNITS,
            "n_fft_small": N_FFT_SMALL,
            "n_fft_large": N_FFT_LARGE,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "mel_fmin": MEL_FMIN,
            "mel_fmax": MEL_FMAX,
            "mixed_precision": USE_MIXED_PRECISION,
            "resume_training": RESUME_TRAINING,
            "force_wav_audio": FORCE_WAV_AUDIO,
        },
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "algorithm_classes": algorithm_classes,
        "final_metrics": final_metrics,
        "target_specs": TARGET_SPECS,
        "history": history_df.to_dict(orient="list"),
        "test_prediction_preview": predictions.head(10).to_dict(orient="records"),
        "best_weights_path": str(CHECKPOINT_BEST_WEIGHTS),
        "latest_weights_path": str(CHECKPOINT_LATEST_WEIGHTS),
    }

    atomic_json_dump(OUTPUT_DIR / "results.json", results)
    with open(OUTPUT_DIR / "algorithm_map.json", "w", encoding="utf-8") as f:
        json.dump(algorithm_map, f, indent=2, ensure_ascii=False)
    for spec in TARGET_SPECS:
        joblib.dump(scalers[spec["head"]], OUTPUT_DIR / f"{spec['head']}_scaler.joblib")

    print(f"Results written to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
