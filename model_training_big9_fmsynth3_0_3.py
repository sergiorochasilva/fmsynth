"""Spectral FM multitask model for `dataset_big9`, version 0_3.

Architecture:
- Raw waveform is converted into a mean log-magnitude spectrum per sample
- 1D residual CNN over the spectrum with small kernels and light downsampling
- Shared dense trunk with categorical heads for `algorithm` and `ratio_carrier`
- Regression heads over `log2(ratio_carrier)` and `log2(frequencia_base)`

Data flow:
- Input: `dataset_big9/parameters.csv` plus `audio_big9_manifest.json` shards or `sample_*.wav`
- Output: spectral cache, trained weights, preprocessing artifacts, predictions, plots, and `results.json`
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (
    Add,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MaxPooling1D,
)
from keras.models import Model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import mixed_precision

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_PATH = Path(os.getenv("DATASET_PATH", "dataset_big9"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "model_training_big9_fmsynth3_0_3"))
MODEL_NAME = "model_training_big9_fmsynth3_0_3"

MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "0"))
RANDOM_STATE = int(os.getenv("SEED", "42"))
TRAIN_FRAC = float(os.getenv("TRAIN_FRAC", "0.75"))
VAL_FRAC = float(os.getenv("VAL_FRAC", "0.20"))

BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "32"))
PRED_BATCH_SIZE = int(os.getenv("PRED_BATCH_SIZE", "64"))
EPOCHS = int(os.getenv("EPOCHS", "24"))
PATIENCE = int(os.getenv("PATIENCE", "4"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2.5e-4"))
DROPOUT = float(os.getenv("DROPOUT", "0.10"))
BASE_FILTERS = int(os.getenv("BASE_FILTERS", "48"))
DENSE_UNITS = int(os.getenv("DENSE_UNITS", "224"))
USE_MIXED_PRECISION = os.getenv("MIXED_PRECISION", "0") == "1"
ENABLE_XLA = os.getenv("ENABLE_XLA", "0") == "1"
RESUME_TRAINING = os.getenv("RESUME_TRAINING", "1") == "1"

FRAME_LENGTH = int(os.getenv("FRAME_LENGTH", "1024"))
FRAME_STEP = int(os.getenv("FRAME_STEP", "256"))
FFT_LENGTH = int(os.getenv("FFT_LENGTH", "1024"))
SPECTRAL_SIGNATURE = f"f{FRAME_LENGTH}_s{FRAME_STEP}_fft{FFT_LENGTH}"

AUDIO_MANIFEST_PATH = BASE_PATH / "audio_big9_manifest.json"
AUDIO_LEGACY_CACHE_PATH = BASE_PATH / "audio_big9_int16.npy"
PARAMS_PATH = BASE_PATH / "parameters.csv"
SPECTRAL_CACHE_PATH = OUTPUT_DIR / f"spectral_big9_mean_log_{SPECTRAL_SIGNATURE}.npy"
SPECTRAL_META_PATH = OUTPUT_DIR / f"spectral_big9_meta_{SPECTRAL_SIGNATURE}.json"
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
    if "algorithm" not in frame.columns or "ratio_idx" not in frame.columns:
        return None
    return pd.Series(
        [f"{algo}__{ratio}" for algo, ratio in zip(frame["algorithm"].astype(str), frame["ratio_idx"].astype(int))]
    )


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


def frame_signal(audio: np.ndarray, frame_length: int, frame_step: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono signal, got shape {audio.shape}")
    if len(audio) < frame_length:
        audio = np.pad(audio, (0, frame_length - len(audio)))

    n_frames = 1 + max(0, int(np.ceil((len(audio) - frame_length) / float(frame_step))))
    target_len = (n_frames - 1) * frame_step + frame_length
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))

    shape = (n_frames, frame_length)
    strides = (audio.strides[0] * frame_step, audio.strides[0])
    return np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)


def mean_log_spectrum(sample: np.ndarray) -> np.ndarray:
    sample = np.asarray(sample, dtype=np.float32).reshape(-1)
    sample = np.clip(sample / 32768.0, -1.0, 1.0)
    frames = frame_signal(sample, FRAME_LENGTH, FRAME_STEP)
    window = np.hanning(FRAME_LENGTH).astype(np.float32)
    windowed = frames * window[None, :]
    spectrum = np.fft.rfft(windowed, n=FFT_LENGTH, axis=1)
    magnitude = np.log1p(np.abs(spectrum).astype(np.float32)).mean(axis=0)
    return magnitude.astype(np.float32, copy=False)


def build_spectral_cache(audio_store, sample_indices: np.ndarray, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape[0] == len(sample_indices):
            return np.asarray(cached, dtype=np.float32)

    batch_size = int(os.getenv("SPECTRAL_CACHE_BATCH_SIZE", "64"))
    features = []
    for start in range(0, len(sample_indices), batch_size):
        end = min(start + batch_size, len(sample_indices))
        batch_ids = sample_indices[start:end]
        audio_batch = audio_store.get_batch(batch_ids)
        for sample in audio_batch:
            features.append(mean_log_spectrum(sample.squeeze(-1) * 32768.0))
    feature_matrix = np.asarray(features, dtype=np.float32)
    np.save(cache_path, feature_matrix)
    atomic_json_dump(
        SPECTRAL_META_PATH,
        {
            "frame_length": FRAME_LENGTH,
            "frame_step": FRAME_STEP,
            "fft_length": FFT_LENGTH,
            "feature_shape": list(feature_matrix.shape),
        },
    )
    return feature_matrix


class FeatureSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        features: np.ndarray,
        y: dict[str, np.ndarray],
        batch_size: int,
        shuffle: bool = True,
    ):
        super().__init__()
        self.features = np.asarray(features, dtype=np.float32)
        self.y = y
        self.batch_size = int(max(batch_size, 1))
        self.shuffle = bool(shuffle)
        self.indices = np.arange(self.features.shape[0], dtype=np.int32)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        batch_ids = self.indices[start:end]
        x_batch = self.features[batch_ids][..., np.newaxis]
        y_batch = {name: values[batch_ids] for name, values in self.y.items()}
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def residual_block(x, filters: int, dilation: int, block_name: str) -> tf.Tensor:
    shortcut = x
    if int(x.shape[-1]) != filters:
        shortcut = Conv1D(filters, kernel_size=1, padding="same", name=f"{block_name}_proj")(shortcut)
        shortcut = BatchNormalization(name=f"{block_name}_proj_bn")(shortcut)

    y = Conv1D(filters, kernel_size=3, padding="same", dilation_rate=dilation, activation="swish", name=f"{block_name}_conv1")(x)
    y = BatchNormalization(name=f"{block_name}_bn1")(y)
    y = Conv1D(filters, kernel_size=3, padding="same", dilation_rate=dilation, activation="swish", name=f"{block_name}_conv2")(y)
    y = BatchNormalization(name=f"{block_name}_bn2")(y)
    y = Add(name=f"{block_name}_add")([shortcut, y])
    y = tf.keras.layers.Activation("swish", name=f"{block_name}_act")(y)
    y = Dropout(DROPOUT, name=f"{block_name}_drop")(y)
    return y


def build_model(input_len: int, n_algorithm_classes: int, n_ratio_classes: int) -> Model:
    inputs = Input(shape=(input_len, 1), name="spectrum_input")
    x = BatchNormalization(name="input_bn")(inputs)
    x = Conv1D(BASE_FILTERS, kernel_size=5, padding="same", activation="swish", name="stem_conv")(x)
    x = BatchNormalization(name="stem_bn")(x)

    filters = BASE_FILTERS
    dilations = [1, 1, 2, 2, 4, 8]
    for block_idx, dilation in enumerate(dilations, start=1):
        x = residual_block(x, filters, dilation, block_name=f"res_{block_idx}")
        if block_idx in {2, 4}:
            x = Conv1D(
                min(filters * 2, BASE_FILTERS * 4),
                kernel_size=3,
                strides=2,
                padding="same",
                activation="swish",
                name=f"down_{block_idx}",
            )(x)
            x = BatchNormalization(name=f"down_{block_idx}_bn")(x)
            filters = min(filters * 2, BASE_FILTERS * 4)

    x = GlobalAveragePooling1D(name="gap")(x)
    x = LayerNormalization(name="shared_ln")(x)
    x = Dense(DENSE_UNITS, activation="swish", name="shared_dense")(x)
    x = Dropout(DROPOUT, name="shared_drop")(x)
    x = Dense(max(DENSE_UNITS // 2, 96), activation="swish", name="shared_dense_2")(x)

    algorithm_head = Dense(n_algorithm_classes, activation="softmax", name="algorithm_head")(x)
    ratio_class_head = Dense(n_ratio_classes, activation="softmax", name="ratio_class_head")(x)
    ratio_log2_head = Dense(1, activation=None, dtype="float32", name="ratio_log2_head")(x)
    freq_log2_head = Dense(1, activation=None, dtype="float32", name="freq_log2_head")(x)

    return Model(
        inputs=inputs,
        outputs=[algorithm_head, ratio_class_head, ratio_log2_head, freq_log2_head],
        name="spectral_rawwave_multitask_big9_0_3",
    )


def main() -> None:
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"Missing parameters CSV: {PARAMS_PATH}")

    target_raw = pd.read_csv(PARAMS_PATH)
    if MAX_SAMPLES > 0 and MAX_SAMPLES < len(target_raw):
        target_raw = target_raw.iloc[:MAX_SAMPLES].copy()

    if "id" in target_raw.columns:
        sample_ids = target_raw["id"].astype(int).tolist()
    else:
        sample_ids = list(range(len(target_raw)))

    required_columns = {"algorithm", "frequencia_base", "ratio_carrier"}
    if not required_columns.issubset(target_raw.columns):
        raise ValueError("dataset_big9 must contain `algorithm`, `frequencia_base`, and `ratio_carrier` columns.")

    target_raw = target_raw.copy()
    target_raw["algorithm"] = target_raw["algorithm"].astype(str)
    target_raw["frequencia_base"] = target_raw["frequencia_base"].astype(np.float32)
    target_raw["ratio_carrier"] = np.round(target_raw["ratio_carrier"].astype(np.float32), 4)
    target_raw["freq_log2"] = np.log2(target_raw["frequencia_base"])
    target_raw["ratio_log2"] = np.log2(target_raw["ratio_carrier"])

    ratio_classes = build_ratio_classes(target_raw["ratio_carrier"].to_numpy(dtype=np.float32))
    target_raw["ratio_idx"] = assign_ratio_classes(target_raw["ratio_carrier"].to_numpy(dtype=np.float32), ratio_classes)

    audio_store = load_audio_store(BASE_PATH, sample_ids)

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

    algorithm_classes = sorted(target_raw["algorithm"].unique().tolist())
    algorithm_map = {name: idx for idx, name in enumerate(algorithm_classes)}
    ratio_map = {float(value): idx for idx, value in enumerate(ratio_classes.tolist())}
    y_train_full["algorithm_idx"] = y_train_full["algorithm"].map(algorithm_map).astype(np.int32)
    y_test_full["algorithm_idx"] = y_test_full["algorithm"].map(algorithm_map).astype(np.int32)

    features_all = build_spectral_cache(audio_store, dataset_indices, SPECTRAL_CACHE_PATH)

    y_train_model = y_train_full.copy()
    fit_idx, val_idx = stratified_split_indices(
        np.arange(len(y_train_model), dtype=np.int32),
        make_stratify_key(y_train_model),
        test_size=VAL_FRAC,
        random_state=RANDOM_STATE,
    )

    feature_scaler = StandardScaler()
    features_train = features_all[train_idx]
    features_test = features_all[test_idx]
    features_train_scaled = feature_scaler.fit_transform(features_train).astype(np.float32)
    features_test_scaled = feature_scaler.transform(features_test).astype(np.float32)

    features_fit = features_train_scaled[fit_idx]
    features_val = features_train_scaled[val_idx]
    feature_len = int(features_fit.shape[1])

    freq_scaler = StandardScaler()
    ratio_scaler = StandardScaler()
    y_freq_train = freq_scaler.fit_transform(y_train_model["freq_log2"].to_numpy(dtype=np.float32).reshape(-1, 1)).astype(np.float32)
    y_ratio_train = ratio_scaler.fit_transform(y_train_model["ratio_log2"].to_numpy(dtype=np.float32).reshape(-1, 1)).astype(np.float32)
    y_freq_fit = y_freq_train[fit_idx]
    y_freq_val = y_freq_train[val_idx]
    y_ratio_fit = y_ratio_train[fit_idx]
    y_ratio_val = y_ratio_train[val_idx]

    y_fit = {
        "algorithm_head": y_train_model["algorithm_idx"].to_numpy(dtype=np.int32)[fit_idx],
        "ratio_class_head": y_train_model["ratio_idx"].to_numpy(dtype=np.int32)[fit_idx],
        "ratio_log2_head": y_ratio_fit,
        "freq_log2_head": y_freq_fit,
    }
    y_val = {
        "algorithm_head": y_train_model["algorithm_idx"].to_numpy(dtype=np.int32)[val_idx],
        "ratio_class_head": y_train_model["ratio_idx"].to_numpy(dtype=np.int32)[val_idx],
        "ratio_log2_head": y_ratio_val,
        "freq_log2_head": y_freq_val,
    }

    y_test_freq_log2 = freq_scaler.transform(y_test_full["freq_log2"].to_numpy(dtype=np.float32).reshape(-1, 1)).astype(np.float32)
    y_test_ratio_log2 = ratio_scaler.transform(y_test_full["ratio_log2"].to_numpy(dtype=np.float32).reshape(-1, 1)).astype(np.float32)

    np.save(OUTPUT_DIR / "train_audio_indices.npy", train_audio_indices)
    np.save(OUTPUT_DIR / "test_audio_indices.npy", test_audio_indices)
    y_train_full.to_csv(OUTPUT_DIR / "y_train_big9.csv", index=False)
    y_test_full.to_csv(OUTPUT_DIR / "y_test_big9.csv", index=False)
    np.save(OUTPUT_DIR / "ratio_classes.npy", ratio_classes.astype(np.float32))

    train_seq = FeatureSequence(features_fit, y_fit, BATCH_SIZE, shuffle=True)
    val_seq = FeatureSequence(features_val, y_val, PRED_BATCH_SIZE, shuffle=False)
    test_seq = FeatureSequence(
        features_test_scaled,
        {
            "algorithm_head": y_test_full["algorithm_idx"].to_numpy(dtype=np.int32),
            "ratio_class_head": y_test_full["ratio_idx"].to_numpy(dtype=np.int32),
            "ratio_log2_head": y_test_ratio_log2.astype(np.float32),
            "freq_log2_head": y_test_freq_log2.astype(np.float32),
        },
        PRED_BATCH_SIZE,
        shuffle=False,
    )

    model = build_model(feature_len, len(algorithm_classes), len(ratio_classes))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            "algorithm_head": tf.keras.losses.SparseCategoricalCrossentropy(),
            "ratio_class_head": tf.keras.losses.SparseCategoricalCrossentropy(),
            "ratio_log2_head": tf.keras.losses.Huber(delta=1.0),
            "freq_log2_head": tf.keras.losses.Huber(delta=1.0),
        },
        metrics={
            "algorithm_head": ["sparse_categorical_accuracy"],
            "ratio_class_head": ["sparse_categorical_accuracy"],
            "ratio_log2_head": ["mae", "mse"],
            "freq_log2_head": ["mae", "mse"],
        },
        loss_weights={
            "algorithm_head": 1.0,
            "ratio_class_head": 0.45,
            "ratio_log2_head": 1.0,
            "freq_log2_head": 1.05,
        },
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
        f"frame_length={FRAME_LENGTH}, fft_length={FFT_LENGTH}, base_filters={BASE_FILTERS}, "
        f"ratio_class_count={len(ratio_classes)}, resume_epoch={resume_epoch}"
    )

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS,
        initial_epoch=resume_epoch,
        callbacks=callbacks,
        verbose=1,
    )

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(resume_epoch, resume_epoch + len(history_df), dtype=np.int32))
    history_df.to_csv(OUTPUT_DIR / "history.csv", index=False)

    if CHECKPOINT_BEST_WEIGHTS.exists():
        model.load_weights(str(CHECKPOINT_BEST_WEIGHTS))

    preds = model.predict(test_seq, verbose=0)
    algo_pred = preds[0]
    ratio_class_pred = preds[1]
    ratio_log2_pred_scaled = preds[2]
    freq_log2_pred_scaled = preds[3]

    algo_pred_idx = np.argmax(algo_pred, axis=1)
    ratio_class_pred_idx = np.argmax(ratio_class_pred, axis=1)

    algo_true = y_test_full["algorithm_idx"].to_numpy(dtype=np.int32)
    ratio_true_idx = y_test_full["ratio_idx"].to_numpy(dtype=np.int32)
    ratio_true_value = y_test_full["ratio_carrier"].to_numpy(dtype=np.float32)
    ratio_log2_true = y_test_full["ratio_log2"].to_numpy(dtype=np.float32)
    ratio_log2_pred = ratio_scaler.inverse_transform(np.asarray(ratio_log2_pred_scaled, dtype=np.float32)).reshape(-1)
    ratio_pred_value = np.power(2.0, ratio_log2_pred)
    ratio_snap_idx = np.abs(ratio_classes[None, :] - ratio_pred_value[:, None]).argmin(axis=1)
    freq_log2_true = y_test_full["freq_log2"].to_numpy(dtype=np.float32)
    freq_log2_pred = freq_scaler.inverse_transform(np.asarray(freq_log2_pred_scaled, dtype=np.float32)).reshape(-1)

    freq_hz_true = np.power(2.0, freq_log2_true)
    freq_hz_pred = np.power(2.0, freq_log2_pred)
    carrier_hz_true = freq_hz_true * ratio_true_value
    carrier_hz_pred = freq_hz_pred * ratio_pred_value

    algo_acc = float(np.mean(algo_pred_idx == algo_true))
    ratio_class_acc = float(np.mean(ratio_class_pred_idx == ratio_true_idx))
    ratio_snap_acc = float(np.mean(ratio_snap_idx == ratio_true_idx))
    algo_ce = float(tf.keras.losses.sparse_categorical_crossentropy(algo_true, algo_pred).numpy().mean())
    ratio_mae = float(mean_absolute_error(ratio_true_value, ratio_pred_value))
    ratio_log2_mae = float(mean_absolute_error(ratio_log2_true, ratio_log2_pred))

    freq_mae_hz = float(mean_absolute_error(freq_hz_true, freq_hz_pred))
    freq_mse_hz = float(mean_squared_error(freq_hz_true, freq_hz_pred))
    freq_rmse_hz = float(np.sqrt(freq_mse_hz))
    freq_mae_cents = float(np.mean(np.abs(1200.0 * (freq_log2_pred - freq_log2_true))))
    freq_log2_mae = float(mean_absolute_error(freq_log2_true, freq_log2_pred))

    carrier_mae_hz = float(mean_absolute_error(carrier_hz_true, carrier_hz_pred))
    carrier_mse_hz = float(mean_squared_error(carrier_hz_true, carrier_hz_pred))
    carrier_rmse_hz = float(np.sqrt(carrier_mse_hz))

    predictions = pd.DataFrame(
        {
            "sample_id": test_audio_indices.astype(int),
            "algorithm_true": [algorithm_classes[i] for i in algo_true],
            "algorithm_pred": [algorithm_classes[i] for i in algo_pred_idx],
            "algorithm_prob": np.max(algo_pred, axis=1),
            "ratio_true_idx": ratio_true_idx,
            "ratio_pred_idx": ratio_class_pred_idx,
            "ratio_carrier_true": ratio_true_value,
            "ratio_carrier_pred": ratio_pred_value,
            "ratio_carrier_snap": ratio_classes[ratio_snap_idx],
            "ratio_log2_true": ratio_log2_true,
            "ratio_log2_pred": ratio_log2_pred,
            "frequencia_base_true": freq_hz_true,
            "frequencia_base_pred": freq_hz_pred,
            "carrier_frequency_true": carrier_hz_true,
            "carrier_frequency_pred": carrier_hz_pred,
            "freq_log2_true": freq_log2_true,
            "freq_log2_pred": freq_log2_pred,
        }
    )
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    plot_metric(history_df, "loss", "val_loss", "Loss", "train_loss.png")
    plot_metric(history_df, "algorithm_head_sparse_categorical_accuracy", "val_algorithm_head_sparse_categorical_accuracy", "Algorithm Accuracy", "train_algorithm_accuracy.png")
    plot_metric(history_df, "ratio_class_head_sparse_categorical_accuracy", "val_ratio_class_head_sparse_categorical_accuracy", "Ratio Class Accuracy", "train_ratio_class_accuracy.png")
    plot_metric(history_df, "ratio_log2_head_mae", "val_ratio_log2_head_mae", "Ratio Log2 MAE", "train_ratio_log2_mae.png")
    plot_metric(history_df, "freq_log2_head_mae", "val_freq_log2_head_mae", "Frequency Log2 MAE", "train_freq_log2_mae.png")

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
            "dense_units": DENSE_UNITS,
            "frame_length": FRAME_LENGTH,
            "frame_step": FRAME_STEP,
            "fft_length": FFT_LENGTH,
            "ratio_class_count": int(len(ratio_classes)),
            "mixed_precision": USE_MIXED_PRECISION,
            "resume_training": RESUME_TRAINING,
        },
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "algorithm_classes": algorithm_classes,
        "ratio_classes": ratio_classes.tolist(),
        "final_metrics": {
            "algorithm_accuracy": algo_acc,
            "algorithm_crossentropy": algo_ce,
            "ratio_class_accuracy": ratio_class_acc,
            "ratio_snap_accuracy": ratio_snap_acc,
            "ratio_mae": ratio_mae,
            "ratio_log2_mae": ratio_log2_mae,
            "freq_mae_hz": freq_mae_hz,
            "freq_rmse_hz": freq_rmse_hz,
            "freq_mae_cents": freq_mae_cents,
            "freq_log2_mae": freq_log2_mae,
            "carrier_mae_hz": carrier_mae_hz,
            "carrier_rmse_hz": carrier_rmse_hz,
        },
        "history": history_df.to_dict(orient="list"),
        "test_prediction_preview": predictions.head(10).to_dict(orient="records"),
        "freq_scaler_mean": freq_scaler.mean_.tolist(),
        "freq_scaler_scale": freq_scaler.scale_.tolist(),
        "ratio_scaler_mean": ratio_scaler.mean_.tolist(),
        "ratio_scaler_scale": ratio_scaler.scale_.tolist(),
        "feature_scaler_mean_preview": feature_scaler.mean_[:10].tolist(),
        "feature_scaler_scale_preview": feature_scaler.scale_[:10].tolist(),
        "best_weights_path": str(CHECKPOINT_BEST_WEIGHTS),
        "latest_weights_path": str(CHECKPOINT_LATEST_WEIGHTS),
        "spectral_cache_path": str(SPECTRAL_CACHE_PATH),
    }

    atomic_json_dump(OUTPUT_DIR / "results.json", results)
    joblib.dump(freq_scaler, OUTPUT_DIR / "freq_log2_scaler.joblib")
    joblib.dump(ratio_scaler, OUTPUT_DIR / "ratio_log2_scaler.joblib")
    joblib.dump(feature_scaler, OUTPUT_DIR / "feature_scaler.joblib")
    with open(OUTPUT_DIR / "algorithm_map.json", "w", encoding="utf-8") as f:
        json.dump(algorithm_map, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_DIR / "ratio_map.json", "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in ratio_map.items()}, f, indent=2, ensure_ascii=False)

    print(f"Results written to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
