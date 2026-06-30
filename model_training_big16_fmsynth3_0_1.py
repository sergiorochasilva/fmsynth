"""Hierarchical FM predictors for `dataset_big16`, version 0_1.

Architecture:
- Raw waveform input
- Compact multi-resolution log-mel front-end inside the model
- Residual 1D CNN backbone over the time axis of the stacked spectrogram features
- Separate training stages with explicit hierarchy:
  - `family`: coarse classifier for `series` vs `parallel`
  - `exact`: fine classifier for the 4 exact algorithms, conditioned on the true or predicted family

Data flow:
- Input: `dataset_big16/parameters.csv` plus the contiguous prefix of rendered `sample_*.wav`
- Output: separate trained weights, preprocessing artifacts, predictions, plots, and `results.json`
"""

from __future__ import annotations

import argparse
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
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
    Lambda,
    LayerNormalization,
    MaxPooling1D,
)
from keras.models import Model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from model_training_big13_fmsynth3_0_1 import (
    atomic_json_dump,
    inverse_transform_series,
    load_json_file,
    make_json_safe,
    make_stratify_key,
    normalize_audio_batch,
    sparse_categorical_focal_loss,
    stratified_split_indices,
    to_json_scalar,
    transform_series,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_PATH = Path(os.getenv("DATASET_PATH", "dataset_big16"))
MODEL_PREFIX = "model_training_big16_fmsynth3_0_1"
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "0"))
RANDOM_STATE = int(os.getenv("SEED", "42"))
TRAIN_FRAC = float(os.getenv("TRAIN_FRAC", "0.80"))
VAL_FRAC = float(os.getenv("VAL_FRAC", "0.15"))
BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "8"))
PRED_BATCH_SIZE = int(os.getenv("PRED_BATCH_SIZE", "12"))
EPOCHS = int(os.getenv("EPOCHS", "30"))
PATIENCE = int(os.getenv("PATIENCE", "5"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2.0e-4"))
DROPOUT = float(os.getenv("DROPOUT", "0.10"))
BASE_FILTERS = int(os.getenv("BASE_FILTERS", "40"))
CNN_BLOCKS = int(os.getenv("CNN_BLOCKS", "4"))
DENSE_UNITS = int(os.getenv("DENSE_UNITS", "192"))
EXACT_BRANCH_FILTERS = int(os.getenv("EXACT_BRANCH_FILTERS", "128"))
EXACT_BRANCH_BLOCKS = int(os.getenv("EXACT_BRANCH_BLOCKS", "4"))
EXACT_BRANCH_DENSE_UNITS = int(os.getenv("EXACT_BRANCH_DENSE_UNITS", "384"))
EXACT_BRANCH_DROPOUT = float(os.getenv("EXACT_BRANCH_DROPOUT", "0.06"))
USE_MIXED_PRECISION = os.getenv("MIXED_PRECISION", "0") == "1"
ENABLE_XLA = os.getenv("ENABLE_XLA", "0") == "1"
RESUME_TRAINING = os.getenv("RESUME_TRAINING", "1") == "1"
FIT_VERBOSE = int(os.getenv("FIT_VERBOSE", "1"))
TARGET_GROUP = os.getenv("TARGET_GROUP", "family").strip().lower()

N_FFT_SMALL = int(os.getenv("N_FFT_SMALL", "512"))
N_FFT_LARGE = int(os.getenv("N_FFT_LARGE", "1024"))
HOP_LENGTH = int(os.getenv("HOP_LENGTH", "512"))
N_MELS = int(os.getenv("N_MELS", "48"))
MEL_FMIN = float(os.getenv("MEL_FMIN", "30.0"))
MEL_FMAX = float(os.getenv("MEL_FMAX", "7600.0"))
EPS = 1e-6

GROUP_SPECS = {
    "family": [
        {"head": "algorithm_family_head", "column": "algorithm_family", "transform": "categorical", "loss_weight": 1.0, "group": "family"},
    ],
    "exact": [
        {"head": "algorithm_head", "column": "algorithm", "transform": "categorical", "loss_weight": 1.0, "group": "exact"},
    ],
}

if TARGET_GROUP not in GROUP_SPECS:
    raise ValueError(f"TARGET_GROUP inválido: {TARGET_GROUP}. Use one of {sorted(GROUP_SPECS)}")

TARGET_SPECS = GROUP_SPECS[TARGET_GROUP]
CONDITION_HEADS = {
    "exact": ["family_condition_input"],
}
MODEL_NAME = f"{MODEL_PREFIX}_{TARGET_GROUP}"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", f"{MODEL_PREFIX}_{TARGET_GROUP}"))
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_LATEST_WEIGHTS = CHECKPOINT_DIR / "latest.weights.h5"
CHECKPOINT_BEST_WEIGHTS = CHECKPOINT_DIR / "best.weights.h5"
TRAIN_STATE_PATH = CHECKPOINT_DIR / "training_state.json"
PARAMS_PATH = BASE_PATH / "parameters.csv"

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

if USE_MIXED_PRECISION:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
if not ENABLE_XLA:
    tf.config.optimizer.set_jit(False)


def discover_contiguous_prefix(base_path: Path) -> int:
    idx = 0
    while (base_path / f"sample_{idx}.wav").exists():
        idx += 1
    return idx


def load_audio_manifest(base_path: Path) -> dict | None:
    dataset_suffix = base_path.name.replace("dataset_", "")
    manifest_path = base_path / f"audio_{dataset_suffix}_manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


class ShardedAudioStore:
    def __init__(self, base_path: Path, manifest: dict):
        self.base_path = Path(base_path)
        self.sample_len = int(manifest["audio_sample_len"])
        self.shard_size = int(manifest["audio_shard_size"])
        self.total_rows = int(manifest["total_rows"])
        self.shards = manifest["shards"]
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.total_rows

    def _load_shard(self, shard_idx: int) -> np.ndarray:
        if shard_idx not in self._cache:
            shard_path = self.base_path / self.shards[shard_idx]["file"]
            self._cache[shard_idx] = np.load(shard_path, mmap_mode="r")
        return self._cache[shard_idx]

    def __getitem__(self, item):
        indices = np.asarray(item, dtype=np.int64).reshape(-1)
        batch = np.empty((indices.shape[0], self.sample_len), dtype=np.int16)
        for out_idx, sample_id in enumerate(indices):
            shard_idx = int(sample_id // self.shard_size)
            row_idx = int(sample_id % self.shard_size)
            shard = self._load_shard(shard_idx)
            batch[out_idx] = np.asarray(shard[row_idx], dtype=np.int16)
        return batch


def load_audio_store(base_path: Path):
    manifest = load_audio_manifest(base_path)
    if manifest is not None:
        return ShardedAudioStore(base_path, manifest)
    sample_ids = []
    idx = 0
    while (base_path / f"sample_{idx}.wav").exists():
        sample_ids.append(idx)
        idx += 1
    if not sample_ids:
        raise FileNotFoundError(f"No rendered audio found in {base_path}")
    samples = []
    for sample_id in sample_ids:
        wav_path = base_path / f"sample_{sample_id}.wav"
        signal, sr = sf.read(str(wav_path), dtype="int16", always_2d=False)
        if sr != 16000:
            raise ValueError(f"Unexpected sample rate in cached wav {wav_path}: {sr}")
        samples.append(np.asarray(signal, dtype=np.int16).reshape(-1))
    return np.asarray(samples, dtype=np.int16)


def compute_logmel_local(x, n_fft: int, mel_matrix: tf.Tensor):
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


def temporal_delta(feature: tf.Tensor) -> tf.Tensor:
    delta = feature[:, 1:, :] - feature[:, :-1, :]
    delta = tf.pad(delta, paddings=[[0, 0], [1, 0], [0, 0]])
    delta = delta - tf.reduce_mean(delta, axis=[1, 2], keepdims=True)
    return delta / (tf.math.reduce_std(delta, axis=[1, 2], keepdims=True) + EPS)


def logmel_frontend(x):
    log_mel_small = compute_logmel_local(x, N_FFT_SMALL, MEL_MATRIX_SMALL)
    log_mel_large = compute_logmel_local(x, N_FFT_LARGE, MEL_MATRIX_LARGE)
    delta_small = temporal_delta(log_mel_small)
    delta_large = temporal_delta(log_mel_large)
    return tf.concat([log_mel_small, log_mel_large, delta_small, delta_large], axis=-1)


def target_columns(target_specs: list[dict] | None = None) -> list[str]:
    target_specs = TARGET_SPECS if target_specs is None else target_specs
    return [spec["column"] for spec in target_specs]


def target_head_names(target_specs: list[dict] | None = None) -> list[str]:
    target_specs = TARGET_SPECS if target_specs is None else target_specs
    return [spec["head"] for spec in target_specs]


def build_model(input_len: int, n_classes: int, target_group: str | None = None, target_specs: list[dict] | None = None) -> Model:
    target_group = TARGET_GROUP if target_group is None else target_group
    audio_input = Input(shape=(input_len, 1), name="audio_input")
    x = Lambda(logmel_frontend, output_shape=(None, N_MELS * 4), name="logmel_frontend")(audio_input)
    x = BatchNormalization(name="mel_bn")(x)

    filters = BASE_FILTERS
    for block_idx in range(CNN_BLOCKS):
        residual = x
        kernel_size = 7 if block_idx == 0 else 5 if block_idx == 1 else 3
        dilation = 1 if block_idx < 2 else 2 if block_idx == 2 else 4
        x = Conv1D(filters, kernel_size=kernel_size, dilation_rate=dilation, padding="same", activation="swish", name=f"conv_{block_idx + 1}_a")(x)
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
        filters = min(filters + BASE_FILTERS, BASE_FILTERS * 3)

    shared_seq = x
    gap = GlobalAveragePooling1D(name="gap")(shared_seq)
    gmp = GlobalMaxPooling1D(name="gmp")(shared_seq)
    shared_dense = Concatenate(name="global_concat")([gap, gmp])
    shared_dense = LayerNormalization(name="shared_ln")(shared_dense)
    shared_dense = Dense(DENSE_UNITS, activation="swish", name="shared_dense")(shared_dense)
    shared_dense = Dropout(DROPOUT, name="shared_drop")(shared_dense)
    shared_dense = Dense(max(DENSE_UNITS // 2, 96), activation="swish", name="shared_dense_2")(shared_dense)
    shared_dense = Dropout(DROPOUT, name="shared_drop_2")(shared_dense)

    input_tensors = [audio_input]
    outputs = []
    if target_group == "family":
        family_branch = Dense(max(DENSE_UNITS // 2, 96), activation="swish", name="family_branch_dense")(shared_dense)
        family_branch = LayerNormalization(name="family_branch_ln")(family_branch)
        family_branch = Dropout(DROPOUT, name="family_branch_drop")(family_branch)
        outputs.append(Dense(n_classes, activation="softmax", name="algorithm_family_head")(family_branch))
    elif target_group == "exact":
        family_condition_input = Input(shape=(n_classes // 2,), name="family_condition_input")
        input_tensors.append(family_condition_input)
        family_condition = Dense(max(DENSE_UNITS // 4, 64), activation="swish", name="family_condition_dense")(family_condition_input)
        family_condition = LayerNormalization(name="family_condition_ln")(family_condition)
        exact_seq = Concatenate(name="exact_condition_concat")([shared_dense, family_condition])
        exact_seq = Dense(EXACT_BRANCH_DENSE_UNITS, activation="swish", name="exact_branch_dense_1")(exact_seq)
        exact_seq = LayerNormalization(name="exact_branch_ln_1")(exact_seq)
        exact_seq = Dropout(EXACT_BRANCH_DROPOUT, name="exact_branch_drop_1")(exact_seq)
        exact_seq = Dense(max(EXACT_BRANCH_DENSE_UNITS // 2, 128), activation="swish", name="exact_branch_dense_2")(exact_seq)
        exact_seq = LayerNormalization(name="exact_branch_ln_2")(exact_seq)
        exact_seq = Dropout(EXACT_BRANCH_DROPOUT, name="exact_branch_drop_2")(exact_seq)
        outputs.append(Dense(n_classes, activation="softmax", name="algorithm_head")(exact_seq))
    else:
        raise ValueError(f"Unsupported target group: {target_group}")

    return Model(inputs=input_tensors, outputs=outputs, name=f"hierarchical_fm_big16_{target_group}")


def group_stratify_key(frame: pd.DataFrame, target_group: str | None = None) -> pd.Series:
    target_group = TARGET_GROUP if target_group is None else target_group
    if target_group == "family":
        return frame["algorithm_family"].astype(str)
    if target_group == "exact":
        family_bucket = frame["algorithm_family"].astype(str)
        ratio_bucket = pd.qcut(frame["ratio_carrier"].rank(method="first"), q=4, labels=False, duplicates="drop")
        return family_bucket + "__" + frame["algorithm"].astype(str) + "__" + ratio_bucket.astype(str)
    raise ValueError(f"Unsupported target group for stratification: {target_group}")


def load_target_frame(max_samples: int | None = None) -> pd.DataFrame:
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"Missing parameters CSV: {PARAMS_PATH}")
    target_raw = pd.read_csv(PARAMS_PATH)
    if "id" not in target_raw.columns:
        target_raw = target_raw.reset_index().rename(columns={"index": "id"})
    if "algorithm_family" not in target_raw.columns:
        target_raw["algorithm_family"] = target_raw["algorithm"].map(
            lambda x: "series" if str(x) in {"series3", "series3_parallel2"} else "parallel"
        )
    manifest = load_audio_manifest(BASE_PATH)
    available_count = int(manifest["total_rows"]) if manifest and "total_rows" in manifest else discover_contiguous_prefix(BASE_PATH)
    if available_count <= 0:
        raise FileNotFoundError(f"No contiguous rendered prefix found in {BASE_PATH}")
    effective_max = MAX_SAMPLES if max_samples is None else int(max_samples)
    if effective_max > 0:
        available_count = min(available_count, effective_max)
    target_raw = target_raw.iloc[:available_count].copy()
    return target_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hierarchical FM predictors for dataset_big16.")
    parser.add_argument("--target-group", type=str, default=TARGET_GROUP, choices=sorted(GROUP_SPECS))
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    group = args.target_group
    output_dir = Path(os.getenv("OUTPUT_DIR", f"{MODEL_PREFIX}_{group}"))
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_raw = load_target_frame(args.max_samples)
    if "algorithm" not in target_raw.columns:
        raise ValueError("dataset_big16 must contain an `algorithm` column.")
    if "algorithm_family" not in target_raw.columns:
        raise ValueError("dataset_big16 must contain an `algorithm_family` column.")

    target_raw = target_raw.copy()
    target_raw["algorithm"] = target_raw["algorithm"].astype(str)
    target_raw["algorithm_family"] = target_raw["algorithm_family"].astype(str)
    exact_classes = sorted(target_raw["algorithm"].unique().tolist())
    exact_map = {name: idx for idx, name in enumerate(exact_classes)}
    target_raw["algorithm_idx"] = target_raw["algorithm"].map(exact_map).astype(np.int32)
    family_classes = sorted(target_raw["algorithm_family"].unique().tolist())
    family_map = {name: idx for idx, name in enumerate(family_classes)}
    target_raw["algorithm_family_idx"] = target_raw["algorithm_family"].map(family_map).astype(np.int32)

    audio_store = load_audio_store(BASE_PATH)
    sample_ids = target_raw["id"].astype(int).tolist()
    audio_len = int(getattr(audio_store, "sample_len", 0) or np.asarray(audio_store[sample_ids[:1]]).shape[1])

    dataset_indices = target_raw.index.to_numpy(dtype=np.int32)
    train_strata = group_stratify_key(target_raw, group)
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
        group_stratify_key(y_train_model, group),
        test_size=VAL_FRAC,
        random_state=RANDOM_STATE,
    )

    fit_audio_indices = train_audio_indices[fit_idx]
    val_audio_indices = train_audio_indices[val_idx]
    y_fit: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    cond_fit: dict[str, np.ndarray] = {}
    cond_val: dict[str, np.ndarray] = {}
    cond_test: dict[str, np.ndarray] = {}

    if group == "family":
        y_fit = y_train_model["algorithm_family_idx"].to_numpy(dtype=np.int32)[fit_idx]
        y_val = y_train_model["algorithm_family_idx"].to_numpy(dtype=np.int32)[val_idx]
        y_test = y_test_full["algorithm_family_idx"].to_numpy(dtype=np.int32)
    else:
        y_fit = y_train_model["algorithm_idx"].to_numpy(dtype=np.int32)[fit_idx]
        y_val = y_train_model["algorithm_idx"].to_numpy(dtype=np.int32)[val_idx]
        y_test = y_test_full["algorithm_idx"].to_numpy(dtype=np.int32)
        family_onehot = tf.keras.utils.to_categorical(target_raw["algorithm_family_idx"].to_numpy(dtype=np.int32), num_classes=len(family_classes)).astype(np.float32)
        cond_fit["family_condition_input"] = family_onehot[fit_idx]
        cond_val["family_condition_input"] = family_onehot[val_idx]
        cond_test["family_condition_input"] = family_onehot[test_idx]

    np.save(output_dir / "train_audio_indices.npy", train_audio_indices)
    np.save(output_dir / "val_audio_indices.npy", val_audio_indices)
    np.save(output_dir / "test_audio_indices.npy", test_audio_indices)
    y_train_full.to_csv(output_dir / f"y_train_{group}.csv", index=False)
    y_test_full.to_csv(output_dir / f"y_test_{group}.csv", index=False)

    class SplitSequence(tf.keras.utils.Sequence):
        def __init__(self, audio: np.ndarray, sample_indices: np.ndarray, y: np.ndarray, conditions: dict[str, np.ndarray], batch_size: int, shuffle: bool = True, return_y: bool = True):
            self.audio = audio
            self.sample_indices = np.asarray(sample_indices, dtype=np.int32)
            self.y = np.asarray(y, dtype=np.int32)
            self.conditions = conditions
            self.batch_size = int(max(batch_size, 1))
            self.shuffle = bool(shuffle)
            self.return_y = bool(return_y)
            self.indices = np.arange(self.sample_indices.shape[0], dtype=np.int32)
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.indices) / self.batch_size))

        def __getitem__(self, idx):
            start = idx * self.batch_size
            end = min(start + self.batch_size, len(self.indices))
            batch_ids = self.indices[start:end]
            audio_ids = self.sample_indices[batch_ids]
            audio_batch = normalize_audio_batch(self.audio[audio_ids]).reshape(len(audio_ids), audio_len, 1)
            if not self.return_y:
                if not self.conditions:
                    return audio_batch
                x_batch = {"audio_input": audio_batch}
                for name, values in self.conditions.items():
                    x_batch[name] = values[batch_ids]
                return x_batch
            y_batch = self.y[batch_ids]
            if not self.conditions:
                return audio_batch, y_batch
            x_batch = {"audio_input": audio_batch}
            for name, values in self.conditions.items():
                x_batch[name] = values[batch_ids]
            return x_batch, y_batch

        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.indices)

    train_seq = SplitSequence(audio_store, fit_audio_indices, y_fit, cond_fit, BATCH_SIZE, shuffle=True)
    val_seq = SplitSequence(audio_store, val_audio_indices, y_val, cond_val, PRED_BATCH_SIZE, shuffle=False)
    test_seq = SplitSequence(audio_store, test_audio_indices, y_test, cond_test, PRED_BATCH_SIZE, shuffle=False)

    n_classes = len(family_classes) if group == "family" else len(exact_classes)
    model = build_model(audio_len, n_classes, group)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["sparse_categorical_accuracy"],
    )

    if group == "exact":
        pretrain_weights = Path(os.getenv("PRETRAIN_WEIGHTS", f"{MODEL_PREFIX}_family/checkpoints/best.weights.h5"))
        if pretrain_weights.exists():
            try:
                model.load_weights(str(pretrain_weights), skip_mismatch=True)
                print(f"Loaded hierarchical pretrain weights from {pretrain_weights}")
            except Exception as exc:
                print(f"Pretrain load skipped: {exc}")

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
                    "target_group": group,
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
        f"dataset={BASE_PATH}, target_group={group}, available_samples={len(target_raw)}, epochs={EPOCHS}, batch_size={BATCH_SIZE}, "
        f"base_filters={BASE_FILTERS}, blocks={CNN_BLOCKS}, mel_bins={N_MELS}, resume_epoch={resume_epoch}"
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
    history_df.to_csv(output_dir / "history.csv", index=False)

    if CHECKPOINT_BEST_WEIGHTS.exists():
        try:
            model.load_weights(str(CHECKPOINT_BEST_WEIGHTS))
        except Exception as exc:
            print(f"Best-weight reload skipped: {exc}")

    model_path = output_dir / f"{MODEL_NAME}.keras"
    try:
        model.save(model_path)
        print(f"Model saved to {model_path}")
    except Exception as exc:
        print(f"Model save skipped: {exc}")

    preds_oracle = np.asarray(model.predict(test_seq, verbose=0), dtype=np.float32)
    final_metrics = {}
    final_predictions = {"sample_id": test_audio_indices.astype(int)}

    if group == "family":
        family_true = np.asarray(y_test, dtype=np.int32)
        family_pred_idx = np.argmax(preds_oracle, axis=1)
        final_predictions["algorithm_family_true"] = [family_classes[i] for i in family_true]
        final_predictions["algorithm_family_pred"] = [family_classes[i] for i in family_pred_idx]
        final_predictions["algorithm_family_prob"] = np.max(preds_oracle, axis=1)
        final_metrics["family_accuracy"] = float(np.mean(family_pred_idx == family_true))
        final_metrics["family_crossentropy"] = float(tf.keras.losses.sparse_categorical_crossentropy(family_true, preds_oracle).numpy().mean())
    else:
        exact_true = np.asarray(y_test, dtype=np.int32)
        exact_pred_idx_oracle = np.argmax(preds_oracle, axis=1)
        final_predictions["algorithm_true"] = [exact_classes[i] for i in exact_true]
        final_predictions["algorithm_pred_oracle"] = [exact_classes[i] for i in exact_pred_idx_oracle]
        final_predictions["algorithm_prob_oracle"] = np.max(preds_oracle, axis=1)
        final_metrics["algorithm_accuracy_oracle"] = float(np.mean(exact_pred_idx_oracle == exact_true))
        final_metrics["algorithm_crossentropy_oracle"] = float(tf.keras.losses.sparse_categorical_crossentropy(exact_true, preds_oracle).numpy().mean())

        family_weights_path = Path(f"{MODEL_PREFIX}_family") / "checkpoints" / "best.weights.h5"
        if family_weights_path.exists():
            try:
                family_model = build_model(audio_len, len(family_classes), "family")
                family_model.load_weights(str(family_weights_path))
                family_only_seq = SplitSequence(audio_store, test_audio_indices, y_test, {}, PRED_BATCH_SIZE, shuffle=False, return_y=False)
                family_preds = np.asarray(family_model.predict(family_only_seq, verbose=0), dtype=np.float32)
                predicted_family_idx = np.argmax(family_preds, axis=1)
                cascade_conditions = {"family_condition_input": tf.keras.utils.to_categorical(predicted_family_idx, num_classes=len(family_classes)).astype(np.float32)}
                cascade_seq = SplitSequence(audio_store, test_audio_indices, y_test, cascade_conditions, PRED_BATCH_SIZE, shuffle=False)
                cascade_preds = np.asarray(model.predict(cascade_seq, verbose=0), dtype=np.float32)
                cascade_idx = np.argmax(cascade_preds, axis=1)
                final_predictions["family_pred_cascade"] = [family_classes[i] for i in predicted_family_idx]
                final_predictions["algorithm_pred_cascade"] = [exact_classes[i] for i in cascade_idx]
                final_metrics["family_accuracy_cascade"] = float(np.mean(predicted_family_idx == np.asarray(y_test_full["algorithm_family_idx"], dtype=np.int32)))
                final_metrics["algorithm_accuracy_cascade"] = float(np.mean(cascade_idx == exact_true))
                final_metrics["algorithm_crossentropy_cascade"] = float(tf.keras.losses.sparse_categorical_crossentropy(exact_true, cascade_preds).numpy().mean())
            except Exception as exc:
                print(f"Cascade evaluation skipped: {exc}")

    pd.DataFrame(final_predictions).to_csv(output_dir / "predictions.csv", index=False)

    results = {
        "model_name": MODEL_NAME,
        "dataset": str(BASE_PATH),
        "target_group": group,
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
        },
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "available_samples": int(len(target_raw)),
        "exact_classes": exact_classes,
        "family_classes": family_classes,
        "final_metrics": final_metrics,
        "target_specs": make_json_safe(TARGET_SPECS),
    }

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe(results), f, indent=2, ensure_ascii=False)

    with open(output_dir / "algorithm_map.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe({"exact_classes": exact_classes, "exact_mapping": exact_map, "family_classes": family_classes, "family_mapping": family_map}), f, indent=2, ensure_ascii=False)

    print(f"Results written to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
