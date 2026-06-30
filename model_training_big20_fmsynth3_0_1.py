"""Hierarchical algorithm classifier for `dataset_big19`, version 0_1.

Architecture:
- Raw waveform input
- Differentiable multi-resolution log-mel front-end inside the model
- Residual 1D CNN backbone over the time axis of stacked spectrogram features
- Explicit hierarchy for `algorithm`:
  - coarse `algorithm_family` classifier
  - teacher-forced family-conditioned exact heads for `series` and `parallel`

Data flow:
- Input: `dataset_big19/parameters.csv` plus the contiguous prefix of rendered audio
- Output: trained weights, predictions, learning curves, and `results.json`
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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

from model_training_big13_fmsynth3_0_1 import (
    atomic_json_dump,
    load_json_file,
    make_json_safe,
    normalize_audio_batch,
    stratified_split_indices,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_PATH = Path(os.getenv("DATASET_PATH", "dataset_big19"))
MODEL_NAME = "model_training_big20_fmsynth3_0_1"
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "0"))
RANDOM_STATE = int(os.getenv("SEED", "42"))
TRAIN_FRAC = float(os.getenv("TRAIN_FRAC", "0.80"))
VAL_FRAC = float(os.getenv("VAL_FRAC", "0.15"))
BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "8"))
PRED_BATCH_SIZE = int(os.getenv("PRED_BATCH_SIZE", "12"))
EPOCHS = int(os.getenv("EPOCHS", "40"))
PATIENCE = int(os.getenv("PATIENCE", "6"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2.0e-4"))
DROPOUT = float(os.getenv("DROPOUT", "0.08"))
BASE_FILTERS = int(os.getenv("BASE_FILTERS", "56"))
CNN_BLOCKS = int(os.getenv("CNN_BLOCKS", "4"))
DENSE_UNITS = int(os.getenv("DENSE_UNITS", "320"))
EXACT_BRANCH_DENSE_UNITS = int(os.getenv("EXACT_BRANCH_DENSE_UNITS", "256"))
EXACT_BRANCH_DROPOUT = float(os.getenv("EXACT_BRANCH_DROPOUT", "0.06"))
USE_MIXED_PRECISION = os.getenv("MIXED_PRECISION", "0") == "1"
ENABLE_XLA = os.getenv("ENABLE_XLA", "0") == "1"
RESUME_TRAINING = os.getenv("RESUME_TRAINING", "1") == "1"
FIT_VERBOSE = int(os.getenv("FIT_VERBOSE", "1"))

N_FFT_SMALL = int(os.getenv("N_FFT_SMALL", "512"))
N_FFT_LARGE = int(os.getenv("N_FFT_LARGE", "1024"))
HOP_LENGTH = int(os.getenv("HOP_LENGTH", "512"))
N_MELS = int(os.getenv("N_MELS", "48"))
MEL_FMIN = float(os.getenv("MEL_FMIN", "30.0"))
MEL_FMAX = float(os.getenv("MEL_FMAX", "7600.0"))
EPS = 1e-6

SERIES_FAMILY = "series"
PARALLEL_FAMILY = "parallel"
ALGORITHMS = [
    "series3",
    "series3_parallel2",
    "parallel5",
    "series2x2_parallel1",
]
FAMILY_TO_ALGOS = {
    SERIES_FAMILY: ["series3", "series3_parallel2"],
    PARALLEL_FAMILY: ["parallel5", "series2x2_parallel1"],
}

PARAMS_PATH = BASE_PATH / "parameters.csv"
CHECKPOINT_DIR = Path(os.getenv("OUTPUT_DIR", MODEL_NAME)) / "checkpoints"
OUTPUT_DIR = CHECKPOINT_DIR.parent
CHECKPOINT_LATEST_WEIGHTS = CHECKPOINT_DIR / "latest.weights.h5"
CHECKPOINT_BEST_WEIGHTS = CHECKPOINT_DIR / "best.weights.h5"
TRAIN_STATE_PATH = CHECKPOINT_DIR / "training_state.json"


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


def discover_contiguous_prefix(base_path: Path) -> int:
    idx = 0
    while (base_path / f"sample_{idx}.wav").exists():
        idx += 1
    return idx


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


if USE_MIXED_PRECISION:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
if not ENABLE_XLA:
    tf.config.optimizer.set_jit(False)


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


def algorithm_family(algorithm: str) -> str:
    if algorithm in FAMILY_TO_ALGOS[SERIES_FAMILY]:
        return SERIES_FAMILY
    if algorithm in FAMILY_TO_ALGOS[PARALLEL_FAMILY]:
        return PARALLEL_FAMILY
    raise ValueError(f"Unknown algorithm family for {algorithm}")


def load_target_frame(max_samples: int | None = None) -> pd.DataFrame:
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"Missing parameters CSV: {PARAMS_PATH}")
    target_raw = pd.read_csv(PARAMS_PATH)
    if "id" not in target_raw.columns:
        target_raw = target_raw.reset_index().rename(columns={"index": "id"})
    if "algorithm" not in target_raw.columns:
        raise ValueError("dataset_big19 must contain an `algorithm` column.")
    if "algorithm_family" not in target_raw.columns:
        target_raw["algorithm_family"] = target_raw["algorithm"].map(algorithm_family)
    target_raw = target_raw.copy()
    target_raw["algorithm"] = target_raw["algorithm"].astype(str)
    target_raw["algorithm_family"] = target_raw["algorithm_family"].astype(str)
    manifest = load_audio_manifest(BASE_PATH)
    available_count = int(manifest["total_rows"]) if manifest and "total_rows" in manifest else discover_contiguous_prefix(BASE_PATH)
    if available_count <= 0:
        raise FileNotFoundError(f"No contiguous rendered prefix found in {BASE_PATH}")
    effective_max = MAX_SAMPLES if max_samples is None else int(max_samples)
    if effective_max > 0:
        available_count = min(available_count, effective_max)
    return target_raw.iloc[:available_count].copy()


def group_stratify_key(frame: pd.DataFrame, max_classes: int | None = None) -> pd.Series:
    family = frame["algorithm_family"].astype(str)
    algorithm = frame["algorithm"].astype(str)
    ratio_bucket = pd.qcut(
        frame["ratio_carrier"].rank(method="first"),
        q=min(4, frame["ratio_carrier"].nunique()),
        labels=False,
        duplicates="drop",
    )
    freq_bucket = pd.qcut(frame["frequencia_base"].rank(method="first"), q=4, labels=False, duplicates="drop")
    candidates = [
        family + "__" + algorithm + "__" + ratio_bucket.astype(str) + "__" + freq_bucket.astype(str),
        family + "__" + algorithm + "__" + ratio_bucket.astype(str),
        family + "__" + algorithm,
        family,
    ]
    for candidate in candidates:
        counts = candidate.value_counts()
        if counts.min() >= 2 and (max_classes is None or len(counts) <= max_classes):
            return candidate
    if max_classes is not None:
        for candidate in candidates:
            if candidate.value_counts().min() >= 2:
                return candidate
    return candidates[-1]


def build_model(input_len: int, n_family_classes: int) -> Model:
    audio_input = Input(shape=(input_len, 1), name="audio_input")
    family_condition_input = Input(shape=(n_family_classes,), name="family_condition_input")

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

    shared_gap = GlobalAveragePooling1D(name="shared_gap")(x)
    shared_gmp = GlobalMaxPooling1D(name="shared_gmp")(x)
    shared = Concatenate(name="shared_concat")([shared_gap, shared_gmp])
    shared = LayerNormalization(name="shared_ln")(shared)
    shared = Dense(DENSE_UNITS, activation="swish", name="shared_dense_1")(shared)
    shared = Dropout(DROPOUT, name="shared_drop_1")(shared)
    shared = Dense(max(DENSE_UNITS // 2, 128), activation="swish", name="shared_dense_2")(shared)
    shared = LayerNormalization(name="shared_ln_2")(shared)
    shared = Dropout(DROPOUT, name="shared_drop_2")(shared)

    family_branch = Dense(max(DENSE_UNITS // 2, 128), activation="swish", name="family_dense_1")(shared)
    family_branch = LayerNormalization(name="family_ln_1")(family_branch)
    family_branch = Dropout(DROPOUT, name="family_drop_1")(family_branch)
    family_output = Dense(n_family_classes, activation="softmax", name="algorithm_family_head")(family_branch)

    family_embed = Dense(max(DENSE_UNITS // 4, 64), activation="swish", name="family_condition_dense")(family_condition_input)
    family_embed = LayerNormalization(name="family_condition_ln")(family_embed)
    family_exact_input = Concatenate(name="family_exact_concat")([shared, family_embed])

    series_branch = Dense(EXACT_BRANCH_DENSE_UNITS, activation="swish", name="series_exact_dense_1")(family_exact_input)
    series_branch = LayerNormalization(name="series_exact_ln_1")(series_branch)
    series_branch = Dropout(EXACT_BRANCH_DROPOUT, name="series_exact_drop_1")(series_branch)
    series_branch = Dense(max(EXACT_BRANCH_DENSE_UNITS // 2, 96), activation="swish", name="series_exact_dense_2")(series_branch)
    series_branch = LayerNormalization(name="series_exact_ln_2")(series_branch)
    series_branch = Dropout(EXACT_BRANCH_DROPOUT, name="series_exact_drop_2")(series_branch)
    series_output = Dense(len(FAMILY_TO_ALGOS[SERIES_FAMILY]), activation="softmax", name="series_exact_head")(series_branch)

    parallel_branch = Dense(EXACT_BRANCH_DENSE_UNITS, activation="swish", name="parallel_exact_dense_1")(family_exact_input)
    parallel_branch = LayerNormalization(name="parallel_exact_ln_1")(parallel_branch)
    parallel_branch = Dropout(EXACT_BRANCH_DROPOUT, name="parallel_exact_drop_1")(parallel_branch)
    parallel_branch = Dense(max(EXACT_BRANCH_DENSE_UNITS // 2, 96), activation="swish", name="parallel_exact_dense_2")(parallel_branch)
    parallel_branch = LayerNormalization(name="parallel_exact_ln_2")(parallel_branch)
    parallel_branch = Dropout(EXACT_BRANCH_DROPOUT, name="parallel_exact_drop_2")(parallel_branch)
    parallel_output = Dense(len(FAMILY_TO_ALGOS[PARALLEL_FAMILY]), activation="softmax", name="parallel_exact_head")(parallel_branch)

    return Model(
        inputs=[audio_input, family_condition_input],
        outputs=[family_output, series_output, parallel_output],
        name="hierarchical_algorithm_classifier_big20",
    )


def build_categorical_mappings(target_raw: pd.DataFrame) -> dict:
    family_classes = sorted(target_raw["algorithm_family"].unique().tolist())
    family_map = {name: idx for idx, name in enumerate(family_classes)}
    family_idx = target_raw["algorithm_family"].map(family_map).astype(np.int32)

    series_algorithms = FAMILY_TO_ALGOS[SERIES_FAMILY]
    parallel_algorithms = FAMILY_TO_ALGOS[PARALLEL_FAMILY]
    series_map = {name: idx for idx, name in enumerate(series_algorithms)}
    parallel_map = {name: idx for idx, name in enumerate(parallel_algorithms)}

    return {
        "family_classes": family_classes,
        "family_map": family_map,
        "family_idx": family_idx,
        "series_algorithms": series_algorithms,
        "parallel_algorithms": parallel_algorithms,
        "series_map": series_map,
        "parallel_map": parallel_map,
        "series_family_idx": family_map[SERIES_FAMILY],
        "parallel_family_idx": family_map[PARALLEL_FAMILY],
    }


class HierarchicalSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        audio: np.ndarray,
        sample_indices: np.ndarray,
        family_idx: np.ndarray,
        series_exact_idx: np.ndarray,
        parallel_exact_idx: np.ndarray,
        batch_size: int,
        n_family_classes: int,
        series_family_idx: int,
        parallel_family_idx: int,
        shuffle: bool = True,
    ):
        super().__init__()
        self.audio = audio
        self.sample_indices = np.asarray(sample_indices, dtype=np.int32)
        self.family_idx = np.asarray(family_idx, dtype=np.int32)
        self.series_exact_idx = np.asarray(series_exact_idx, dtype=np.int32)
        self.parallel_exact_idx = np.asarray(parallel_exact_idx, dtype=np.int32)
        self.batch_size = int(max(batch_size, 1))
        self.n_family_classes = int(n_family_classes)
        self.series_family_idx = int(series_family_idx)
        self.parallel_family_idx = int(parallel_family_idx)
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
        audio_batch = normalize_audio_batch(self.audio[audio_ids]).reshape(len(audio_ids), -1, 1)
        family_batch = tf.keras.utils.to_categorical(self.family_idx[batch_ids], num_classes=self.n_family_classes).astype(np.float32)
        x = {
            "audio_input": audio_batch,
            "family_condition_input": family_batch,
        }
        y = (
            self.family_idx[batch_ids],
            self.series_exact_idx[batch_ids],
            self.parallel_exact_idx[batch_ids],
        )
        sample_weight = (
            np.ones(len(batch_ids), dtype=np.float32),
            (self.family_idx[batch_ids] == self.series_family_idx).astype(np.float32),
            (self.family_idx[batch_ids] == self.parallel_family_idx).astype(np.float32),
        )
        return x, y, sample_weight

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class AudioOnlySequence(tf.keras.utils.Sequence):
    def __init__(self, audio: np.ndarray, sample_indices: np.ndarray, batch_size: int, shuffle: bool = False):
        super().__init__()
        self.audio = audio
        self.sample_indices = np.asarray(sample_indices, dtype=np.int32)
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
        audio_batch = normalize_audio_batch(self.audio[audio_ids]).reshape(len(audio_ids), -1, 1)
        return audio_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class AudioConditionSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        audio: np.ndarray,
        sample_indices: np.ndarray,
        condition_idx: np.ndarray,
        batch_size: int,
        n_family_classes: int,
        shuffle: bool = False,
    ):
        super().__init__()
        self.audio = audio
        self.sample_indices = np.asarray(sample_indices, dtype=np.int32)
        self.condition_idx = np.asarray(condition_idx, dtype=np.int32)
        self.batch_size = int(max(batch_size, 1))
        self.n_family_classes = int(n_family_classes)
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
        audio_batch = normalize_audio_batch(self.audio[audio_ids]).reshape(len(audio_ids), -1, 1)
        family_batch = tf.keras.utils.to_categorical(self.condition_idx[batch_ids], num_classes=self.n_family_classes).astype(np.float32)
        return {
            "audio_input": audio_batch,
            "family_condition_input": family_batch,
        }

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def decode_exact_predictions(
    family_indices: np.ndarray,
    series_probs: np.ndarray,
    parallel_probs: np.ndarray,
    family_classes: list[str],
) -> tuple[list[str], np.ndarray]:
    predictions: list[str] = []
    probabilities = np.zeros(len(family_indices), dtype=np.float32)
    for i, family_idx in enumerate(family_indices):
        family_name = family_classes[int(family_idx)]
        if family_name == SERIES_FAMILY:
            local_probs = series_probs[i]
            local_idx = int(np.argmax(local_probs))
            predictions.append(FAMILY_TO_ALGOS[SERIES_FAMILY][local_idx])
            probabilities[i] = float(np.max(local_probs))
        else:
            local_probs = parallel_probs[i]
            local_idx = int(np.argmax(local_probs))
            predictions.append(FAMILY_TO_ALGOS[PARALLEL_FAMILY][local_idx])
            probabilities[i] = float(np.max(local_probs))
    return predictions, probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the big20 hierarchical algorithm classifier.")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(os.getenv("OUTPUT_DIR", MODEL_NAME))
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_raw = load_target_frame(args.max_samples)
    mappings = build_categorical_mappings(target_raw)
    family_classes = mappings["family_classes"]
    family_map = mappings["family_map"]
    family_idx = mappings["family_idx"].to_numpy(dtype=np.int32)
    series_algorithms = mappings["series_algorithms"]
    parallel_algorithms = mappings["parallel_algorithms"]
    series_map = mappings["series_map"]
    parallel_map = mappings["parallel_map"]
    series_family_idx = mappings["series_family_idx"]
    parallel_family_idx = mappings["parallel_family_idx"]

    target_raw = target_raw.copy()
    target_raw["family_idx"] = family_idx
    target_raw["series_exact_idx"] = target_raw["algorithm"].map(series_map).fillna(0).astype(np.int32)
    target_raw["parallel_exact_idx"] = target_raw["algorithm"].map(parallel_map).fillna(0).astype(np.int32)

    sample_ids = target_raw["id"].astype(int).tolist()
    audio_store = load_audio_store(BASE_PATH)
    audio_len = int(getattr(audio_store, "sample_len", 0) or np.asarray(audio_store[sample_ids[:1]]).shape[1])

    dataset_indices = target_raw.index.to_numpy(dtype=np.int32)
    train_strata = group_stratify_key(target_raw, max_classes=max(2, int((1.0 - TRAIN_FRAC) * len(dataset_indices))))
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

    fit_idx, val_idx = stratified_split_indices(
        np.arange(len(y_train_full), dtype=np.int32),
        group_stratify_key(y_train_full, max_classes=max(2, int(VAL_FRAC * len(y_train_full)))),
        test_size=VAL_FRAC,
        random_state=RANDOM_STATE,
    )

    fit_audio_indices = train_audio_indices[fit_idx]
    val_audio_indices = train_audio_indices[val_idx]

    y_fit = y_train_full.iloc[fit_idx].reset_index(drop=True)
    y_val = y_train_full.iloc[val_idx].reset_index(drop=True)

    train_seq = HierarchicalSequence(
        audio_store,
        fit_audio_indices,
        y_fit["family_idx"].to_numpy(dtype=np.int32),
        y_fit["series_exact_idx"].to_numpy(dtype=np.int32),
        y_fit["parallel_exact_idx"].to_numpy(dtype=np.int32),
        BATCH_SIZE,
        len(family_classes),
        series_family_idx,
        parallel_family_idx,
        shuffle=True,
    )
    val_seq = HierarchicalSequence(
        audio_store,
        val_audio_indices,
        y_val["family_idx"].to_numpy(dtype=np.int32),
        y_val["series_exact_idx"].to_numpy(dtype=np.int32),
        y_val["parallel_exact_idx"].to_numpy(dtype=np.int32),
        PRED_BATCH_SIZE,
        len(family_classes),
        series_family_idx,
        parallel_family_idx,
        shuffle=False,
    )
    test_seq = HierarchicalSequence(
        audio_store,
        test_audio_indices,
        y_test_full["family_idx"].to_numpy(dtype=np.int32),
        y_test_full["series_exact_idx"].to_numpy(dtype=np.int32),
        y_test_full["parallel_exact_idx"].to_numpy(dtype=np.int32),
        PRED_BATCH_SIZE,
        len(family_classes),
        series_family_idx,
        parallel_family_idx,
        shuffle=False,
    )

    family_pred_seq = AudioOnlySequence(audio_store, test_audio_indices, PRED_BATCH_SIZE, shuffle=False)

    model = build_model(audio_len, len(family_classes))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss=[
            tf.keras.losses.SparseCategoricalCrossentropy(),
            tf.keras.losses.SparseCategoricalCrossentropy(),
            tf.keras.losses.SparseCategoricalCrossentropy(),
        ],
        loss_weights=[1.0, 1.0, 1.0],
        metrics=[
            ["sparse_categorical_accuracy"],
            ["sparse_categorical_accuracy"],
            ["sparse_categorical_accuracy"],
        ],
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
        f"dataset={BASE_PATH}, available_samples={len(target_raw)}, epochs={EPOCHS}, batch_size={BATCH_SIZE}, "
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

    family_model = Model(inputs=model.inputs[0], outputs=model.get_layer("algorithm_family_head").output, name="family_only_model")

    family_true = y_test_full["family_idx"].to_numpy(dtype=np.int32)
    family_probs = family_model.predict(family_pred_seq, verbose=0)
    family_pred_idx = np.argmax(family_probs, axis=1).astype(np.int32)

    true_family_seq = AudioConditionSequence(
        audio_store,
        test_audio_indices,
        family_true,
        PRED_BATCH_SIZE,
        len(family_classes),
        shuffle=False,
    )
    pred_family_seq = AudioConditionSequence(
        audio_store,
        test_audio_indices,
        family_pred_idx,
        PRED_BATCH_SIZE,
        len(family_classes),
        shuffle=False,
    )

    true_family_outputs = model.predict(true_family_seq, verbose=0)
    pred_family_outputs = model.predict(pred_family_seq, verbose=0)

    true_series_probs = np.asarray(true_family_outputs[1], dtype=np.float32)
    true_parallel_probs = np.asarray(true_family_outputs[2], dtype=np.float32)
    pred_series_probs = np.asarray(pred_family_outputs[1], dtype=np.float32)
    pred_parallel_probs = np.asarray(pred_family_outputs[2], dtype=np.float32)

    oracle_pred, oracle_prob = decode_exact_predictions(
        family_true,
        true_series_probs,
        true_parallel_probs,
        family_classes,
    )
    cascade_pred, cascade_prob = decode_exact_predictions(
        family_pred_idx,
        pred_series_probs,
        pred_parallel_probs,
        family_classes,
    )

    exact_true = y_test_full["algorithm"].astype(str).tolist()
    family_true_names = [family_classes[int(i)] for i in family_true]
    family_pred_names = [family_classes[int(i)] for i in family_pred_idx]

    oracle_acc = float(np.mean(np.asarray(oracle_pred, dtype=object) == np.asarray(exact_true, dtype=object)))
    cascade_acc = float(np.mean(np.asarray(cascade_pred, dtype=object) == np.asarray(exact_true, dtype=object)))
    family_acc = float(np.mean(family_pred_idx == family_true))

    series_mask = family_true == family_map[SERIES_FAMILY]
    parallel_mask = family_true == family_map[PARALLEL_FAMILY]
    series_oracle_acc = (
        float(np.mean(np.argmax(true_series_probs[series_mask], axis=1) == y_test_full.loc[series_mask, "series_exact_idx"].to_numpy(dtype=np.int32)))
        if np.any(series_mask)
        else None
    )
    parallel_oracle_acc = (
        float(np.mean(np.argmax(true_parallel_probs[parallel_mask], axis=1) == y_test_full.loc[parallel_mask, "parallel_exact_idx"].to_numpy(dtype=np.int32)))
        if np.any(parallel_mask)
        else None
    )

    final_predictions = {
        "sample_id": test_audio_indices.astype(int),
        "algorithm_true": exact_true,
        "algorithm_pred_oracle": oracle_pred,
        "algorithm_pred_cascade": cascade_pred,
        "algorithm_oracle_prob": oracle_prob.tolist(),
        "algorithm_cascade_prob": cascade_prob.tolist(),
        "algorithm_family_true": family_true_names,
        "algorithm_family_pred": family_pred_names,
        "algorithm_family_prob": np.max(family_probs, axis=1).tolist(),
    }
    pd.DataFrame(final_predictions).to_csv(output_dir / "predictions.csv", index=False)

    final_metrics = {
        "algorithm_family_accuracy": family_acc,
        "algorithm_exact_accuracy_oracle": oracle_acc,
        "algorithm_exact_accuracy_cascade": cascade_acc,
        "series_exact_accuracy_oracle": series_oracle_acc,
        "parallel_exact_accuracy_oracle": parallel_oracle_acc,
    }

    results = {
        "model_name": MODEL_NAME,
        "dataset": str(BASE_PATH),
        "target": "algorithm",
        "hierarchical": True,
        "runtime_config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "pred_batch_size": PRED_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "dropout": DROPOUT,
            "base_filters": BASE_FILTERS,
            "cnn_blocks": CNN_BLOCKS,
            "dense_units": DENSE_UNITS,
            "exact_branch_dense_units": EXACT_BRANCH_DENSE_UNITS,
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
        "family_classes": family_classes,
        "series_algorithms": series_algorithms,
        "parallel_algorithms": parallel_algorithms,
        "final_metrics": final_metrics,
    }

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe(results), f, indent=2, ensure_ascii=False)

    with open(output_dir / "family_map.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe({"classes": family_classes, "mapping": family_map}), f, indent=2, ensure_ascii=False)

    with open(output_dir / "algorithm_maps.json", "w", encoding="utf-8") as f:
        json.dump(
            make_json_safe(
                {
                    "series": {"classes": series_algorithms, "mapping": series_map},
                    "parallel": {"classes": parallel_algorithms, "mapping": parallel_map},
                }
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Results written to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
