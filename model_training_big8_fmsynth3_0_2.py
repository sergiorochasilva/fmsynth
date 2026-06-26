"""Compact FM multitask regressor for `dataset_big8`, version 0_2.

Architecture:
- Raw waveform input
- Log-mel spectrogram front-end with a compact 2D CNN backbone
- Shared dense trunk with categorical heads for `algorithm` and `frequency bin`
- A regression head over `log2(frequencia_base)` for finer frequency recovery

Data flow:
- Input: `dataset_big8/parameters.csv` plus `audio_big8_manifest.json` shards or `sample_*.wav`
- Output: trained weights, preprocessing artifacts, predictions, plots, and `results.json`
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
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    LayerNormalization,
    MaxPooling2D,
)
from keras.models import Model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import mixed_precision

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_PATH = Path(os.getenv("DATASET_PATH", "dataset_big8"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "model_training_big8_fmsynth3_0_2"))
MODEL_NAME = "model_training_big8_fmsynth3_0_2"

MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "0"))
RANDOM_STATE = int(os.getenv("SEED", "42"))
TRAIN_FRAC = float(os.getenv("TRAIN_FRAC", "0.75"))
VAL_FRAC = float(os.getenv("VAL_FRAC", "0.20"))

BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "16"))
PRED_BATCH_SIZE = int(os.getenv("PRED_BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "28"))
PATIENCE = int(os.getenv("PATIENCE", "5"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2.5e-4"))
DROPOUT = float(os.getenv("DROPOUT", "0.15"))
BASE_FILTERS = int(os.getenv("BASE_FILTERS", "32"))
CNN_BLOCKS = int(os.getenv("CNN_BLOCKS", "4"))
DENSE_UNITS = int(os.getenv("DENSE_UNITS", "128"))
FREQ_BIN_COUNT = int(os.getenv("FREQ_BIN_COUNT", "12"))
USE_MIXED_PRECISION = os.getenv("MIXED_PRECISION", "0") == "1"
ENABLE_XLA = os.getenv("ENABLE_XLA", "0") == "1"
RESUME_TRAINING = os.getenv("RESUME_TRAINING", "1") == "1"

N_FFT = int(os.getenv("N_FFT", "1024"))
HOP_LENGTH = int(os.getenv("HOP_LENGTH", "256"))
N_MELS = int(os.getenv("N_MELS", "64"))
MEL_FMIN = float(os.getenv("MEL_FMIN", "20.0"))
MEL_FMAX = float(os.getenv("MEL_FMAX", "7600.0"))
EPS = 1e-6
MEL_MATRIX = tf.constant(
    tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=N_FFT // 2 + 1,
        sample_rate=16000,
        lower_edge_hertz=MEL_FMIN,
        upper_edge_hertz=MEL_FMAX,
    ),
    dtype=tf.float32,
)

AUDIO_MANIFEST_PATH = BASE_PATH / "audio_big8_manifest.json"
AUDIO_LEGACY_CACHE_PATH = BASE_PATH / "audio_big8_int16.npy"
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
    if "algorithm" not in frame.columns:
        return None
    return frame["algorithm"].astype(str)


def stratified_split_indices(indices: np.ndarray, strata: pd.Series | None, test_size: float, random_state: int):
    if strata is None:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(indices)
        cut = int(round((1.0 - test_size) * len(perm)))
        return perm[:cut], perm[cut:]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_pos, test_pos = next(splitter.split(np.zeros(len(indices)), strata))
    return indices[train_pos], indices[test_pos]


def build_combined_strata(algorithm_labels: np.ndarray, freq_bins: np.ndarray) -> pd.Series:
    labels = [f"{algo}__{bin_id}" for algo, bin_id in zip(algorithm_labels, freq_bins)]
    return pd.Series(labels)


def build_freq_bins(train_freq_log2: np.ndarray, n_bins: int) -> tuple[np.ndarray, int]:
    series = pd.Series(train_freq_log2)
    _, edges = pd.qcut(series, q=n_bins, labels=False, retbins=True, duplicates="drop")
    edges = np.asarray(edges, dtype=np.float32)
    edges = np.unique(edges)
    if edges.size < 3:
        edges = np.linspace(float(train_freq_log2.min()), float(train_freq_log2.max()), num=max(n_bins, 2) + 1)
        edges = np.asarray(edges, dtype=np.float32)
    freq_bin_count = int(edges.size - 1)
    return edges, freq_bin_count


def assign_freq_bins(freq_log2: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.digitize(freq_log2, edges[1:-1], right=False)
    bins = np.clip(bins, 0, len(edges) - 2)
    return bins.astype(np.int32)


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


def logmel_frontend(x):
    x = tf.cast(tf.squeeze(x, axis=-1), tf.float32)
    stft = tf.signal.stft(
        x,
        frame_length=N_FFT,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        pad_end=False,
    )
    magnitude = tf.abs(stft)
    mel = tf.tensordot(magnitude, MEL_MATRIX, axes=1)
    mel = tf.maximum(mel, EPS)
    log_mel = tf.math.log(mel)
    return tf.expand_dims(log_mel, axis=-1)


def build_model(input_len: int, n_algorithm_classes: int, n_freq_bins: int) -> Model:
    inputs = Input(shape=(input_len, 1), name="audio_input")
    x = tf.keras.layers.Lambda(logmel_frontend, name="logmel_frontend")(inputs)
    x = BatchNormalization(name="spec_bn")(x)

    filters = BASE_FILTERS
    for block_idx in range(CNN_BLOCKS):
        x = Conv2D(filters, kernel_size=(3, 3), padding="same", activation="swish", name=f"conv_{block_idx + 1}_a")(x)
        x = BatchNormalization(name=f"bn_{block_idx + 1}_a")(x)
        x = Conv2D(filters, kernel_size=(3, 3), padding="same", activation="swish", name=f"conv_{block_idx + 1}_b")(x)
        x = BatchNormalization(name=f"bn_{block_idx + 1}_b")(x)
        x = MaxPooling2D(pool_size=(2, 2), name=f"pool_{block_idx + 1}")(x)
        x = Dropout(DROPOUT, name=f"drop_{block_idx + 1}")(x)
        filters = min(filters * 2, BASE_FILTERS * 4)

    x = GlobalAveragePooling2D(name="gap")(x)
    x = LayerNormalization(name="shared_ln")(x)
    x = Dense(DENSE_UNITS, activation="swish", name="shared_dense")(x)
    x = Dropout(DROPOUT, name="shared_drop")(x)
    x = Dense(max(DENSE_UNITS // 2, 64), activation="swish", name="shared_dense_2")(x)

    algorithm_head = Dense(n_algorithm_classes, activation="softmax", name="algorithm_head")(x)
    freq_bin_head = Dense(n_freq_bins, activation="softmax", name="freq_bin_head")(x)
    freq_log2_head = Dense(1, activation=None, dtype="float32", name="freq_log2_head")(x)

    return Model(
        inputs=inputs,
        outputs=[algorithm_head, freq_bin_head, freq_log2_head],
        name="compact_logmel_multitask_big8_0_2",
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

    if "algorithm" not in target_raw.columns or "frequencia_base" not in target_raw.columns:
        raise ValueError("dataset_big8 must contain `algorithm` and `frequencia_base` columns.")

    target_raw = target_raw.copy()
    target_raw["algorithm"] = target_raw["algorithm"].astype(str)
    target_raw["freq_log2"] = np.log2(target_raw["frequencia_base"].astype(np.float32))

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

    algorithm_categories = pd.Categorical(target_raw["algorithm"])
    algorithm_classes = [str(x) for x in algorithm_categories.categories]
    algorithm_map = {name: idx for idx, name in enumerate(algorithm_classes)}

    y_train_full["algorithm_idx"] = y_train_full["algorithm"].map(algorithm_map).astype(np.int32)
    y_test_full["algorithm_idx"] = y_test_full["algorithm"].map(algorithm_map).astype(np.int32)

    freq_bin_edges, freq_bin_count = build_freq_bins(y_train_full["freq_log2"].to_numpy(dtype=np.float32), FREQ_BIN_COUNT)
    y_train_full["freq_bin"] = assign_freq_bins(y_train_full["freq_log2"].to_numpy(dtype=np.float32), freq_bin_edges)
    y_test_full["freq_bin"] = assign_freq_bins(y_test_full["freq_log2"].to_numpy(dtype=np.float32), freq_bin_edges)

    y_train_model = y_train_full.copy()
    combined_strata = build_combined_strata(
        y_train_model["algorithm"].to_numpy(dtype=str),
        y_train_model["freq_bin"].to_numpy(dtype=np.int32),
    )
    fit_idx, val_idx = stratified_split_indices(
        np.arange(len(y_train_model), dtype=np.int32),
        combined_strata,
        test_size=VAL_FRAC,
        random_state=RANDOM_STATE,
    )

    fit_audio_indices = train_audio_indices[fit_idx]
    val_audio_indices = train_audio_indices[val_idx]

    freq_scaler = StandardScaler()
    y_freq_train = freq_scaler.fit_transform(y_train_model["freq_log2"].to_numpy(dtype=np.float32).reshape(-1, 1)).astype(np.float32)
    y_freq_fit = y_freq_train[fit_idx]
    y_freq_val = y_freq_train[val_idx]

    y_fit = {
        "algorithm_head": y_train_model["algorithm_idx"].to_numpy(dtype=np.int32)[fit_idx],
        "freq_bin_head": y_train_model["freq_bin"].to_numpy(dtype=np.int32)[fit_idx],
        "freq_log2_head": y_freq_fit,
    }
    y_val = {
        "algorithm_head": y_train_model["algorithm_idx"].to_numpy(dtype=np.int32)[val_idx],
        "freq_bin_head": y_train_model["freq_bin"].to_numpy(dtype=np.int32)[val_idx],
        "freq_log2_head": y_freq_val,
    }

    y_test_freq_log2 = freq_scaler.transform(y_test_full["freq_log2"].to_numpy(dtype=np.float32).reshape(-1, 1)).astype(np.float32)

    np.save(OUTPUT_DIR / "train_audio_indices.npy", train_audio_indices)
    np.save(OUTPUT_DIR / "val_audio_indices.npy", val_audio_indices)
    np.save(OUTPUT_DIR / "test_audio_indices.npy", test_audio_indices)
    y_train_full.to_csv(OUTPUT_DIR / "y_train_big8_v2.csv", index=False)
    y_test_full.to_csv(OUTPUT_DIR / "y_test_big8_v2.csv", index=False)
    np.save(OUTPUT_DIR / "freq_bin_edges.npy", freq_bin_edges.astype(np.float32))

    train_seq = MultiTaskSequence(audio_store, fit_audio_indices, y_fit, BATCH_SIZE, shuffle=True)
    val_seq = MultiTaskSequence(audio_store, val_audio_indices, y_val, PRED_BATCH_SIZE, shuffle=False)
    test_seq = MultiTaskSequence(
        audio_store,
        test_audio_indices,
        {
            "algorithm_head": y_test_full["algorithm_idx"].to_numpy(dtype=np.int32),
            "freq_bin_head": y_test_full["freq_bin"].to_numpy(dtype=np.int32),
            "freq_log2_head": y_test_freq_log2.astype(np.float32),
        },
        PRED_BATCH_SIZE,
        shuffle=False,
    )

    model = build_model(audio_len, len(algorithm_classes), freq_bin_count)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            "algorithm_head": tf.keras.losses.SparseCategoricalCrossentropy(),
            "freq_bin_head": tf.keras.losses.SparseCategoricalCrossentropy(),
            "freq_log2_head": tf.keras.losses.Huber(delta=1.0),
        },
        metrics={
            "algorithm_head": ["sparse_categorical_accuracy"],
            "freq_bin_head": ["sparse_categorical_accuracy"],
            "freq_log2_head": ["mae", "mse"],
        },
        loss_weights={
            "algorithm_head": 1.0,
            "freq_bin_head": 0.7,
            "freq_log2_head": 1.2,
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
        f"base_filters={BASE_FILTERS}, blocks={CNN_BLOCKS}, freq_bin_count={freq_bin_count}, "
        f"resume_epoch={resume_epoch}"
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
    freq_bin_pred = preds[1]
    freq_log2_pred_scaled = preds[2]

    algo_pred_idx = np.argmax(algo_pred, axis=1)
    freq_bin_pred_idx = np.argmax(freq_bin_pred, axis=1)

    algo_true = y_test_full["algorithm_idx"].to_numpy(dtype=np.int32)
    freq_bin_true = y_test_full["freq_bin"].to_numpy(dtype=np.int32)
    freq_log2_true = y_test_full["freq_log2"].to_numpy(dtype=np.float32)
    freq_log2_pred = freq_scaler.inverse_transform(np.asarray(freq_log2_pred_scaled, dtype=np.float32)).reshape(-1)

    freq_hz_true = np.power(2.0, freq_log2_true)
    freq_hz_pred = np.power(2.0, freq_log2_pred)

    algo_acc = float(np.mean(algo_pred_idx == algo_true))
    freq_bin_acc = float(np.mean(freq_bin_pred_idx == freq_bin_true))
    algo_ce = float(tf.keras.losses.sparse_categorical_crossentropy(algo_true, algo_pred).numpy().mean())
    freq_bin_ce = float(tf.keras.losses.sparse_categorical_crossentropy(freq_bin_true, freq_bin_pred).numpy().mean())

    freq_mae_hz = float(mean_absolute_error(freq_hz_true, freq_hz_pred))
    freq_mse_hz = float(mean_squared_error(freq_hz_true, freq_hz_pred))
    freq_rmse_hz = float(np.sqrt(freq_mse_hz))
    freq_mae_cents = float(np.mean(np.abs(1200.0 * (freq_log2_pred - freq_log2_true))))
    freq_log2_mae = float(mean_absolute_error(freq_log2_true, freq_log2_pred))

    predictions = pd.DataFrame(
        {
            "sample_id": test_audio_indices.astype(int),
            "algorithm_true": [algorithm_classes[i] for i in algo_true],
            "algorithm_pred": [algorithm_classes[i] for i in algo_pred_idx],
            "algorithm_prob": np.max(algo_pred, axis=1),
            "freq_bin_true": freq_bin_true,
            "freq_bin_pred": freq_bin_pred_idx,
            "freq_bin_prob": np.max(freq_bin_pred, axis=1),
            "frequencia_base_true": freq_hz_true,
            "frequencia_base_pred": freq_hz_pred,
            "freq_log2_true": freq_log2_true,
            "freq_log2_pred": freq_log2_pred,
        }
    )
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    plot_metric(history_df, "loss", "val_loss", "Loss", "train_loss.png")
    plot_metric(history_df, "algorithm_head_sparse_categorical_accuracy", "val_algorithm_head_sparse_categorical_accuracy", "Algorithm Accuracy", "train_algorithm_accuracy.png")
    plot_metric(history_df, "freq_bin_head_sparse_categorical_accuracy", "val_freq_bin_head_sparse_categorical_accuracy", "Frequency Bin Accuracy", "train_freq_bin_accuracy.png")

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
            "freq_bin_count": freq_bin_count,
            "mixed_precision": USE_MIXED_PRECISION,
            "resume_training": RESUME_TRAINING,
        },
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "algorithm_classes": algorithm_classes,
        "freq_bin_edges": freq_bin_edges.tolist(),
        "final_metrics": {
            "algorithm_accuracy": algo_acc,
            "algorithm_crossentropy": algo_ce,
            "freq_bin_accuracy": freq_bin_acc,
            "freq_bin_crossentropy": freq_bin_ce,
            "freq_mae_hz": freq_mae_hz,
            "freq_rmse_hz": freq_rmse_hz,
            "freq_mae_cents": freq_mae_cents,
            "freq_log2_mae": freq_log2_mae,
        },
        "history": history_df.to_dict(orient="list"),
        "test_prediction_preview": predictions.head(10).to_dict(orient="records"),
        "freq_scaler_mean": freq_scaler.mean_.tolist(),
        "freq_scaler_scale": freq_scaler.scale_.tolist(),
        "best_weights_path": str(CHECKPOINT_BEST_WEIGHTS),
        "latest_weights_path": str(CHECKPOINT_LATEST_WEIGHTS),
    }

    atomic_json_dump(OUTPUT_DIR / "results.json", results)
    joblib.dump(freq_scaler, OUTPUT_DIR / "freq_log2_scaler.joblib")
    with open(OUTPUT_DIR / "algorithm_map.json", "w", encoding="utf-8") as f:
        json.dump(algorithm_map, f, indent=2, ensure_ascii=False)

    print(f"Results written to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
