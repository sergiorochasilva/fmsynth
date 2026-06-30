"""Pitch-focused `big17` training for `dataset_big13`, version `0_2`.

Architecture:
- Raw waveform input
- Multi-resolution log-mel front-end
- Residual 1D CNN backbone
- Explicit algorithm conditioning
- Pitch submodel with three heads:
  - `ratio_log2_head` regression
  - `ratio_class_head` discrete classification over corpus ratio values
  - `freq_log2_head` regression

Data flow:
- Input: `dataset_big13/parameters.csv` plus rendered `sample_*.wav` audio
- Output: trained pitch weights, scalers, ratio classes, predictions, and `results.json`
"""

from __future__ import annotations

import argparse
import json
import os
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
from sklearn.preprocessing import StandardScaler

os.environ.pop("TARGET_GROUP", None)

from model_training_big13_fmsynth3_0_1 import (
    atomic_json_dump,
    build_ratio_classes,
    inverse_transform_series,
    load_json_file,
    make_json_safe,
    normalize_audio_batch,
    sparse_categorical_focal_loss,
    stratified_split_indices,
    to_json_scalar,
    transform_series,
)
from model_training_big17_fmsynth3_0_1 import load_audio_store, group_stratify_key

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_PATH = Path(os.getenv("DATASET_PATH", "dataset_big13"))
MODEL_PREFIX = "model_training_big17_fmsynth3_0_2"
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "0"))
RANDOM_STATE = int(os.getenv("SEED", "42"))
TRAIN_FRAC = float(os.getenv("TRAIN_FRAC", "0.80"))
VAL_FRAC = float(os.getenv("VAL_FRAC", "0.15"))
BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "6"))
PRED_BATCH_SIZE = int(os.getenv("PRED_BATCH_SIZE", "8"))
EPOCHS = int(os.getenv("EPOCHS", "36"))
PATIENCE = int(os.getenv("PATIENCE", "6"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1.5e-4"))
DROPOUT = float(os.getenv("DROPOUT", "0.08"))
BASE_FILTERS = int(os.getenv("BASE_FILTERS", "48"))
CNN_BLOCKS = int(os.getenv("CNN_BLOCKS", "4"))
DENSE_UNITS = int(os.getenv("DENSE_UNITS", "256"))
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

PITCH_SPECS = [
    {"head": "ratio_log2_head", "column": "ratio_carrier", "transform": "log2", "loss_weight": 0.60},
    {"head": "ratio_class_head", "column": "ratio_carrier", "transform": "categorical", "loss_weight": 1.00},
    {"head": "freq_log2_head", "column": "frequencia_base", "transform": "log2", "loss_weight": 0.90},
]

MODEL_NAME = f"{MODEL_PREFIX}_pitch"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", MODEL_NAME))
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


def load_target_frame(max_samples: int | None = None) -> pd.DataFrame:
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"Missing parameters CSV: {PARAMS_PATH}")
    frame = pd.read_csv(PARAMS_PATH)
    if "id" not in frame.columns:
        frame = frame.reset_index().rename(columns={"index": "id"})
    manifest_count = discover_contiguous_prefix(BASE_PATH)
    available_count = min(len(frame), manifest_count)
    effective_max = MAX_SAMPLES if max_samples is None else int(max_samples)
    if effective_max > 0:
        available_count = min(available_count, effective_max)
    return frame.iloc[:available_count].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pitch-focused big17 v0_2 model.")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    return parser.parse_args()


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


def build_model(input_len: int, n_algorithm_classes: int, ratio_class_count: int) -> Model:
    audio_input = Input(shape=(input_len, 1), name="audio_input")
    algorithm_condition_input = Input(shape=(n_algorithm_classes,), name="algorithm_condition_input")

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

    gap = GlobalAveragePooling1D(name="gap")(x)
    gmp = GlobalMaxPooling1D(name="gmp")(x)
    shared = Concatenate(name="global_concat")([gap, gmp])
    shared = LayerNormalization(name="shared_ln")(shared)
    shared = Dense(DENSE_UNITS, activation="swish", name="shared_dense")(shared)
    shared = Dropout(DROPOUT, name="shared_drop")(shared)
    shared = Dense(max(DENSE_UNITS // 2, 128), activation="swish", name="shared_dense_2")(shared)
    shared = Dropout(DROPOUT, name="shared_drop_2")(shared)

    algo_condition = Dense(max(DENSE_UNITS // 4, 64), activation="swish", name="algorithm_condition_dense")(algorithm_condition_input)
    algo_condition = LayerNormalization(name="algorithm_condition_ln")(algo_condition)

    pitch_base = Concatenate(name="pitch_condition_concat")([shared, algo_condition])
    pitch_base = Dense(max(DENSE_UNITS // 2, 160), activation="swish", name="pitch_base_dense")(pitch_base)
    pitch_base = LayerNormalization(name="pitch_base_ln")(pitch_base)
    pitch_base = Dropout(DROPOUT, name="pitch_base_drop")(pitch_base)

    ratio_branch = Dense(max(DENSE_UNITS // 2, 160), activation="swish", name="ratio_branch_dense")(pitch_base)
    ratio_branch = LayerNormalization(name="ratio_branch_ln")(ratio_branch)
    ratio_branch = Dropout(DROPOUT, name="ratio_branch_drop")(ratio_branch)
    ratio_branch = Dense(max(DENSE_UNITS // 3, 128), activation="swish", name="ratio_branch_dense_2")(ratio_branch)
    ratio_branch = LayerNormalization(name="ratio_branch_ln_2")(ratio_branch)
    ratio_branch = Dropout(DROPOUT, name="ratio_branch_drop_2")(ratio_branch)

    freq_branch = Concatenate(name="freq_condition_concat")([pitch_base, ratio_branch])
    freq_branch = Dense(max(DENSE_UNITS // 2, 160), activation="swish", name="freq_branch_dense")(freq_branch)
    freq_branch = LayerNormalization(name="freq_branch_ln")(freq_branch)
    freq_branch = Dropout(DROPOUT, name="freq_branch_drop")(freq_branch)
    freq_branch = Dense(max(DENSE_UNITS // 3, 128), activation="swish", name="freq_branch_dense_2")(freq_branch)
    freq_branch = LayerNormalization(name="freq_branch_ln_2")(freq_branch)
    freq_branch = Dropout(DROPOUT, name="freq_branch_drop_2")(freq_branch)

    outputs = [
        Dense(1, activation=None, dtype="float32", name="ratio_log2_head")(ratio_branch),
        Dense(ratio_class_count, activation="softmax", name="ratio_class_head")(ratio_branch),
        Dense(1, activation=None, dtype="float32", name="freq_log2_head")(freq_branch),
    ]
    return Model(inputs=[audio_input, algorithm_condition_input], outputs=outputs, name="pitch_fm_big17_0_2")


def plot_metric(history_df: pd.DataFrame, train_key: str, val_key: str, title: str, output_name: str, output_dir: Path) -> None:
    if train_key not in history_df.columns or val_key not in history_df.columns:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(history_df["epoch"], history_df[train_key], label=train_key)
    plt.plot(history_df["epoch"], history_df[val_key], label=val_key)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / output_name)
    plt.close()


def require_gpu_available() -> None:
    physical_gpus = tf.config.list_physical_devices("GPU")
    if physical_gpus:
        return

    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "<unset>")
    built_with_cuda = tf.test.is_built_with_cuda()
    raise RuntimeError(
        "GPU unavailable for TensorFlow. "
        f"built_with_cuda={built_with_cuda}, "
        f"physical_gpus={physical_gpus}, "
        f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}. "
        "Abortando para evitar fallback em CPU."
    )


def main() -> None:
    args = parse_args()
    require_gpu_available()
    output_dir = Path(os.getenv("OUTPUT_DIR", MODEL_NAME))
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_target_frame(args.max_samples)
    if "algorithm" not in frame.columns:
        raise ValueError("dataset_big13 must contain an `algorithm` column.")

    frame = frame.copy()
    frame["algorithm"] = frame["algorithm"].astype(str)
    algorithm_classes = sorted(frame["algorithm"].unique().tolist())
    algorithm_map = {name: idx for idx, name in enumerate(algorithm_classes)}
    frame["algorithm_idx"] = frame["algorithm"].map(algorithm_map).astype(np.int32)

    ratio_classes = build_ratio_classes(frame["ratio_carrier"].to_numpy(dtype=np.float32))
    ratio_class_map = {float(value): idx for idx, value in enumerate(ratio_classes.tolist())}

    frame["ratio_log2_head"] = transform_series(frame["ratio_carrier"], "log2")
    frame["freq_log2_head"] = transform_series(frame["frequencia_base"], "log2")

    required_columns = {"ratio_carrier", "frequencia_base"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"dataset_big13 is missing required columns: {missing_columns}")

    sample_ids = frame["id"].astype(int).tolist()
    audio_store = load_audio_store(BASE_PATH)
    audio_len = int(getattr(audio_store, "sample_len", 0) or np.asarray(audio_store[sample_ids[:1]]).shape[1])

    dataset_indices = frame.index.to_numpy(dtype=np.int32)
    train_strata = group_stratify_key(frame, "pitch")
    train_idx, test_idx = stratified_split_indices(dataset_indices, train_strata, test_size=1.0 - TRAIN_FRAC, random_state=RANDOM_STATE)
    train_audio_indices = np.asarray(train_idx, dtype=np.int32)
    test_audio_indices = np.asarray(test_idx, dtype=np.int32)

    train_frame = frame.loc[train_idx].reset_index(drop=True)
    test_frame = frame.loc[test_idx].reset_index(drop=True)
    train_ratio_class_indices = np.array([ratio_class_map[float(round(v, 4))] for v in train_frame["ratio_carrier"].to_numpy(dtype=np.float32)], dtype=np.int32)
    test_ratio_class_indices = np.array([ratio_class_map[float(round(v, 4))] for v in test_frame["ratio_carrier"].to_numpy(dtype=np.float32)], dtype=np.int32)
    fit_idx, val_idx = stratified_split_indices(
        np.arange(len(train_frame), dtype=np.int32),
        group_stratify_key(train_frame, "pitch"),
        test_size=VAL_FRAC,
        random_state=RANDOM_STATE,
    )
    fit_audio_indices = train_audio_indices[fit_idx]
    val_audio_indices = train_audio_indices[val_idx]

    ratio_scaler = StandardScaler()
    freq_scaler = StandardScaler()
    ratio_fit_scaled = ratio_scaler.fit_transform(train_frame["ratio_log2_head"].to_numpy(dtype=np.float32).reshape(-1, 1)).astype(np.float32)
    ratio_val_scaled = ratio_scaler.transform(train_frame["ratio_log2_head"].to_numpy(dtype=np.float32)[val_idx].reshape(-1, 1)).astype(np.float32)
    ratio_test_scaled = ratio_scaler.transform(test_frame["ratio_log2_head"].to_numpy(dtype=np.float32).reshape(-1, 1)).astype(np.float32)

    freq_fit_scaled = freq_scaler.fit_transform(train_frame["freq_log2_head"].to_numpy(dtype=np.float32).reshape(-1, 1)).astype(np.float32)
    freq_val_scaled = freq_scaler.transform(train_frame["freq_log2_head"].to_numpy(dtype=np.float32)[val_idx].reshape(-1, 1)).astype(np.float32)
    freq_test_scaled = freq_scaler.transform(test_frame["freq_log2_head"].to_numpy(dtype=np.float32).reshape(-1, 1)).astype(np.float32)

    y_fit = {
        "ratio_log2_head": ratio_fit_scaled[fit_idx],
        "ratio_class_head": train_ratio_class_indices[fit_idx],
        "freq_log2_head": freq_fit_scaled[fit_idx],
    }
    y_val = {
        "ratio_log2_head": ratio_val_scaled,
        "ratio_class_head": train_ratio_class_indices[val_idx],
        "freq_log2_head": freq_val_scaled,
    }
    y_test = {
        "ratio_log2_head": ratio_test_scaled,
        "ratio_class_head": test_ratio_class_indices,
        "freq_log2_head": freq_test_scaled,
    }

    algorithm_onehot = tf.keras.utils.to_categorical(frame["algorithm_idx"].to_numpy(dtype=np.int32), num_classes=len(algorithm_classes)).astype(np.float32)
    cond_fit = {"algorithm_condition_input": algorithm_onehot[fit_audio_indices]}
    cond_val = {"algorithm_condition_input": algorithm_onehot[val_audio_indices]}
    cond_test = {"algorithm_condition_input": algorithm_onehot[test_audio_indices]}

    np.save(output_dir / "train_audio_indices.npy", train_audio_indices)
    np.save(output_dir / "val_audio_indices.npy", val_audio_indices)
    np.save(output_dir / "test_audio_indices.npy", test_audio_indices)
    np.save(output_dir / "ratio_carrier_classes.npy", ratio_classes.astype(np.float32))
    train_frame.to_csv(output_dir / "y_train_pitch.csv", index=False)
    test_frame.to_csv(output_dir / "y_test_pitch.csv", index=False)

    class PitchSequence(tf.keras.utils.Sequence):
        def __init__(self, audio, sample_indices, y, conditions, batch_size, shuffle=True):
            self.audio = audio
            self.sample_indices = np.asarray(sample_indices, dtype=np.int32)
            self.y = y
            self.conditions = conditions
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
            audio_batch = normalize_audio_batch(self.audio[audio_ids]).reshape(len(audio_ids), audio_len, 1)
            y_batch = {name: values[batch_ids] for name, values in self.y.items()}
            x_batch = {"audio_input": audio_batch}
            for name, values in self.conditions.items():
                x_batch[name] = values[batch_ids]
            return x_batch, y_batch

        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.indices)

    audio_store_np = audio_store
    train_seq = PitchSequence(audio_store_np, fit_audio_indices, y_fit, cond_fit, BATCH_SIZE, shuffle=True)
    val_seq = PitchSequence(audio_store_np, val_audio_indices, y_val, cond_val, PRED_BATCH_SIZE, shuffle=False)
    test_seq = PitchSequence(audio_store_np, test_audio_indices, y_test, cond_test, PRED_BATCH_SIZE, shuffle=False)

    model = build_model(audio_len, len(algorithm_classes), len(ratio_classes))
    losses = {
        "ratio_log2_head": tf.keras.losses.Huber(delta=1.0),
        "ratio_class_head": tf.keras.losses.SparseCategoricalCrossentropy(),
        "freq_log2_head": tf.keras.losses.Huber(delta=1.0),
    }
    metrics = {
        "ratio_log2_head": ["mae"],
        "ratio_class_head": ["sparse_categorical_accuracy"],
        "freq_log2_head": ["mae"],
    }
    loss_weights = {
        "ratio_log2_head": 0.60,
        "ratio_class_head": 1.00,
        "freq_log2_head": 0.90,
    }
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=losses, metrics=metrics, loss_weights=loss_weights)

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
                    "target_group": "pitch",
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
        f"dataset={BASE_PATH}, group=pitch, available_samples={len(frame)}, epochs={EPOCHS}, batch_size={BATCH_SIZE}, "
        f"base_filters={BASE_FILTERS}, blocks={CNN_BLOCKS}, dense_units={DENSE_UNITS}, resume_epoch={resume_epoch}, ratio_classes={len(ratio_classes)}"
    )

    history = model.fit(train_seq, validation_data=val_seq, epochs=EPOCHS, initial_epoch=resume_epoch, callbacks=callbacks, verbose=FIT_VERBOSE)
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

    preds = model.predict(test_seq, verbose=0)
    ratio_log2_pred = np.asarray(preds[0], dtype=np.float32)
    ratio_class_pred = np.asarray(preds[1], dtype=np.float32)
    freq_log2_pred = np.asarray(preds[2], dtype=np.float32)
    ratio_class_idx_pred = np.argmax(ratio_class_pred, axis=1)
    ratio_class_values = ratio_classes[ratio_class_idx_pred]
    ratio_log2_pred_scaled = ratio_scaler.inverse_transform(ratio_log2_pred).reshape(-1)
    freq_log2_pred_scaled = freq_scaler.inverse_transform(freq_log2_pred).reshape(-1)
    ratio_log2_pred_raw = inverse_transform_series(ratio_log2_pred_scaled, "log2")
    freq_log2_pred_raw = inverse_transform_series(freq_log2_pred_scaled, "log2")

    true_ratio = test_frame["ratio_carrier"].to_numpy(dtype=np.float32)
    true_freq = test_frame["frequencia_base"].to_numpy(dtype=np.float32)
    true_ratio_log2 = test_frame["ratio_log2_head"].to_numpy(dtype=np.float32)
    true_freq_log2 = test_frame["freq_log2_head"].to_numpy(dtype=np.float32)
    ratio_class_true = test_ratio_class_indices

    final_predictions = {
        "sample_id": test_audio_indices.astype(int),
        "ratio_carrier_true": true_ratio,
        "ratio_carrier_pred": ratio_class_values.astype(np.float32),
        "ratio_class_true": ratio_class_true,
        "ratio_class_pred": ratio_class_idx_pred.astype(np.int32),
        "freq_true": true_freq,
        "freq_pred": freq_log2_pred_raw.astype(np.float32),
    }

    final_metrics = {
        "ratio_class_accuracy": float(np.mean(ratio_class_idx_pred == ratio_class_true)),
        "ratio_log2_head_mae": float(mean_absolute_error(true_ratio, ratio_class_values)),
        "ratio_log2_head_rmse": float(np.sqrt(mean_squared_error(true_ratio, ratio_class_values))),
        "ratio_regression_log2_mae": float(mean_absolute_error(true_ratio_log2, ratio_log2_pred_scaled)),
        "freq_log2_head_mae": float(mean_absolute_error(true_freq, freq_log2_pred_raw)),
        "freq_log2_head_rmse": float(np.sqrt(mean_squared_error(true_freq, freq_log2_pred_raw))),
        "freq_log2_head_log2_mae": float(mean_absolute_error(true_freq_log2, freq_log2_pred_scaled)),
    }

    pd.DataFrame(final_predictions).to_csv(output_dir / "predictions.csv", index=False)
    plot_metric(history_df, "loss", "val_loss", "Loss", "train_loss.png", output_dir)
    plot_metric(history_df, "ratio_class_head_sparse_categorical_accuracy", "val_ratio_class_head_sparse_categorical_accuracy", "Ratio Class Accuracy", "train_ratio_class_accuracy.png", output_dir)
    plot_metric(history_df, "ratio_log2_head_mae", "val_ratio_log2_head_mae", "Ratio Log2 MAE", "train_ratio_log2_mae.png", output_dir)
    plot_metric(history_df, "freq_log2_head_mae", "val_freq_log2_head_mae", "Frequency Log2 MAE", "train_freq_log2_mae.png", output_dir)

    results = {
        "model_name": MODEL_NAME,
        "dataset": str(BASE_PATH),
        "target_group": "pitch",
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
            "ratio_classes": len(ratio_classes),
        },
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "available_samples": int(len(frame)),
        "algorithm_classes": algorithm_classes,
        "ratio_classes": ratio_classes.astype(float).tolist(),
        "final_metrics": final_metrics,
        "target_specs": make_json_safe(PITCH_SPECS),
        "history": history_df.to_dict(orient="list"),
        "test_prediction_preview": pd.DataFrame(final_predictions).head(10).to_dict(orient="records"),
        "best_weights_path": str(CHECKPOINT_BEST_WEIGHTS),
        "latest_weights_path": str(CHECKPOINT_LATEST_WEIGHTS),
    }

    atomic_json_dump(output_dir / "results.json", results)
    with open(output_dir / "algorithm_map.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe({"classes": algorithm_classes, "mapping": algorithm_map}), f, indent=2, ensure_ascii=False)
    with open(output_dir / "ratio_carrier_classes.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe({"classes": ratio_classes.astype(float).tolist(), "mapping": ratio_class_map}), f, indent=2, ensure_ascii=False)
    joblib.dump(ratio_scaler, output_dir / "ratio_log2_head_scaler.joblib")
    joblib.dump(freq_scaler, output_dir / "freq_log2_head_scaler.joblib")

    print(f"Results written to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
