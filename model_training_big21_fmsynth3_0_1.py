"""Separated hierarchical algorithm classifiers for `dataset_big19`, version 0_1.

Architecture:
- Raw waveform input
- Differentiable multi-resolution log-mel front-end inside the model
- Residual 1D CNN backbone over the time axis of stacked spectrogram features
- Three independently trained classifiers:
  - coarse `algorithm_family`
  - exact `series` algorithms
  - exact `parallel` algorithms

This variant removes the shared exact heads used by `big20` and trains each exact
classifier only on the samples from its own family.

Data flow:
- Input: `dataset_big19/parameters.csv` plus the contiguous prefix of rendered audio
- Output: trained weights for the three classifiers, predictions, learning curves, and `results.json`
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
from model_training_big20_fmsynth3_0_1 import (
    FAMILY_TO_ALGOS,
    group_stratify_key,
    load_audio_store,
    load_target_frame,
    logmel_frontend,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_PATH = Path(os.getenv("DATASET_PATH", "dataset_big19"))
MODEL_PREFIX = "model_training_big21_fmsynth3_0_1"
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "0"))
RANDOM_STATE = int(os.getenv("SEED", "42"))
TRAIN_FRAC = float(os.getenv("TRAIN_FRAC", "0.80"))
VAL_FRAC = float(os.getenv("VAL_FRAC", "0.15"))
BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "8"))
PRED_BATCH_SIZE = int(os.getenv("PRED_BATCH_SIZE", "12"))
EPOCHS = int(os.getenv("EPOCHS", "28"))
PATIENCE = int(os.getenv("PATIENCE", "6"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2.0e-4"))
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

SERIES_FAMILY = "series"
PARALLEL_FAMILY = "parallel"
SERIES_ALGOS = FAMILY_TO_ALGOS[SERIES_FAMILY]
PARALLEL_ALGOS = FAMILY_TO_ALGOS[PARALLEL_FAMILY]
FAMILY_CLASSES = sorted({SERIES_FAMILY, PARALLEL_FAMILY})

PARAMS_PATH = BASE_PATH / "parameters.csv"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", MODEL_PREFIX))

if USE_MIXED_PRECISION:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
if not ENABLE_XLA:
    tf.config.optimizer.set_jit(False)


def build_classifier(input_len: int, n_classes: int, model_name: str) -> Model:
    audio_input = Input(shape=(input_len, 1), name="audio_input")
    x = Lambda(logmel_frontend, output_shape=(None, N_MELS * 4), name=f"{model_name}_frontend")(audio_input)
    x = BatchNormalization(name=f"{model_name}_bn")(x)

    filters = BASE_FILTERS
    for block_idx in range(CNN_BLOCKS):
        residual = x
        kernel_size = 7 if block_idx == 0 else 5 if block_idx == 1 else 3
        dilation = 1 if block_idx < 2 else 2 if block_idx == 2 else 4
        x = Conv1D(filters, kernel_size=kernel_size, dilation_rate=dilation, padding="same", activation="swish", name=f"{model_name}_conv_{block_idx + 1}_a")(x)
        x = BatchNormalization(name=f"{model_name}_bn_{block_idx + 1}_a")(x)
        x = Conv1D(filters, kernel_size=3, padding="same", activation="swish", name=f"{model_name}_conv_{block_idx + 1}_b")(x)
        x = BatchNormalization(name=f"{model_name}_bn_{block_idx + 1}_b")(x)
        x = Conv1D(filters, kernel_size=1, padding="same", name=f"{model_name}_proj_{block_idx + 1}")(x)
        if residual.shape[-1] != filters:
            residual = Conv1D(filters, kernel_size=1, padding="same", name=f"{model_name}_res_proj_{block_idx + 1}")(residual)
        x = Add(name=f"{model_name}_res_add_{block_idx + 1}")([x, residual])
        x = BatchNormalization(name=f"{model_name}_bn_{block_idx + 1}_c")(x)
        if block_idx in {1, 3}:
            x = MaxPooling1D(pool_size=2, name=f"{model_name}_pool_{block_idx + 1}")(x)
        x = Dropout(DROPOUT, name=f"{model_name}_drop_{block_idx + 1}")(x)
        filters = min(filters + BASE_FILTERS, BASE_FILTERS * 3)

    gap = GlobalAveragePooling1D(name=f"{model_name}_gap")(x)
    gmp = GlobalMaxPooling1D(name=f"{model_name}_gmp")(x)
    x = Concatenate(name=f"{model_name}_concat")([gap, gmp])
    x = LayerNormalization(name=f"{model_name}_ln")(x)
    x = Dense(DENSE_UNITS, activation="swish", name=f"{model_name}_head_dense_1")(x)
    x = Dropout(DROPOUT, name=f"{model_name}_head_drop_1")(x)
    x = Dense(max(DENSE_UNITS // 2, 128), activation="swish", name=f"{model_name}_head_dense_2")(x)
    x = LayerNormalization(name=f"{model_name}_head_ln_2")(x)
    x = Dropout(DROPOUT, name=f"{model_name}_head_drop_2")(x)
    outputs = Dense(n_classes, activation="softmax", name=f"{model_name}_head")(x)
    return Model(audio_input, outputs, name=model_name)


class AudioClassSequence(tf.keras.utils.Sequence):
    def __init__(self, audio: np.ndarray, sample_indices: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
        super().__init__()
        self.audio = audio
        self.sample_indices = np.asarray(sample_indices, dtype=np.int32)
        self.y = np.asarray(y, dtype=np.int32)
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
        return audio_batch, self.y[batch_ids]

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


def build_stage_splits(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_indices = frame.index.to_numpy(dtype=np.int32)
    train_strata = group_stratify_key(frame, max_classes=max(2, int((1.0 - TRAIN_FRAC) * len(dataset_indices))))
    train_idx, test_idx = stratified_split_indices(
        dataset_indices,
        train_strata,
        test_size=1.0 - TRAIN_FRAC,
        random_state=RANDOM_STATE,
    )
    train_frame = frame.loc[train_idx].reset_index(drop=True)
    test_frame = frame.loc[test_idx].reset_index(drop=True)
    val_strata = group_stratify_key(train_frame, max_classes=max(2, int(VAL_FRAC * len(train_frame))))
    fit_idx, val_idx = stratified_split_indices(
        np.arange(len(train_frame), dtype=np.int32),
        val_strata,
        test_size=VAL_FRAC,
        random_state=RANDOM_STATE,
    )
    fit_frame = train_frame.loc[fit_idx].reset_index(drop=True)
    val_frame = train_frame.loc[val_idx].reset_index(drop=True)
    return fit_frame, val_frame, test_frame


def train_stage(
    stage_name: str,
    n_classes: int,
    audio_store: np.ndarray,
    frame_fit: pd.DataFrame,
    frame_val: pd.DataFrame,
    input_len: int,
    *,
    label_column: str,
) -> tuple[Model, dict]:
    stage_dir = OUTPUT_DIR / stage_name
    checkpoint_dir = stage_dir / "checkpoints"
    stage_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = build_classifier(input_len, n_classes, f"{MODEL_PREFIX}_{stage_name}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["sparse_categorical_accuracy"],
    )

    resume_state = load_json_file(checkpoint_dir / "training_state.json") if RESUME_TRAINING else None
    resume_epoch = 0
    latest_weights = checkpoint_dir / "latest.weights.h5"
    best_weights = checkpoint_dir / "best.weights.h5"
    if RESUME_TRAINING and resume_state and latest_weights.exists():
        try:
            model.load_weights(str(latest_weights))
            resume_epoch = int(resume_state.get("last_completed_epoch", -1)) + 1
        except Exception as exc:
            print(f"[{stage_name}] resume disabled: {exc}")
            resume_epoch = 0

    class ResumableCheckpointCallback(tf.keras.callbacks.Callback):
        def __init__(self, latest_path: Path, best_path: Path, state_path: Path, initial_state: dict | None = None):
            super().__init__()
            self.latest_path = latest_path
            self.best_path = best_path
            self.state_path = state_path
            self.last_completed_epoch = int(initial_state.get("last_completed_epoch", -1)) if initial_state else -1
            self.best_val_loss = float(initial_state.get("best_val_loss")) if initial_state and initial_state.get("best_val_loss") is not None else None
            self.best_val_loss_epoch = int(initial_state.get("best_val_loss_epoch")) if initial_state and initial_state.get("best_val_loss_epoch") is not None else None

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.last_completed_epoch = int(epoch)
            self.model.save_weights(self.latest_path)
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
                    self.model.save_weights(self.best_path)
            atomic_json_dump(
                self.state_path,
                {
                    "model_name": f"{MODEL_PREFIX}_{stage_name}",
                    "dataset": str(BASE_PATH),
                    "stage": stage_name,
                    "last_completed_epoch": self.last_completed_epoch,
                    "total_epochs_target": EPOCHS,
                    "best_val_loss": self.best_val_loss,
                    "best_val_loss_epoch": self.best_val_loss_epoch,
                },
            )

        def on_train_end(self, logs=None):
            self.model.save_weights(self.latest_path)

    train_seq = AudioClassSequence(
        audio_store,
        frame_fit["id"].astype(int).to_numpy(dtype=np.int32),
        frame_fit[label_column].to_numpy(dtype=np.int32),
        BATCH_SIZE,
        shuffle=True,
    )
    val_seq = AudioClassSequence(
        audio_store,
        frame_val["id"].astype(int).to_numpy(dtype=np.int32),
        frame_val[label_column].to_numpy(dtype=np.int32),
        PRED_BATCH_SIZE,
        shuffle=False,
    )

    callbacks = [
        ResumableCheckpointCallback(latest_weights, best_weights, checkpoint_dir / "training_state.json", resume_state),
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(PATIENCE // 2, 2), min_lr=1e-6, verbose=1),
    ]

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
    history_df.to_csv(stage_dir / "history.csv", index=False)

    if best_weights.exists():
        try:
            model.load_weights(str(best_weights))
        except Exception as exc:
            print(f"[{stage_name}] best-weight reload skipped: {exc}")

    model_path = stage_dir / f"{MODEL_PREFIX}_{stage_name}.keras"
    try:
        model.save(model_path)
    except Exception as exc:
        print(f"[{stage_name}] model save skipped: {exc}")

    return model, {
        "stage_dir": str(stage_dir),
        "resume_epoch": resume_epoch,
        "best_weights": str(best_weights),
        "history_rows": int(len(history_df)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train separated hierarchical algorithm classifiers for dataset_big19.")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    target_raw = load_target_frame(args.max_samples)
    if "algorithm_family" not in target_raw.columns:
        target_raw["algorithm_family"] = target_raw["algorithm"].map(
            lambda algo: SERIES_FAMILY if str(algo).startswith("series") else PARALLEL_FAMILY
        )

    family_map = {name: idx for idx, name in enumerate(FAMILY_CLASSES)}
    series_map = {name: idx for idx, name in enumerate(SERIES_ALGOS)}
    parallel_map = {name: idx for idx, name in enumerate(PARALLEL_ALGOS)}
    target_raw = target_raw.copy()
    target_raw["algorithm"] = target_raw["algorithm"].astype(str)
    target_raw["algorithm_family"] = target_raw["algorithm_family"].astype(str)
    target_raw["family_idx"] = target_raw["algorithm_family"].map(family_map).astype(np.int32)
    target_raw["series_idx"] = target_raw["algorithm"].map(series_map).fillna(0).astype(np.int32)
    target_raw["parallel_idx"] = target_raw["algorithm"].map(parallel_map).fillna(0).astype(np.int32)

    audio_store = load_audio_store(BASE_PATH)
    sample_ids = target_raw["id"].astype(int).tolist()
    audio_len = int(getattr(audio_store, "sample_len", 0) or np.asarray(audio_store[sample_ids[:1]]).shape[1])

    fit_frame, val_frame, test_frame = build_stage_splits(target_raw)

    tf.keras.backend.clear_session()
    family_model, family_stage_meta = train_stage(
        "family",
        len(FAMILY_CLASSES),
        audio_store,
        fit_frame,
        val_frame,
        audio_len,
        label_column="family_idx",
    )

    series_fit = fit_frame[fit_frame["algorithm_family"] == SERIES_FAMILY].reset_index(drop=True)
    series_val = val_frame[val_frame["algorithm_family"] == SERIES_FAMILY].reset_index(drop=True)
    tf.keras.backend.clear_session()
    series_model, series_stage_meta = train_stage(
        "series_exact",
        len(SERIES_ALGOS),
        audio_store,
        series_fit,
        series_val,
        audio_len,
        label_column="series_idx",
    )

    parallel_fit = fit_frame[fit_frame["algorithm_family"] == PARALLEL_FAMILY].reset_index(drop=True)
    parallel_val = val_frame[val_frame["algorithm_family"] == PARALLEL_FAMILY].reset_index(drop=True)
    tf.keras.backend.clear_session()
    parallel_model, parallel_stage_meta = train_stage(
        "parallel_exact",
        len(PARALLEL_ALGOS),
        audio_store,
        parallel_fit,
        parallel_val,
        audio_len,
        label_column="parallel_idx",
    )

    test_audio_indices = test_frame["id"].astype(int).to_numpy(dtype=np.int32)
    test_audio_seq = AudioOnlySequence(audio_store, test_audio_indices, PRED_BATCH_SIZE, shuffle=False)
    family_probs = family_model.predict(test_audio_seq, verbose=0)
    series_probs = series_model.predict(test_audio_seq, verbose=0)
    parallel_probs = parallel_model.predict(test_audio_seq, verbose=0)

    family_true = test_frame["family_idx"].to_numpy(dtype=np.int32)
    family_pred_idx = np.argmax(family_probs, axis=1).astype(np.int32)
    exact_true = test_frame["algorithm"].astype(str).tolist()

    oracle_pred: list[str] = []
    cascade_pred: list[str] = []
    oracle_prob = np.zeros(len(test_frame), dtype=np.float32)
    cascade_prob = np.zeros(len(test_frame), dtype=np.float32)

    for i, family_idx in enumerate(family_true):
        if FAMILY_CLASSES[int(family_idx)] == SERIES_FAMILY:
            local_true_probs = series_probs[i]
            oracle_pred.append(SERIES_ALGOS[int(np.argmax(local_true_probs))])
            oracle_prob[i] = float(np.max(local_true_probs))
        else:
            local_true_probs = parallel_probs[i]
            oracle_pred.append(PARALLEL_ALGOS[int(np.argmax(local_true_probs))])
            oracle_prob[i] = float(np.max(local_true_probs))

    for i, family_idx in enumerate(family_pred_idx):
        if FAMILY_CLASSES[int(family_idx)] == SERIES_FAMILY:
            local_probs = series_probs[i]
            cascade_pred.append(SERIES_ALGOS[int(np.argmax(local_probs))])
            cascade_prob[i] = float(np.max(local_probs))
        else:
            local_probs = parallel_probs[i]
            cascade_pred.append(PARALLEL_ALGOS[int(np.argmax(local_probs))])
            cascade_prob[i] = float(np.max(local_probs))

    series_mask = family_true == family_map[SERIES_FAMILY]
    parallel_mask = family_true == family_map[PARALLEL_FAMILY]
    series_oracle_acc = (
        float(np.mean(np.argmax(series_probs[series_mask], axis=1) == test_frame.loc[series_mask, "series_idx"].to_numpy(dtype=np.int32)))
        if np.any(series_mask)
        else None
    )
    parallel_oracle_acc = (
        float(np.mean(np.argmax(parallel_probs[parallel_mask], axis=1) == test_frame.loc[parallel_mask, "parallel_idx"].to_numpy(dtype=np.int32)))
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
        "algorithm_family_true": [FAMILY_CLASSES[int(i)] for i in family_true],
        "algorithm_family_pred": [FAMILY_CLASSES[int(i)] for i in family_pred_idx],
        "algorithm_family_prob": np.max(family_probs, axis=1).tolist(),
    }
    pd.DataFrame(final_predictions).to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    final_metrics = {
        "algorithm_family_accuracy": float(np.mean(family_pred_idx == family_true)),
        "algorithm_exact_accuracy_oracle": float(np.mean(np.asarray(oracle_pred, dtype=object) == np.asarray(exact_true, dtype=object))),
        "algorithm_exact_accuracy_cascade": float(np.mean(np.asarray(cascade_pred, dtype=object) == np.asarray(exact_true, dtype=object))),
        "series_exact_accuracy_oracle": series_oracle_acc,
        "parallel_exact_accuracy_oracle": parallel_oracle_acc,
    }

    results = {
        "model_name": MODEL_PREFIX,
        "dataset": str(BASE_PATH),
        "hierarchical": "separate_models",
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
            "mixed_precision": USE_MIXED_PRECISION,
            "resume_training": RESUME_TRAINING,
        },
        "train_size": int(len(fit_frame)),
        "val_size": int(len(val_frame)),
        "test_size": int(len(test_frame)),
        "available_samples": int(len(target_raw)),
        "family_classes": FAMILY_CLASSES,
        "series_algorithms": SERIES_ALGOS,
        "parallel_algorithms": PARALLEL_ALGOS,
        "stages": {
            "family": family_stage_meta,
            "series_exact": series_stage_meta,
            "parallel_exact": parallel_stage_meta,
        },
        "final_metrics": final_metrics,
    }

    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe(results), f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "family_map.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe({"classes": FAMILY_CLASSES, "mapping": family_map}), f, indent=2, ensure_ascii=False)
    with open(OUTPUT_DIR / "series_map.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe({"classes": SERIES_ALGOS, "mapping": series_map}), f, indent=2, ensure_ascii=False)
    with open(OUTPUT_DIR / "parallel_map.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe({"classes": PARALLEL_ALGOS, "mapping": parallel_map}), f, indent=2, ensure_ascii=False)

    print(f"Results written to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
