"""Stable raw-audio-to-algorithm classifier for `dataset_big5`.

Architecture:
- Input remains raw waveform audio (`sample_*.wav`), so the experiment is still
  end-to-end from the repository dataset.
- A deterministic differentiable log-mel frontend inside the model converts the
  waveform to a time-frequency representation.
- A compact 2D CNN encoder classifies only the `algorithm` used by `fm_synth3`.

Why this variant exists:
- Version `0_1` used a Conv1D encoder directly over waveform samples and became
  numerically unstable (`NaN` loss) with mixed precision and a high learning rate.
- This version disables mixed precision by default, lowers the learning rate,
  clips gradients, uses a stratified split by `algorithm`, and stops on non-finite
  metrics.

Data flow:
- Input: `dataset_big5/parameters.csv` and `dataset_big5/sample_*.wav`
- Output: classifier `.keras`, encoder `.keras`, latent exports, prediction
  tables, history plots, and `results.json`
"""

import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from keras.models import Model
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_PATH = os.getenv("MEC_BASE_PATH", "dataset_big5")
OUTPUT_DIR = os.getenv(
    "MEC_OUTPUT_DIR",
    "model_encoder_classification_training_big5_fmsynth3_0_2",
)
MODEL_NAME = "model_encoder_classification_training_big5_fmsynth3_0_2"
ENCODER_NAME = f"encoder_{MODEL_NAME}"
TARGET_COL = "algorithm"

RANDOM_STATE = int(os.getenv("MEC_RANDOM_STATE", "0"))
TRAIN_FRAC = float(os.getenv("MEC_TRAIN_FRAC", "0.75"))
VAL_FRAC = float(os.getenv("MEC_VAL_FRAC", "0.2"))
MAX_SAMPLES = int(os.getenv("MEC_MAX_SAMPLES", "0"))

BATCH_SIZE = int(os.getenv("MEC_BATCH_SIZE", "8"))
PRED_BATCH_SIZE = int(os.getenv("MEC_PRED_BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("MEC_EPOCHS", "120"))
LEARNING_RATE = float(os.getenv("MEC_LEARNING_RATE", "3e-4"))
CLIPNORM = float(os.getenv("MEC_CLIPNORM", "1.0"))

LATENT_DIM = int(os.getenv("MEC_LATENT_DIM", "192"))
CLASSIFIER_DROPOUT = float(os.getenv("MEC_CLASSIFIER_DROPOUT", "0.30"))

N_FFT = int(os.getenv("MEC_N_FFT", "1024"))
HOP_LENGTH = int(os.getenv("MEC_HOP_LENGTH", "256"))
WIN_LENGTH = int(os.getenv("MEC_WIN_LENGTH", str(N_FFT)))
N_MELS = int(os.getenv("MEC_N_MELS", "96"))
MEL_FMIN = float(os.getenv("MEC_MEL_FMIN", "20.0"))

EPS = float(os.getenv("MEC_EPS", "1e-6"))
USE_MIXED_PRECISION = os.getenv("MEC_MIXED_PRECISION", "0") == "1"
DISABLE_XLA_JIT = os.getenv("MEC_DISABLE_XLA_JIT", "1") == "1"
AUDIO_DTYPE = os.getenv("MEC_AUDIO_DTYPE", "float32").strip().lower()
SAVE_SPLIT_ARRAYS = os.getenv("MEC_SAVE_SPLIT_ARRAYS", "0") == "1"
SAVE_LATENT_ARRAYS = os.getenv("MEC_SAVE_LATENT_ARRAYS", "1") == "1"

ALGORITHM_MERGE_MAP = {
    "dual_chain": "series2x2_parallel1",
}

if AUDIO_DTYPE not in {"float16", "float32"}:
    raise ValueError("MEC_AUDIO_DTYPE must be 'float16' or 'float32'.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

if USE_MIXED_PRECISION:
    mixed_precision.set_global_policy("mixed_float16")
else:
    mixed_precision.set_global_policy("float32")

if DISABLE_XLA_JIT:
    tf.config.optimizer.set_jit(False)

print(
    "Runtime config: "
    f"batch_size={BATCH_SIZE}, pred_batch_size={PRED_BATCH_SIZE}, epochs={EPOCHS}, "
    f"lr={LEARNING_RATE}, clipnorm={CLIPNORM}, mixed_precision={USE_MIXED_PRECISION}, "
    f"policy={mixed_precision.global_policy().name}, disable_xla_jit={DISABLE_XLA_JIT}, "
    f"audio_dtype={AUDIO_DTYPE}, max_samples={MAX_SAMPLES}, target={TARGET_COL}"
)


def cat_output_name(column_name: str) -> str:
    return f"cat__{column_name}"


def to_json_scalar(value):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def plot_metric(history_df, train_key, val_key, ylabel, filename):
    if train_key not in history_df.columns or val_key not in history_df.columns:
        return

    plt.figure(dpi=350)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.plot(history_df["epoch"], history_df[train_key], label=f"{ylabel} Training")
    plt.plot(history_df["epoch"], history_df[val_key], label=f"{ylabel} Validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=350)
    plt.close()


def preprocess_audio(signal: np.ndarray, expected_len: int | None) -> tuple[np.ndarray, int]:
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    signal = np.asarray(signal, dtype=np.float32)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = 0.891 * signal / peak

    if expected_len is None:
        expected_len = int(signal.shape[0])

    if signal.shape[0] > expected_len:
        signal = signal[:expected_len]
    elif signal.shape[0] < expected_len:
        signal = np.pad(signal, (0, expected_len - signal.shape[0]), mode="constant")

    return signal.astype(np.float16 if AUDIO_DTYPE == "float16" else np.float32), expected_len


def load_audio_dataset(base_path: str, sample_ids: list[int]) -> tuple[np.ndarray, int, int]:
    samples = []
    sample_rate_ref = None
    audio_len = None

    for idx, sample_id in enumerate(sample_ids):
        wav_path = os.path.join(base_path, f"sample_{sample_id}.wav")
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        signal, sample_rate = sf.read(wav_path)
        if sample_rate_ref is None:
            sample_rate_ref = int(sample_rate)
        elif int(sample_rate) != sample_rate_ref:
            raise ValueError(
                f"Inconsistent sample rate in {wav_path}: {sample_rate} vs {sample_rate_ref}"
            )

        signal, audio_len = preprocess_audio(signal, audio_len)
        samples.append(signal)

        if (idx + 1) % 1000 == 0:
            print(f"Loaded {idx + 1}/{len(sample_ids)} audio files")

    x = np.asarray(samples, dtype=np.float16 if AUDIO_DTYPE == "float16" else np.float32)
    if x.ndim != 2:
        raise ValueError(f"Unexpected audio shape: {x.shape}")

    return x, sample_rate_ref, int(audio_len)


def stratified_train_val_test_indices(
    labels: np.ndarray,
    train_frac: float,
    val_frac: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    fit_parts = []
    val_parts = []
    test_parts = []

    for label in np.unique(labels):
        label_idx = np.where(labels == label)[0]
        rng.shuffle(label_idx)

        train_size = int(round(train_frac * len(label_idx)))
        train_size = min(max(train_size, 1), len(label_idx) - 1)
        train_idx = label_idx[:train_size]
        test_idx = label_idx[train_size:]

        val_size = int(round(val_frac * len(train_idx)))
        val_size = min(max(val_size, 1), len(train_idx) - 1)

        val_parts.append(train_idx[:val_size])
        fit_parts.append(train_idx[val_size:])
        test_parts.append(test_idx)

    fit_idx = np.concatenate(fit_parts).astype(np.int32)
    val_idx = np.concatenate(val_parts).astype(np.int32)
    test_idx = np.concatenate(test_parts).astype(np.int32)

    rng.shuffle(fit_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return fit_idx, val_idx, test_idx


class ClassificationAudioSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ):
        super().__init__()
        self.x = x
        self.labels = np.asarray(labels, dtype=np.int32)
        self.indices = np.asarray(indices, dtype=np.int32).copy()
        self.batch_size = int(max(batch_size, 1))
        self.shuffle = bool(shuffle)
        self.output_name = cat_output_name(TARGET_COL)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        batch_ids = self.indices[start:end]
        return self.x[batch_ids], {self.output_name: self.labels[batch_ids]}

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class LogMelSpectrogram(tf.keras.layers.Layer):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        n_mels: int,
        mel_fmin: float,
        mel_fmax: float,
        eps: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.n_mels = int(n_mels)
        self.mel_fmin = float(mel_fmin)
        self.mel_fmax = float(mel_fmax)
        self.eps = float(eps)

    def build(self, input_shape):
        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=self.n_fft // 2 + 1,
            sample_rate=float(self.sample_rate),
            lower_edge_hertz=self.mel_fmin,
            upper_edge_hertz=self.mel_fmax,
            dtype=tf.float32,
        )
        self.mel_matrix = tf.constant(mel_matrix, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        waveform = tf.cast(inputs, tf.float32)
        waveform = tf.squeeze(waveform, axis=-1)

        stft = tf.signal.stft(
            waveform,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
            window_fn=tf.signal.hann_window,
            pad_end=True,
        )
        magnitude = tf.abs(stft)
        mel_power = tf.tensordot(tf.square(magnitude), self.mel_matrix, axes=[[-1], [0]])
        log_mel = tf.math.log(mel_power + self.eps)

        mean = tf.reduce_mean(log_mel, axis=[1, 2], keepdims=True)
        std = tf.math.reduce_std(log_mel, axis=[1, 2], keepdims=True)
        normalized = (log_mel - mean) / (std + self.eps)
        normalized = tf.expand_dims(normalized, axis=-1)
        return normalized

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sample_rate": self.sample_rate,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "win_length": self.win_length,
                "n_mels": self.n_mels,
                "mel_fmin": self.mel_fmin,
                "mel_fmax": self.mel_fmax,
                "eps": self.eps,
            }
        )
        return config


class NonFiniteMetricStopper(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            if value is not None and not np.isfinite(float(value)):
                print(f"Stopping at epoch {epoch}: non-finite metric {key}={value}")
                self.model.stop_training = True
                return


def conv_block(x, filters: int, name: str, pool: bool = True):
    x = Conv2D(filters, kernel_size=(3, 3), padding="same", activation="gelu", name=f"{name}_conv")(x)
    x = BatchNormalization(name=f"{name}_bn")(x)
    if pool:
        x = MaxPooling2D(pool_size=(2, 2), name=f"{name}_pool")(x)
    return x


def build_model(input_len: int, sample_rate: int, n_classes: int) -> tuple[Model, Model]:
    input_layer = Input(shape=(input_len, 1), name="audio_input")

    x = LogMelSpectrogram(
        sample_rate=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        mel_fmin=MEL_FMIN,
        mel_fmax=MEL_FMAX,
        eps=EPS,
        name="log_mel_frontend",
    )(input_layer)

    x = BatchNormalization(name="mel_batch_norm")(x)
    x = conv_block(x, 32, "enc_block_1")
    x = Dropout(0.10, name="enc_dropout_1")(x)
    x = conv_block(x, 64, "enc_block_2")
    x = Dropout(0.15, name="enc_dropout_2")(x)
    x = conv_block(x, 96, "enc_block_3")
    x = conv_block(x, 128, "enc_block_4", pool=False)

    x = GlobalAveragePooling2D(name="enc_gap")(x)
    x = Dense(256, activation="gelu", name="enc_dense")(x)
    x = Dropout(CLASSIFIER_DROPOUT, name="enc_dense_dropout")(x)
    latent = Dense(LATENT_DIM, activation="gelu", name="latent")(x)

    c = BatchNormalization(name="cls_latent_batch_norm")(latent)
    c = Dense(160, activation="gelu", name="cls_dense_1")(c)
    c = Dropout(CLASSIFIER_DROPOUT, name="cls_dropout_1")(c)
    c = Dense(96, activation="gelu", name="cls_dense_2")(c)
    output = Dense(
        n_classes,
        activation="softmax",
        dtype="float32",
        name=cat_output_name(TARGET_COL),
    )(c)

    model = Model(input_layer, {cat_output_name(TARGET_COL): output}, name=MODEL_NAME)
    encoder = Model(input_layer, latent, name=ENCODER_NAME)
    return model, encoder


def encode_by_indices(
    encoder_model: Model,
    x: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    out = []
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        z_batch = encoder_model.predict(x[indices[start:end]], batch_size=batch_size, verbose=0)
        out.append(z_batch)
    if not out:
        return np.zeros((0, LATENT_DIM), dtype=np.float32)
    return np.concatenate(out, axis=0)


def safe_plot_model(model: Model, filename: str):
    try:
        plot_model(
            model,
            to_file=os.path.join(OUTPUT_DIR, filename),
            show_shapes=True,
            expand_nested=True,
            rankdir="TB",
            dpi=220,
        )
    except Exception as exc:
        print(f"Warning: could not render {filename}: {exc}")


meta = {}
meta_path = os.path.join(BASE_PATH, "meta.json")
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

tamanho_dataset = int(meta.get("tamanho_dataset", 5000))
if MAX_SAMPLES > 0:
    tamanho_dataset = min(tamanho_dataset, MAX_SAMPLES)

SAMPLE_RATE = int(meta.get("sample_rate_out", 16000))
MEL_FMAX = float(os.getenv("MEC_MEL_FMAX", str(SAMPLE_RATE / 2.0)))

params_path = os.path.join(BASE_PATH, "parameters.csv")
if not os.path.exists(params_path):
    raise FileNotFoundError(f"File not found: {params_path}")

params_raw = pd.read_csv(params_path)
if TARGET_COL not in params_raw.columns:
    raise ValueError(f"Target column not found: {TARGET_COL}")

params_raw[TARGET_COL] = params_raw[TARGET_COL].replace(ALGORITHM_MERGE_MAP)

if "id" in params_raw.columns:
    params_raw = params_raw.sort_values("id").reset_index(drop=True)

if tamanho_dataset and tamanho_dataset < len(params_raw):
    params_raw = params_raw.iloc[:tamanho_dataset].copy()

sample_ids = params_raw["id"].astype(int).tolist() if "id" in params_raw.columns else list(range(len(params_raw)))
print(f"Total samples for stable classifier: {len(sample_ids)}")

x_raw, sample_rate_found, audio_len = load_audio_dataset(BASE_PATH, sample_ids)
print(f"Raw audio shape: {x_raw.shape}")
print(f"Detected sample rate: {sample_rate_found}")
print(f"Audio length: {audio_len}")

if sample_rate_found != SAMPLE_RATE:
    print(
        "Warning: audio sample rate differs from meta. "
        f"meta={SAMPLE_RATE}, audio={sample_rate_found}."
    )
    SAMPLE_RATE = sample_rate_found
    MEL_FMAX = float(os.getenv("MEC_MEL_FMAX", str(SAMPLE_RATE / 2.0)))

x_all = x_raw.reshape((x_raw.shape[0], x_raw.shape[1], 1))
n_samples = x_all.shape[0]

target_cat = pd.Categorical(params_raw[TARGET_COL])
categorical_map = [str(x) for x in target_cat.categories]
y_all = target_cat.codes.astype(np.int32)
n_classes = len(categorical_map)

fit_idx, val_idx, test_idx = stratified_train_val_test_indices(
    labels=y_all,
    train_frac=TRAIN_FRAC,
    val_frac=VAL_FRAC,
    random_state=RANDOM_STATE,
)
train_idx = np.concatenate([fit_idx, val_idx]).astype(np.int32)

print(f"Split: fit={len(fit_idx)} val={len(val_idx)} test={len(test_idx)}")

if SAVE_SPLIT_ARRAYS:
    np.save(os.path.join(OUTPUT_DIR, "x_train_big5.npy"), x_all[train_idx])
    np.save(os.path.join(OUTPUT_DIR, "x_test_big5.npy"), x_all[test_idx])

train_sequence = ClassificationAudioSequence(
    x=x_all,
    labels=y_all,
    indices=fit_idx,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_sequence = ClassificationAudioSequence(
    x=x_all,
    labels=y_all,
    indices=val_idx,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
test_sequence = ClassificationAudioSequence(
    x=x_all,
    labels=y_all,
    indices=test_idx,
    batch_size=PRED_BATCH_SIZE,
    shuffle=False,
)

model, encoder = build_model(audio_len, SAMPLE_RATE, n_classes)
head_name = cat_output_name(TARGET_COL)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM),
    loss={head_name: tf.keras.losses.SparseCategoricalCrossentropy()},
    metrics={head_name: ["sparse_categorical_accuracy"]},
)

safe_plot_model(model, "model.png")
safe_plot_model(encoder, "encoder_model.png")

callbacks = [
    TerminateOnNaN(),
    NonFiniteMetricStopper(),
    EarlyStopping(monitor="val_loss", patience=18, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
]

history = model.fit(
    train_sequence,
    epochs=EPOCHS,
    validation_data=val_sequence,
    callbacks=callbacks,
    verbose=1,
)

hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch

plot_metric(hist, "loss", "val_loss", "Total Loss", "train_history_loss.png")

acc_key = "sparse_categorical_accuracy"
val_acc_key = "val_sparse_categorical_accuracy"
if acc_key not in hist.columns:
    acc_key = f"{head_name}_sparse_categorical_accuracy"
    val_acc_key = f"val_{head_name}_sparse_categorical_accuracy"

plot_metric(
    hist,
    acc_key,
    val_acc_key,
    "Algorithm Accuracy",
    "train_history_algorithm_acc.png",
)

test_eval = model.evaluate(test_sequence, return_dict=True, verbose=0)

raw_pred = model.predict(test_sequence, batch_size=PRED_BATCH_SIZE, verbose=0)
if isinstance(raw_pred, dict):
    y_prob = np.asarray(raw_pred[head_name], dtype=np.float64)
elif isinstance(raw_pred, list):
    y_prob = np.asarray(raw_pred[0], dtype=np.float64)
else:
    y_prob = np.asarray(raw_pred, dtype=np.float64)

y_true = y_all[test_idx].astype(np.int32)
y_pred = np.argmax(y_prob, axis=1).astype(np.int32)
accuracy = float((y_pred == y_true).mean())
crossentropy = float(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_prob).numpy().mean())

pred_codes_df = pd.DataFrame(index=np.arange(len(test_idx)))
if "id" in params_raw.columns:
    pred_codes_df["id"] = params_raw.iloc[test_idx]["id"].to_numpy(dtype=np.int32)
pred_codes_df[f"true__{TARGET_COL}"] = y_true
pred_codes_df[f"pred__{TARGET_COL}"] = y_pred

pred_labels_df = pred_codes_df.copy()
pred_labels_df[f"true_label__{TARGET_COL}"] = pred_codes_df[f"true__{TARGET_COL}"].map(
    lambda x: categorical_map[int(x)] if 0 <= int(x) < len(categorical_map) else "<UNK>"
)
pred_labels_df[f"pred_label__{TARGET_COL}"] = pred_codes_df[f"pred__{TARGET_COL}"].map(
    lambda x: categorical_map[int(x)] if 0 <= int(x) < len(categorical_map) else "<UNK>"
)

print("===== Algorithm Test Metrics =====")
print(f"accuracy: {accuracy:.6f}")
print(f"crossentropy: {crossentropy:.6f}")

model.save(os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.keras"))
encoder.save(os.path.join(OUTPUT_DIR, f"{ENCODER_NAME}.keras"))

if SAVE_LATENT_ARRAYS:
    z_train = encode_by_indices(encoder, x_all, train_idx, PRED_BATCH_SIZE)
    z_test = encode_by_indices(encoder, x_all, test_idx, PRED_BATCH_SIZE)
    np.save(os.path.join(OUTPUT_DIR, "latent_train.npy"), z_train)
    np.save(os.path.join(OUTPUT_DIR, "latent_test.npy"), z_test)

preprocess_bundle = {
    "audio_len": int(audio_len),
    "sample_rate": int(SAMPLE_RATE),
    "peak_norm": 0.891,
    "target_col": TARGET_COL,
    "categorical_map": categorical_map,
    "algorithm_merge_map": ALGORITHM_MERGE_MAP,
    "frontend": {
        "kind": "log_mel",
        "n_fft": int(N_FFT),
        "hop_length": int(HOP_LENGTH),
        "win_length": int(WIN_LENGTH),
        "n_mels": int(N_MELS),
        "mel_fmin": float(MEL_FMIN),
        "mel_fmax": float(MEL_FMAX),
        "per_sample_standardization": True,
    },
}

joblib.dump(
    preprocess_bundle,
    os.path.join(OUTPUT_DIR, f"preprocess_{MODEL_NAME}.save"),
)
joblib.dump(
    {
        "target_col": TARGET_COL,
        "categorical_map": categorical_map,
        "algorithm_merge_map": ALGORITHM_MERGE_MAP,
    },
    os.path.join(OUTPUT_DIR, f"target_preprocess_{MODEL_NAME}.save"),
)

pred_codes_df.to_csv(os.path.join(OUTPUT_DIR, "params_pred_test_codes.csv"), index=False)
pred_labels_df.to_csv(os.path.join(OUTPUT_DIR, "params_pred_test_labels.csv"), index=False)
hist.to_csv(os.path.join(OUTPUT_DIR, "train_history.csv"), index=False)

history_last = {}
for col in hist.columns:
    if col == "epoch":
        continue
    try:
        value = float(hist[col].iloc[-1])
    except (TypeError, ValueError):
        continue
    history_last[col] = value if np.isfinite(value) else None

best_val_loss_epoch = None
best_val_loss = None
if "val_loss" in hist and hist["val_loss"].notna().any():
    finite_val_loss = hist[np.isfinite(hist["val_loss"].astype(float))]
    if not finite_val_loss.empty:
        best_idx = int(finite_val_loss["val_loss"].idxmin())
        best_val_loss_epoch = int(hist.loc[best_idx, "epoch"])
        best_val_loss = float(hist.loc[best_idx, "val_loss"])

results = {
    "model_name": MODEL_NAME,
    "encoder_name": ENCODER_NAME,
    "dataset": BASE_PATH,
    "n_samples": int(n_samples),
    "split": {
        "fit_size": int(len(fit_idx)),
        "val_size": int(len(val_idx)),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "train_frac": float(TRAIN_FRAC),
        "val_frac_of_train": float(VAL_FRAC),
        "strategy": "stratified_by_algorithm",
    },
    "runtime_config": {
        "batch_size": int(BATCH_SIZE),
        "pred_batch_size": int(PRED_BATCH_SIZE),
        "epochs": int(EPOCHS),
        "learning_rate": float(LEARNING_RATE),
        "clipnorm": float(CLIPNORM),
        "mixed_precision": bool(USE_MIXED_PRECISION),
        "mixed_precision_policy": mixed_precision.global_policy().name,
        "disable_xla_jit": bool(DISABLE_XLA_JIT),
        "audio_dtype": AUDIO_DTYPE,
        "save_split_arrays": bool(SAVE_SPLIT_ARRAYS),
        "save_latent_arrays": bool(SAVE_LATENT_ARRAYS),
    },
    "architecture": {
        "input": "raw_waveform",
        "frontend": "differentiable_log_mel",
        "encoder": "compact_2d_cnn",
        "target": TARGET_COL,
        "latent_dim": int(LATENT_DIM),
    },
    "algorithm_label_merges": ALGORITHM_MERGE_MAP,
    "categorical_cols": [TARGET_COL],
    "categorical_accuracy": {TARGET_COL: accuracy},
    "categorical_crossentropy": {TARGET_COL: crossentropy},
    "metrics": {
        "categorical_accuracy_mean": accuracy,
        "categorical_crossentropy_mean": crossentropy,
    },
    "test_metrics_by_head": {
        TARGET_COL: {
            "accuracy": accuracy,
            "crossentropy": crossentropy,
            "n_classes": int(n_classes),
        }
    },
    "keras_test_eval": {k: to_json_scalar(v) for k, v in test_eval.items()},
    "history_last": history_last,
    "best_val_loss": {
        "epoch": best_val_loss_epoch,
        "value": best_val_loss,
    },
    "categorical_maps": {TARGET_COL: categorical_map},
    "artifacts": {
        "model": os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.keras"),
        "encoder": os.path.join(OUTPUT_DIR, f"{ENCODER_NAME}.keras"),
        "preprocess": os.path.join(OUTPUT_DIR, f"preprocess_{MODEL_NAME}.save"),
        "target_preprocess": os.path.join(OUTPUT_DIR, f"target_preprocess_{MODEL_NAME}.save"),
        "results": os.path.join(OUTPUT_DIR, "results.json"),
        "history": os.path.join(OUTPUT_DIR, "train_history.csv"),
        "pred_codes": os.path.join(OUTPUT_DIR, "params_pred_test_codes.csv"),
        "pred_labels": os.path.join(OUTPUT_DIR, "params_pred_test_labels.csv"),
        "latent_train": os.path.join(OUTPUT_DIR, "latent_train.npy") if SAVE_LATENT_ARRAYS else None,
        "latent_test": os.path.join(OUTPUT_DIR, "latent_test.npy") if SAVE_LATENT_ARRAYS else None,
    },
}

with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Stable encoder + algorithm classification training finished.")
print(f"Results at: {os.path.join(OUTPUT_DIR, 'results.json')}")
