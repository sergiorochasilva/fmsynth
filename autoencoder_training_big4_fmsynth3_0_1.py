"""Train a raw-audio FM autoencoder on `dataset_big4`.

Architecture:
- 1D CNN encoder over the waveform
- latent bottleneck
- MLP decoder that reconstructs the waveform

Data flow:
- Input: `dataset_big4/parameters.csv` plus `sample_*.wav`
- Output: trained autoencoder and encoder `.keras` files, latent arrays,
  preprocessing bundle, history plots, and `results.json`
"""

import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    MaxPooling1D,
    Reshape,
)
from keras.models import Model
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.utils import plot_model

# Use a non-interactive backend for matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_PATH = "dataset_big4"
OUTPUT_DIR = "autoencoder_training_big4_fmsynth3_0_1"
MODEL_NAME = "autoencoder_training_big4_fmsynth3_0_1"
ENCODER_NAME = f"encoder_{MODEL_NAME}"

RANDOM_STATE = 0
TRAIN_FRAC = 0.75
VAL_FRAC = 0.2

BATCH_SIZE = int(os.getenv("AE_TRAIN_BATCH_SIZE", "2"))
PRED_BATCH_SIZE = int(os.getenv("AE_PRED_BATCH_SIZE", "4"))
EPOCHS = int(os.getenv("AE_EPOCHS", "180"))
LEARNING_RATE = float(os.getenv("AE_LEARNING_RATE", "1e-3"))

LATENT_DIM = int(os.getenv("AE_LATENT_DIM", "256"))
DECODER_HIDDEN_DIM = int(os.getenv("AE_DECODER_HIDDEN_DIM", "384"))
DECODER_DROPOUT = float(os.getenv("AE_DECODER_DROPOUT", "0.1"))

N_FFT = int(os.getenv("AE_N_FFT", "1024"))
HOP_LENGTH = int(os.getenv("AE_HOP_LENGTH", "256"))
WIN_LENGTH = int(os.getenv("AE_WIN_LENGTH", str(N_FFT)))
N_MELS = int(os.getenv("AE_N_MELS", "96"))
MEL_FMIN = float(os.getenv("AE_MEL_FMIN", "20.0"))

LOG_MEL_LOSS_WEIGHT = float(os.getenv("AE_LOG_MEL_LOSS_WEIGHT", "0.7"))
STFT_LOSS_WEIGHT = float(os.getenv("AE_STFT_LOSS_WEIGHT", "0.3"))

EPS = 1e-7
USE_MIXED_PRECISION = os.getenv("AE_MIXED_PRECISION", "1") == "1"
DISABLE_XLA_JIT = os.getenv("AE_DISABLE_XLA_JIT", "1") == "1"
AUDIO_DTYPE = os.getenv("AE_AUDIO_DTYPE", "float16").strip().lower()
SAVE_SPLIT_ARRAYS = os.getenv("AE_SAVE_SPLIT_ARRAYS", "0") == "1"
if AUDIO_DTYPE not in {"float16", "float32"}:
    raise ValueError("AE_AUDIO_DTYPE deve ser 'float16' ou 'float32'.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

if USE_MIXED_PRECISION:
    mixed_precision.set_global_policy("mixed_float16")
if DISABLE_XLA_JIT:
    tf.config.optimizer.set_jit(False)

print(
    "Runtime config: "
    f"batch_size={BATCH_SIZE}, pred_batch_size={PRED_BATCH_SIZE}, "
    f"mixed_precision={USE_MIXED_PRECISION}, policy={mixed_precision.global_policy().name}, "
    f"disable_xla_jit={DISABLE_XLA_JIT}, audio_dtype={AUDIO_DTYPE}, "
    f"save_split_arrays={SAVE_SPLIT_ARRAYS}"
)


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

    plt.figure(dpi=400)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.plot(history_df["epoch"], history_df[train_key], label=f"{ylabel} Training")
    plt.plot(history_df["epoch"], history_df[val_key], label=f"{ylabel} Validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=400)
    plt.close()


class IndexedAutoencoderSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        x: np.ndarray,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ):
        self.x = x
        self.indices = np.asarray(indices, dtype=np.int32).copy()
        self.batch_size = int(max(batch_size, 1))
        self.shuffle = bool(shuffle)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        batch_ids = self.indices[start:end]
        x_batch = self.x[batch_ids]
        return x_batch, x_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def encode_by_indices(
    encoder_model: Model,
    x: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    out = []
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        xb = x[indices[start:end]]
        zb = encoder_model.predict(xb, batch_size=batch_size, verbose=0)
        out.append(zb)
    if not out:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(out, axis=0)


def preprocess_audio(signal: np.ndarray, expected_len: int | None) -> tuple[np.ndarray, int]:
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    signal = np.asarray(signal, dtype=np.float32)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = 0.891 * signal / peak

    if expected_len is None:
        expected_len = int(signal.shape[0])

    if signal.shape[0] > expected_len:
        signal = signal[:expected_len]
    elif signal.shape[0] < expected_len:
        pad_width = expected_len - signal.shape[0]
        signal = np.pad(signal, (0, pad_width), mode="constant")

    if AUDIO_DTYPE == "float16":
        signal = signal.astype(np.float16)
    else:
        signal = signal.astype(np.float32)

    return signal, expected_len


def load_sample_ids(parameters_csv: str, tamanho_dataset: int) -> list[int]:
    target_raw = pd.read_csv(parameters_csv)
    if tamanho_dataset and tamanho_dataset < len(target_raw):
        target_raw = target_raw.iloc[:tamanho_dataset].copy()

    if "id" in target_raw.columns:
        return target_raw["id"].astype(int).tolist()
    return list(range(len(target_raw)))


def load_audio_dataset(base_path: str, sample_ids: list[int]) -> tuple[np.ndarray, int, int]:
    samples = []
    sample_rate_ref = None
    audio_len = None

    for idx, sample_id in enumerate(sample_ids):
        wav_path = os.path.join(base_path, f"sample_{sample_id}.wav")
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {wav_path}")

        signal, sample_rate = sf.read(wav_path)
        if sample_rate_ref is None:
            sample_rate_ref = int(sample_rate)
        elif int(sample_rate) != sample_rate_ref:
            raise ValueError(
                f"Sample rate inconsistente em {wav_path}: {sample_rate} vs {sample_rate_ref}"
            )

        signal, audio_len = preprocess_audio(signal, audio_len)
        samples.append(signal)

        if (idx + 1) % 1000 == 0:
            print(f"Carregados {idx + 1}/{len(sample_ids)} áudios")

    x = np.asarray(samples, dtype=np.float16 if AUDIO_DTYPE == "float16" else np.float32)
    if x.ndim != 2:
        raise ValueError(f"Formato de áudio inesperado: {x.shape}")

    return x, sample_rate_ref, int(audio_len)


def build_mel_matrix(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    mel_fmin: float,
    mel_fmax: float,
) -> tf.Tensor:
    n_spectrogram_bins = n_fft // 2 + 1
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=n_spectrogram_bins,
        sample_rate=float(sample_rate),
        lower_edge_hertz=float(mel_fmin),
        upper_edge_hertz=float(mel_fmax),
        dtype=tf.float32,
    )
    return tf.constant(mel_matrix, dtype=tf.float32)


def spectral_components(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    mel_matrix: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.float32)
    y_pred = tf.cast(tf.squeeze(y_pred, axis=-1), tf.float32)

    stft_true = tf.signal.stft(
        y_true,
        frame_length=WIN_LENGTH,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
        pad_end=True,
    )
    stft_pred = tf.signal.stft(
        y_pred,
        frame_length=WIN_LENGTH,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
        pad_end=True,
    )

    mag_true = tf.abs(stft_true) + EPS
    mag_pred = tf.abs(stft_pred) + EPS

    stft_mae = tf.reduce_mean(tf.abs(mag_true - mag_pred), axis=[1, 2])

    mel_true = tf.tensordot(tf.square(mag_true), mel_matrix, axes=[[-1], [0]])
    mel_pred = tf.tensordot(tf.square(mag_pred), mel_matrix, axes=[[-1], [0]])

    log_mel_true = tf.math.log(mel_true + EPS)
    log_mel_pred = tf.math.log(mel_pred + EPS)

    log_mel_mae = tf.reduce_mean(tf.abs(log_mel_true - log_mel_pred), axis=[1, 2])

    return log_mel_mae, stft_mae


def build_hybrid_loss_and_metrics(mel_matrix: tf.Tensor):
    def hybrid_loss(y_true, y_pred):
        log_mel_mae, stft_mae = spectral_components(y_true, y_pred, mel_matrix)
        return LOG_MEL_LOSS_WEIGHT * log_mel_mae + STFT_LOSS_WEIGHT * stft_mae

    def log_mel_mae_metric(y_true, y_pred):
        log_mel_mae, _stft_mae = spectral_components(y_true, y_pred, mel_matrix)
        return tf.reduce_mean(log_mel_mae)

    def stft_mae_metric(y_true, y_pred):
        _log_mel_mae, stft_mae = spectral_components(y_true, y_pred, mel_matrix)
        return tf.reduce_mean(stft_mae)

    hybrid_loss.__name__ = "hybrid_logmel_stft_loss"
    log_mel_mae_metric.__name__ = "log_mel_mae"
    stft_mae_metric.__name__ = "stft_mae"

    return hybrid_loss, log_mel_mae_metric, stft_mae_metric


def build_autoencoder(input_len: int) -> tuple[Model, Model]:
    input_layer = Input(shape=(input_len, 1), name="audio_input")

    # Encoder CNN
    x = BatchNormalization(name="enc_input_batch_norm")(input_layer)

    x = Conv1D(
        filters=32,
        kernel_size=11,
        strides=2,
        padding="same",
        activation="gelu",
        name="enc_conv_1",
    )(x)
    x = MaxPooling1D(pool_size=2, name="enc_pool_1")(x)

    x = Conv1D(
        filters=64,
        kernel_size=9,
        strides=1,
        padding="same",
        activation="gelu",
        name="enc_conv_2",
    )(x)
    x = MaxPooling1D(pool_size=2, name="enc_pool_2")(x)

    x = Conv1D(
        filters=96,
        kernel_size=7,
        strides=1,
        padding="same",
        activation="gelu",
        name="enc_conv_3",
    )(x)
    x = MaxPooling1D(pool_size=2, name="enc_pool_3")(x)

    x = Conv1D(
        filters=128,
        kernel_size=5,
        strides=1,
        dilation_rate=2,
        padding="same",
        activation="gelu",
        name="enc_conv_4_dilated",
    )(x)

    x = GlobalAveragePooling1D(name="enc_gap")(x)
    x = Dense(max(LATENT_DIM * 2, 256), activation="gelu", name="enc_dense")(x)
    latent = Dense(LATENT_DIM, activation=None, name="latent")(x)

    # Decoder MLP (sem CNN)
    d = Dense(DECODER_HIDDEN_DIM, activation="gelu", name="dec_dense_1")(latent)
    d = Dropout(DECODER_DROPOUT, name="dec_dropout")(d)
    d = Dense(input_len, activation="tanh", dtype="float32", name="dec_waveform_flat")(d)
    output = Reshape((input_len, 1), name="audio_recon")(d)

    autoencoder = Model(input_layer, output, name="cnn_autoencoder_big4_0_1")
    encoder = Model(input_layer, latent, name="cnn_encoder_big4_0_1")
    return autoencoder, encoder


def compute_test_spectral_metrics(
    model: Model,
    x: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, float, float]:
    mel_matrix = build_mel_matrix(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        mel_fmin=MEL_FMIN,
        mel_fmax=MEL_FMAX,
    )

    total = 0
    sum_log_mel = 0.0
    sum_stft = 0.0

    for start in range(0, len(indices), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(indices))
        xb = x[indices[start:end]]
        yb = model.predict(xb, batch_size=PRED_BATCH_SIZE, verbose=0)

        log_mel_mae, stft_mae = spectral_components(
            tf.convert_to_tensor(xb, dtype=tf.float32),
            tf.convert_to_tensor(yb, dtype=tf.float32),
            mel_matrix,
        )

        batch_size_eff = end - start
        total += batch_size_eff
        sum_log_mel += float(tf.reduce_sum(log_mel_mae).numpy())
        sum_stft += float(tf.reduce_sum(stft_mae).numpy())

    log_mel_mean = sum_log_mel / max(total, 1)
    stft_mean = sum_stft / max(total, 1)
    hybrid_mean = LOG_MEL_LOSS_WEIGHT * log_mel_mean + STFT_LOSS_WEIGHT * stft_mean

    return log_mel_mean, stft_mean, hybrid_mean


# -------------------------
# Leitura de metadados
# -------------------------
meta = {}
meta_path = os.path.join(BASE_PATH, "meta.json")
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

tamanho_dataset = int(meta.get("tamanho_dataset", 5000))
SAMPLE_RATE = int(meta.get("sample_rate_out", 16000))
MEL_FMAX = float(os.getenv("AE_MEL_FMAX", str(SAMPLE_RATE / 2.0)))

parameters_csv = os.path.join(BASE_PATH, "parameters.csv")
if not os.path.exists(parameters_csv):
    raise FileNotFoundError(f"Arquivo não encontrado: {parameters_csv}")

sample_ids = load_sample_ids(parameters_csv, tamanho_dataset)
print(f"Total de amostras para autoencoder: {len(sample_ids)}")

x_raw, sample_rate_found, audio_len = load_audio_dataset(BASE_PATH, sample_ids)
print(f"Shape bruto de áudio: {x_raw.shape}")
print(f"Sample rate encontrado: {sample_rate_found}")
print(f"Audio length: {audio_len}")

if sample_rate_found != SAMPLE_RATE:
    print(
        "Aviso: sample rate do áudio difere do meta. "
        f"meta={SAMPLE_RATE}, áudio={sample_rate_found}."
    )
    SAMPLE_RATE = sample_rate_found
    MEL_FMAX = float(os.getenv("AE_MEL_FMAX", str(SAMPLE_RATE / 2.0)))

# Ajustando dimensão de X
x_all = x_raw.reshape((x_raw.shape[0], x_raw.shape[1], 1))

# Split treino / teste / validação
n_samples = x_all.shape[0]
all_idx = np.arange(n_samples)

rng = np.random.default_rng(RANDOM_STATE)
rng.shuffle(all_idx)

train_size = int(TRAIN_FRAC * n_samples)
train_idx = all_idx[:train_size]
test_idx = all_idx[train_size:]

val_size = int(VAL_FRAC * train_size)
val_idx = train_idx[:val_size]
fit_idx = train_idx[val_size:]

print(f"Split: fit={len(fit_idx)} val={len(val_idx)} test={len(test_idx)}")

# Salvando arrays
if SAVE_SPLIT_ARRAYS:
    np.save(os.path.join(OUTPUT_DIR, "x_train_big4.npy"), x_all[train_idx])
    np.save(os.path.join(OUTPUT_DIR, "x_test_big4.npy"), x_all[test_idx])

# GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Loss/metrics
mel_matrix_train = build_mel_matrix(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    n_mels=N_MELS,
    mel_fmin=MEL_FMIN,
    mel_fmax=MEL_FMAX,
)
hybrid_loss, log_mel_mae_metric, stft_mae_metric = build_hybrid_loss_and_metrics(mel_matrix_train)

# Modelo
autoencoder, encoder = build_autoencoder(audio_len)

autoencoder.compile(
    optimizer=Nadam(learning_rate=LEARNING_RATE),
    loss=hybrid_loss,
    metrics=[log_mel_mae_metric, stft_mae_metric, "mae"],
)

plot_model(
    autoencoder,
    to_file=os.path.join(OUTPUT_DIR, "autoencoder_model.png"),
    show_shapes=True,
    expand_nested=True,
    rankdir="TB",
    dpi=250,
)

plot_model(
    encoder,
    to_file=os.path.join(OUTPUT_DIR, "encoder_model.png"),
    show_shapes=True,
    expand_nested=True,
    rankdir="TB",
    dpi=250,
)

# Treino
callbacks = [
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
]

train_sequence = IndexedAutoencoderSequence(
    x=x_all,
    indices=fit_idx,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_sequence = IndexedAutoencoderSequence(
    x=x_all,
    indices=val_idx,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

history = autoencoder.fit(
    train_sequence,
    epochs=EPOCHS,
    validation_data=val_sequence,
    callbacks=callbacks,
)

hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
hist.to_csv(os.path.join(OUTPUT_DIR, "train_history.csv"), index=False)

plot_metric(hist, "loss", "val_loss", "Hybrid Loss", "train_history_loss.png")
plot_metric(hist, "log_mel_mae", "val_log_mel_mae", "Log-Mel MAE", "train_history_logmel.png")
plot_metric(hist, "stft_mae", "val_stft_mae", "STFT MAE", "train_history_stft.png")
plot_metric(hist, "mae", "val_mae", "Waveform MAE", "train_history_wave_mae.png")

# Avaliação
test_sequence = IndexedAutoencoderSequence(
    x=x_all,
    indices=test_idx,
    batch_size=PRED_BATCH_SIZE,
    shuffle=False,
)

test_eval = autoencoder.evaluate(
    test_sequence,
    return_dict=True,
    verbose=0,
)

test_log_mel_mae, test_stft_mae, test_hybrid_recomputed = compute_test_spectral_metrics(
    autoencoder,
    x_all,
    test_idx,
)

print("===== Test Metrics =====")
print(f"loss (keras): {test_eval.get('loss')}")
print(f"mae (waveform): {test_eval.get('mae')}")
print(f"log_mel_mae: {test_log_mel_mae}")
print(f"stft_mae: {test_stft_mae}")
print(f"hybrid_recomputed: {test_hybrid_recomputed}")

# Extraindo embeddings para uso downstream
z_train = encode_by_indices(encoder, x_all, train_idx, PRED_BATCH_SIZE)
z_test = encode_by_indices(encoder, x_all, test_idx, PRED_BATCH_SIZE)

np.save(os.path.join(OUTPUT_DIR, "latent_train.npy"), z_train)
np.save(os.path.join(OUTPUT_DIR, "latent_test.npy"), z_test)

# Salvando modelos
autoencoder.save(os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.keras"))
encoder.save(os.path.join(OUTPUT_DIR, f"{ENCODER_NAME}.keras"))

preprocess_bundle = {
    "audio_len": int(audio_len),
    "sample_rate": int(SAMPLE_RATE),
    "n_fft": int(N_FFT),
    "hop_length": int(HOP_LENGTH),
    "win_length": int(WIN_LENGTH),
    "n_mels": int(N_MELS),
    "mel_fmin": float(MEL_FMIN),
    "mel_fmax": float(MEL_FMAX),
    "peak_norm": 0.891,
}
joblib.dump(
    preprocess_bundle,
    os.path.join(OUTPUT_DIR, f"preprocess_{MODEL_NAME}.save"),
)

history_last = {}
for col in hist.columns:
    if col == "epoch":
        continue
    try:
        history_last[col] = float(hist[col].iloc[-1])
    except (TypeError, ValueError):
        pass

best_val_loss_epoch = None
best_val_loss = None
if "val_loss" in hist:
    best_idx = int(hist["val_loss"].idxmin())
    best_val_loss_epoch = int(hist.loc[best_idx, "epoch"])
    best_val_loss = float(hist.loc[best_idx, "val_loss"])

results = {
    "model_name": MODEL_NAME,
    "encoder_name": ENCODER_NAME,
    "dataset": BASE_PATH,
    "tamanho_dataset": int(n_samples),
    "split": {
        "fit_size": int(len(fit_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "train_frac": float(TRAIN_FRAC),
        "val_frac_of_train": float(VAL_FRAC),
    },
    "runtime_config": {
        "batch_size": int(BATCH_SIZE),
        "pred_batch_size": int(PRED_BATCH_SIZE),
        "epochs": int(EPOCHS),
        "learning_rate": float(LEARNING_RATE),
        "mixed_precision": bool(USE_MIXED_PRECISION),
        "mixed_precision_policy": mixed_precision.global_policy().name,
        "disable_xla_jit": bool(DISABLE_XLA_JIT),
        "audio_dtype": AUDIO_DTYPE,
        "sequence_training": True,
        "save_split_arrays": bool(SAVE_SPLIT_ARRAYS),
    },
    "model_config": {
        "latent_dim": int(LATENT_DIM),
        "decoder_hidden_dim": int(DECODER_HIDDEN_DIM),
        "decoder_dropout": float(DECODER_DROPOUT),
    },
    "loss_config": {
        "log_mel_weight": float(LOG_MEL_LOSS_WEIGHT),
        "stft_weight": float(STFT_LOSS_WEIGHT),
        "n_fft": int(N_FFT),
        "hop_length": int(HOP_LENGTH),
        "win_length": int(WIN_LENGTH),
        "n_mels": int(N_MELS),
        "mel_fmin": float(MEL_FMIN),
        "mel_fmax": float(MEL_FMAX),
    },
    "audio_config": {
        "audio_len": int(audio_len),
        "sample_rate": int(SAMPLE_RATE),
    },
    "metrics": {
        "test_loss_keras": to_json_scalar(test_eval.get("loss", np.nan)),
        "test_waveform_mae": to_json_scalar(test_eval.get("mae", np.nan)),
        "test_log_mel_mae": float(test_log_mel_mae),
        "test_stft_mae": float(test_stft_mae),
        "test_hybrid_recomputed": float(test_hybrid_recomputed),
    },
    "history_last": history_last,
    "best_val_loss": {
        "epoch": best_val_loss_epoch,
        "value": best_val_loss,
    },
}

with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Treino do autoencoder concluído.")
print(f"Resultados em: {os.path.join(OUTPUT_DIR, 'results.json')}")
