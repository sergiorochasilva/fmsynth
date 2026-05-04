"""Two-stage training on `dataset_big5`: autoencoder pretrain, then classifier fine-tune.

Architecture:
- Stage 1: raw-audio autoencoder with a shared CNN encoder and waveform decoder
- Stage 2: reuse the pretrained encoder weights and attach categorical classification heads

Data flow:
- Input: `dataset_big5/parameters.csv` and `sample_*.wav`
- Stage 1 outputs: autoencoder `.keras`, encoder `.keras`, and pretraining metrics
- Stage 2 outputs: fine-tuned classifier `.keras`, encoder `.keras`, latent exports, prediction tables, and `results.json`
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_PATH = os.getenv("MPF_BASE_PATH", "dataset_big5")
OUTPUT_DIR = os.getenv(
    "MPF_OUTPUT_DIR",
    "model_pre_encoder_fine_classification_training_big5_fmsynth3_0_1",
)
MODEL_NAME = "model_pre_encoder_fine_classification_training_big5_fmsynth3_0_1"
ENCODER_NAME = f"encoder_{MODEL_NAME}"
AUTOENCODER_NAME = f"autoencoder_{MODEL_NAME}"

RANDOM_STATE = int(os.getenv("MPF_RANDOM_STATE", "0"))
TRAIN_FRAC = float(os.getenv("MPF_TRAIN_FRAC", "0.75"))
VAL_FRAC = float(os.getenv("MPF_VAL_FRAC", "0.2"))
MAX_SAMPLES = int(os.getenv("MPF_MAX_SAMPLES", "0"))

BATCH_SIZE = int(os.getenv("MPF_BATCH_SIZE", "2"))
PRED_BATCH_SIZE = int(os.getenv("MPF_PRED_BATCH_SIZE", "4"))

PRETRAIN_EPOCHS = int(os.getenv("MPF_PRETRAIN_EPOCHS", "80"))
FINETUNE_EPOCHS = int(os.getenv("MPF_FINETUNE_EPOCHS", "120"))
PRETRAIN_LR = float(os.getenv("MPF_PRETRAIN_LR", "1e-3"))
FINETUNE_LR = float(os.getenv("MPF_FINETUNE_LR", "5e-4"))

LATENT_DIM = int(os.getenv("MPF_LATENT_DIM", "256"))
DECODER_HIDDEN_DIM = int(os.getenv("MPF_DECODER_HIDDEN_DIM", "384"))
DECODER_DROPOUT = float(os.getenv("MPF_DECODER_DROPOUT", "0.1"))

N_FFT = int(os.getenv("MPF_N_FFT", "1024"))
HOP_LENGTH = int(os.getenv("MPF_HOP_LENGTH", "256"))
WIN_LENGTH = int(os.getenv("MPF_WIN_LENGTH", str(N_FFT)))
N_MELS = int(os.getenv("MPF_N_MELS", "96"))
MEL_FMIN = float(os.getenv("MPF_MEL_FMIN", "20.0"))

LOG_MEL_LOSS_WEIGHT = float(os.getenv("MPF_LOG_MEL_LOSS_WEIGHT", "0.7"))
STFT_LOSS_WEIGHT = float(os.getenv("MPF_STFT_LOSS_WEIGHT", "0.3"))

CAT_LOSS_WEIGHT_DEFAULT = float(os.getenv("MPF_CAT_LOSS_WEIGHT_DEFAULT", "0.05"))
CAT_LOSS_WEIGHT_ALGORITHM = float(os.getenv("MPF_CAT_LOSS_WEIGHT_ALGORITHM", "1.0"))
CAT_LOSS_WEIGHT_STYLE = float(os.getenv("MPF_CAT_LOSS_WEIGHT_STYLE", "0.20"))
CAT_LOSS_WEIGHT_ENV_CURVE = float(os.getenv("MPF_CAT_LOSS_WEIGHT_ENV_CURVE", "0.06"))

EPS = 1e-7
USE_MIXED_PRECISION = os.getenv("MPF_MIXED_PRECISION", "1") == "1"
DISABLE_XLA_JIT = os.getenv("MPF_DISABLE_XLA_JIT", "1") == "1"
AUDIO_DTYPE = os.getenv("MPF_AUDIO_DTYPE", "float16").strip().lower()
SAVE_LATENT_ARRAYS = os.getenv("MPF_SAVE_LATENT_ARRAYS", "1") == "1"
SAVE_SPLIT_ARRAYS = os.getenv("MPF_SAVE_SPLIT_ARRAYS", "0") == "1"

ALGORITHM_MERGE_MAP = {
    "dual_chain": "series2x2_parallel1",
}

if AUDIO_DTYPE not in {"float16", "float32"}:
    raise ValueError("MPF_AUDIO_DTYPE deve ser 'float16' ou 'float32'.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

if USE_MIXED_PRECISION:
    mixed_precision.set_global_policy("mixed_float16")
if DISABLE_XLA_JIT:
    tf.config.optimizer.set_jit(False)

print(
    "Runtime config: "
    f"batch_size={BATCH_SIZE}, pred_batch_size={PRED_BATCH_SIZE}, "
    f"pretrain_epochs={PRETRAIN_EPOCHS}, finetune_epochs={FINETUNE_EPOCHS}, "
    f"pretrain_lr={PRETRAIN_LR}, finetune_lr={FINETUNE_LR}, "
    f"mixed_precision={USE_MIXED_PRECISION}, policy={mixed_precision.global_policy().name}, "
    f"disable_xla_jit={DISABLE_XLA_JIT}, audio_dtype={AUDIO_DTYPE}, max_samples={MAX_SAMPLES}"
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


def categorical_loss_weight(col_name: str) -> float:
    if col_name == "algorithm":
        return CAT_LOSS_WEIGHT_ALGORITHM
    if col_name == "style":
        return CAT_LOSS_WEIGHT_STYLE
    if col_name.startswith("env") and "curve" in col_name:
        return CAT_LOSS_WEIGHT_ENV_CURVE
    return CAT_LOSS_WEIGHT_DEFAULT


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


class AutoencoderSequence(tf.keras.utils.Sequence):
    def __init__(self, x: np.ndarray, indices: np.ndarray, batch_size: int, shuffle: bool):
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


class ClassificationSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        x: np.ndarray,
        labels: dict[str, np.ndarray],
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ):
        self.x = x
        self.labels = {k: np.asarray(v) for k, v in labels.items()}
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
        y_batch = {head_name: arr[batch_ids] for head_name, arr in self.labels.items()}
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


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


def build_encoder_backbone(input_len: int) -> tuple[tf.keras.layers.Input, tf.Tensor]:
    input_layer = Input(shape=(input_len, 1), name="audio_input")

    x = BatchNormalization(name="enc_input_batch_norm")(input_layer)
    x = Conv1D(32, 11, strides=2, padding="same", activation="gelu", name="enc_conv_1")(x)
    x = MaxPooling1D(pool_size=2, name="enc_pool_1")(x)
    x = Conv1D(64, 9, strides=1, padding="same", activation="gelu", name="enc_conv_2")(x)
    x = MaxPooling1D(pool_size=2, name="enc_pool_2")(x)
    x = Conv1D(96, 7, strides=1, padding="same", activation="gelu", name="enc_conv_3")(x)
    x = MaxPooling1D(pool_size=2, name="enc_pool_3")(x)
    x = Conv1D(
        128,
        5,
        strides=1,
        dilation_rate=2,
        padding="same",
        activation="gelu",
        name="enc_conv_4_dilated",
    )(x)
    x = GlobalAveragePooling1D(name="enc_gap")(x)
    x = Dense(max(LATENT_DIM * 2, 256), activation="gelu", name="enc_dense")(x)
    latent = Dense(LATENT_DIM, activation=None, name="latent")(x)
    return input_layer, latent


def build_autoencoder(input_len: int) -> tuple[Model, Model]:
    input_layer, latent = build_encoder_backbone(input_len)
    d = Dense(DECODER_HIDDEN_DIM, activation="gelu", name="dec_dense_1")(latent)
    d = Dropout(DECODER_DROPOUT, name="dec_dropout")(d)
    d = Dense(input_len, activation="tanh", dtype="float32", name="dec_waveform_flat")(d)
    output = Reshape((input_len, 1), name="reconstruction")(d)
    autoencoder = Model(input_layer, output, name=AUTOENCODER_NAME)
    encoder = Model(input_layer, latent, name=f"encoder_{AUTOENCODER_NAME}")
    return autoencoder, encoder


def build_classifier(input_len: int, categorical_output_dims: dict[str, int]) -> tuple[Model, Model]:
    input_layer, latent = build_encoder_backbone(input_len)

    c = BatchNormalization(name="cls_latent_batch_norm")(latent)
    c = Dense(384, activation="gelu", name="cls_shared_dense_1")(c)
    c = Dropout(0.20, name="cls_shared_dropout_1")(c)
    c = Dense(256, activation="gelu", name="cls_shared_dense_2")(c)
    c = Dropout(0.15, name="cls_shared_dropout_2")(c)
    shared = Dense(128, activation="gelu", name="cls_shared_dense_3")(c)

    outputs = {}
    for col, n_classes in categorical_output_dims.items():
        head_name = cat_output_name(col)
        if col == "algorithm":
            hidden_units = 160
        elif col == "style":
            hidden_units = 96
        else:
            hidden_units = 72
        h = Dense(hidden_units, activation="gelu", name=f"{head_name}_dense")(shared)
        outputs[head_name] = Dense(
            n_classes,
            activation="softmax",
            dtype="float32",
            name=head_name,
        )(h)

    model = Model(input_layer, outputs, name=MODEL_NAME)
    encoder = Model(input_layer, latent, name=ENCODER_NAME)
    return model, encoder


def transfer_encoder_weights(source_encoder: Model, target_encoder: Model) -> int:
    copied = 0
    source_layers = {layer.name: layer for layer in source_encoder.layers}
    for layer in target_encoder.layers:
        if layer.name not in source_layers:
            continue
        src = source_layers[layer.name]
        if not layer.weights or len(layer.get_weights()) != len(src.get_weights()):
            continue
        layer.set_weights(src.get_weights())
        copied += 1
    return copied


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


def compute_autoencoder_test_metrics(
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
# Load data
# -------------------------
meta = {}
meta_path = os.path.join(BASE_PATH, "meta.json")
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

tamanho_dataset = int(meta.get("tamanho_dataset", 5000))
SAMPLE_RATE = int(meta.get("sample_rate_out", 16000))
MEL_FMAX = float(os.getenv("MPF_MEL_FMAX", str(SAMPLE_RATE / 2.0)))

params_path = os.path.join(BASE_PATH, "parameters.csv")
if not os.path.exists(params_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {params_path}")

params_raw = pd.read_csv(params_path)
if "algorithm" in params_raw.columns:
    params_raw["algorithm"] = params_raw["algorithm"].replace(ALGORITHM_MERGE_MAP)
if "id" in params_raw.columns:
    params_raw = params_raw.sort_values("id").reset_index(drop=True)
if tamanho_dataset and tamanho_dataset < len(params_raw):
    params_raw = params_raw.iloc[:tamanho_dataset].copy()

sample_ids = params_raw["id"].astype(int).tolist() if "id" in params_raw.columns else list(range(len(params_raw)))
print(f"Total de amostras: {len(sample_ids)}")

x_raw, sample_rate_found, audio_len = load_audio_dataset(BASE_PATH, sample_ids)
print(f"Shape bruto de áudio: {x_raw.shape}")
print(f"Sample rate encontrado: {sample_rate_found}")
print(f"Audio length: {audio_len}")

if sample_rate_found != SAMPLE_RATE:
    SAMPLE_RATE = sample_rate_found
    MEL_FMAX = float(os.getenv("MPF_MEL_FMAX", str(SAMPLE_RATE / 2.0)))

x_all = x_raw.reshape((x_raw.shape[0], x_raw.shape[1], 1))
n_samples = x_all.shape[0]

categorical_maps = {}
target_encoded = params_raw.copy()
for col in target_encoded.columns:
    if target_encoded[col].dtype == object:
        cat = pd.Categorical(target_encoded[col])
        categorical_maps[col] = [str(x) for x in cat.categories]
        target_encoded[col] = cat.codes.astype(np.int32)
    elif target_encoded[col].dtype == bool:
        target_encoded[col] = target_encoded[col].astype(np.int32)

if "id" in target_encoded.columns:
    target_all = target_encoded.drop(columns=["id"]).copy()
else:
    target_all = target_encoded.copy()

constant_targets = {}
for col in list(target_all.columns):
    if target_all[col].nunique(dropna=False) <= 1:
        constant_targets[col] = to_json_scalar(target_all[col].iloc[0])

target_model = target_all.drop(columns=list(constant_targets.keys())).copy()
if target_model.empty:
    raise ValueError("Todos os targets foram removidos como constantes.")

categorical_cols = [col for col in target_model.columns if col in categorical_maps]
if not categorical_cols:
    raise ValueError("Nenhuma coluna categórica encontrada para classificação.")

cat_dims = {col: len(categorical_maps[col]) for col in categorical_cols}

all_idx = np.arange(n_samples)
rng = np.random.default_rng(RANDOM_STATE)
rng.shuffle(all_idx)
train_size = int(TRAIN_FRAC * n_samples)
train_idx = all_idx[:train_size]
test_idx = all_idx[train_size:]
val_size = int(VAL_FRAC * len(train_idx))
val_idx = train_idx[:val_size]
fit_idx = train_idx[val_size:]
print(f"Split: fit={len(fit_idx)} val={len(val_idx)} test={len(test_idx)}")

if SAVE_SPLIT_ARRAYS:
    np.save(os.path.join(OUTPUT_DIR, "x_train_big5.npy"), x_all[train_idx])
    np.save(os.path.join(OUTPUT_DIR, "x_test_big5.npy"), x_all[test_idx])

auto_train_seq = AutoencoderSequence(x_all, fit_idx, BATCH_SIZE, shuffle=True)
auto_val_seq = AutoencoderSequence(x_all, val_idx, BATCH_SIZE, shuffle=False)

y_all = {cat_output_name(col): target_model[col].to_numpy(dtype=np.int32) for col in categorical_cols}
class_train_seq = ClassificationSequence(x_all, y_all, fit_idx, BATCH_SIZE, shuffle=True)
class_val_seq = ClassificationSequence(x_all, y_all, val_idx, BATCH_SIZE, shuffle=False)
class_test_seq = ClassificationSequence(x_all, y_all, test_idx, PRED_BATCH_SIZE, shuffle=False)

# -------------------------
# Stage 1: pretrain autoencoder
# -------------------------
mel_matrix = build_mel_matrix(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    n_mels=N_MELS,
    mel_fmin=MEL_FMIN,
    mel_fmax=MEL_FMAX,
)
hybrid_loss, log_mel_mae_metric, stft_mae_metric = build_hybrid_loss_and_metrics(mel_matrix)

autoencoder, pretrained_encoder = build_autoencoder(audio_len)
autoencoder.compile(
    optimizer=Nadam(learning_rate=PRETRAIN_LR),
    loss=hybrid_loss,
    metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), log_mel_mae_metric, stft_mae_metric],
)

plot_model(
    autoencoder,
    to_file=os.path.join(OUTPUT_DIR, "autoencoder_model.png"),
    show_shapes=True,
    expand_nested=True,
    rankdir="TB",
    dpi=220,
)

auto_callbacks = [
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
]

auto_history = autoencoder.fit(
    auto_train_seq,
    epochs=PRETRAIN_EPOCHS,
    validation_data=auto_val_seq,
    callbacks=auto_callbacks,
    verbose=1,
)

auto_hist = pd.DataFrame(auto_history.history)
auto_hist["epoch"] = auto_history.epoch
auto_hist.to_csv(os.path.join(OUTPUT_DIR, "pretrain_history.csv"), index=False)

plot_metric(auto_hist, "loss", "val_loss", "Autoencoder Loss", "pretrain_loss.png")

pretrain_test_log_mel_mae, pretrain_test_stft_mae, pretrain_test_hybrid = compute_autoencoder_test_metrics(
    autoencoder,
    x_all,
    test_idx,
)

# -------------------------
# Stage 2: fine-tune classification
# -------------------------
classifier, classifier_encoder = build_classifier(audio_len, cat_dims)
copied_layers = transfer_encoder_weights(pretrained_encoder, classifier_encoder)
print(f"Pesos do encoder copiados: {copied_layers} camadas.")

losses = {}
metrics = {}
loss_weights = {}
for col in categorical_cols:
    head_name = cat_output_name(col)
    losses[head_name] = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics[head_name] = ["sparse_categorical_accuracy"]
    loss_weights[head_name] = categorical_loss_weight(col)

classifier.compile(
    optimizer=Nadam(learning_rate=FINETUNE_LR),
    loss=losses,
    metrics=metrics,
    loss_weights=loss_weights,
)

plot_model(
    classifier,
    to_file=os.path.join(OUTPUT_DIR, "classifier_model.png"),
    show_shapes=True,
    expand_nested=True,
    rankdir="TB",
    dpi=220,
)
plot_model(
    classifier_encoder,
    to_file=os.path.join(OUTPUT_DIR, "encoder_model.png"),
    show_shapes=True,
    expand_nested=True,
    rankdir="TB",
    dpi=220,
)

class_callbacks = [
    EarlyStopping(monitor="val_loss", patience=24, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
]

class_history = classifier.fit(
    class_train_seq,
    epochs=FINETUNE_EPOCHS,
    validation_data=class_val_seq,
    callbacks=class_callbacks,
    verbose=1,
)

hist = pd.DataFrame(class_history.history)
hist["epoch"] = class_history.epoch

train_cat_acc_cols = [c for c in hist.columns if c.startswith("cat__") and c.endswith("sparse_categorical_accuracy")]
val_cat_acc_cols = [c for c in hist.columns if c.startswith("val_cat__") and c.endswith("sparse_categorical_accuracy")]
train_cat_loss_cols = [c for c in hist.columns if c.startswith("cat__") and c.endswith("_loss")]
val_cat_loss_cols = [c for c in hist.columns if c.startswith("val_cat__") and c.endswith("_loss")]

if train_cat_acc_cols:
    hist["cat_acc_mean"] = hist[train_cat_acc_cols].mean(axis=1)
if val_cat_acc_cols:
    hist["val_cat_acc_mean"] = hist[val_cat_acc_cols].mean(axis=1)
if train_cat_loss_cols:
    hist["cat_loss_mean"] = hist[train_cat_loss_cols].mean(axis=1)
if val_cat_loss_cols:
    hist["val_cat_loss_mean"] = hist[val_cat_loss_cols].mean(axis=1)

hist.to_csv(os.path.join(OUTPUT_DIR, "train_history.csv"), index=False)
plot_metric(hist, "loss", "val_loss", "Classification Loss", "class_loss.png")
plot_metric(hist, "cat_acc_mean", "val_cat_acc_mean", "Categorical Accuracy (mean)", "class_acc_mean.png")
plot_metric(hist, "cat_loss_mean", "val_cat_loss_mean", "Categorical Loss (mean)", "class_loss_mean.png")

algo_acc_key = f"{cat_output_name('algorithm')}_sparse_categorical_accuracy"
val_algo_acc_key = f"val_{algo_acc_key}"
plot_metric(hist, algo_acc_key, val_algo_acc_key, "Algorithm Accuracy", "class_algorithm_acc.png")

# -------------------------
# Evaluation
# -------------------------
test_eval = classifier.evaluate(class_test_seq, return_dict=True, verbose=0)
raw_pred = classifier.predict(class_test_seq, batch_size=PRED_BATCH_SIZE, verbose=0)
if isinstance(raw_pred, dict):
    pred_map = raw_pred
elif isinstance(raw_pred, list):
    pred_map = {name: pred for name, pred in zip(classifier.output_names, raw_pred, strict=False)}
else:
    pred_map = {classifier.output_names[0]: raw_pred}

categorical_accuracy = {}
categorical_crossentropy = {}
per_head = {}
pred_codes_df = pd.DataFrame(index=np.arange(len(test_idx)))
if "id" in params_raw.columns:
    pred_codes_df["id"] = params_raw.iloc[test_idx]["id"].to_numpy(dtype=np.int32)

y_test = {cat_output_name(col): target_model[col].to_numpy(dtype=np.int32)[test_idx] for col in categorical_cols}

for col in categorical_cols:
    head_name = cat_output_name(col)
    y_true = y_test[head_name].astype(np.int32)
    y_prob = np.asarray(pred_map[head_name], dtype=np.float64)
    y_pred = np.argmax(y_prob, axis=1).astype(np.int32)
    acc = float((y_pred == y_true).mean())
    ce = float(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_prob).numpy().mean())
    categorical_accuracy[col] = acc
    categorical_crossentropy[col] = ce
    per_head[col] = {"accuracy": acc, "crossentropy": ce, "n_classes": int(cat_dims[col])}
    pred_codes_df[f"true__{col}"] = y_true
    pred_codes_df[f"pred__{col}"] = y_pred

pred_labels_df = pred_codes_df.copy()
for col in categorical_cols:
    categories = categorical_maps[col]
    pred_labels_df[f"true_label__{col}"] = pred_codes_df[f"true__{col}"].map(
        lambda x: categories[int(x)] if 0 <= int(x) < len(categories) else "<UNK>"
    )
    pred_labels_df[f"pred_label__{col}"] = pred_codes_df[f"pred__{col}"].map(
        lambda x: categories[int(x)] if 0 <= int(x) < len(categories) else "<UNK>"
    )

categorical_accuracy_mean = float(np.mean(list(categorical_accuracy.values())))
categorical_crossentropy_mean = float(np.mean(list(categorical_crossentropy.values())))

print("===== Fine-tuned Classification Test Metrics =====")
print(f"categorical_accuracy_mean: {categorical_accuracy_mean:.6f}")
print(f"categorical_crossentropy_mean: {categorical_crossentropy_mean:.6f}")
print(f"pretrain_test_hybrid: {pretrain_test_hybrid:.6f}")

# -------------------------
# Save artifacts
# -------------------------
autoencoder.save(os.path.join(OUTPUT_DIR, f"{AUTOENCODER_NAME}.keras"))
pretrained_encoder.save(os.path.join(OUTPUT_DIR, f"pretrained_{ENCODER_NAME}.keras"))
classifier.save(os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.keras"))
classifier_encoder.save(os.path.join(OUTPUT_DIR, f"{ENCODER_NAME}.keras"))

if SAVE_LATENT_ARRAYS:
    z_train = encode_by_indices(classifier_encoder, x_all, train_idx, PRED_BATCH_SIZE)
    z_test = encode_by_indices(classifier_encoder, x_all, test_idx, PRED_BATCH_SIZE)
    np.save(os.path.join(OUTPUT_DIR, "latent_train.npy"), z_train)
    np.save(os.path.join(OUTPUT_DIR, "latent_test.npy"), z_test)

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
    "categorical_cols": categorical_cols,
    "categorical_maps": {k: categorical_maps[k] for k in categorical_cols},
    "constant_targets": constant_targets,
    "algorithm_merge_map": ALGORITHM_MERGE_MAP,
}

joblib.dump(preprocess_bundle, os.path.join(OUTPUT_DIR, f"preprocess_{MODEL_NAME}.save"))
joblib.dump(
    {
        "categorical_cols": categorical_cols,
        "categorical_maps": {k: categorical_maps[k] for k in categorical_cols},
        "constant_targets": constant_targets,
        "algorithm_merge_map": ALGORITHM_MERGE_MAP,
    },
    os.path.join(OUTPUT_DIR, f"target_preprocess_{MODEL_NAME}.save"),
)

pred_codes_df.to_csv(os.path.join(OUTPUT_DIR, "params_pred_test_codes.csv"), index=False)
pred_labels_df.to_csv(os.path.join(OUTPUT_DIR, "params_pred_test_labels.csv"), index=False)
auto_hist.to_csv(os.path.join(OUTPUT_DIR, "pretrain_history.csv"), index=False)
hist.to_csv(os.path.join(OUTPUT_DIR, "train_history.csv"), index=False)

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

pretrain_history_last = {}
for col in auto_hist.columns:
    if col == "epoch":
        continue
    try:
        pretrain_history_last[col] = float(auto_hist[col].iloc[-1])
    except (TypeError, ValueError):
        pass

pretrain_best_val_loss_epoch = None
pretrain_best_val_loss = None
if "val_loss" in auto_hist:
    best_idx = int(auto_hist["val_loss"].idxmin())
    pretrain_best_val_loss_epoch = int(auto_hist.loc[best_idx, "epoch"])
    pretrain_best_val_loss = float(auto_hist.loc[best_idx, "val_loss"])

results = {
    "model_name": MODEL_NAME,
    "autoencoder_name": AUTOENCODER_NAME,
    "encoder_name": ENCODER_NAME,
    "dataset": BASE_PATH,
    "n_samples": int(n_samples),
    "split": {
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "train_frac": float(TRAIN_FRAC),
        "val_frac_of_train": float(VAL_FRAC),
    },
    "runtime_config": {
        "batch_size": int(BATCH_SIZE),
        "pred_batch_size": int(PRED_BATCH_SIZE),
        "pretrain_epochs": int(PRETRAIN_EPOCHS),
        "finetune_epochs": int(FINETUNE_EPOCHS),
        "pretrain_lr": float(PRETRAIN_LR),
        "finetune_lr": float(FINETUNE_LR),
        "mixed_precision": bool(USE_MIXED_PRECISION),
        "mixed_precision_policy": mixed_precision.global_policy().name,
        "disable_xla_jit": bool(DISABLE_XLA_JIT),
        "audio_dtype": AUDIO_DTYPE,
        "save_latent_arrays": bool(SAVE_LATENT_ARRAYS),
        "save_split_arrays": bool(SAVE_SPLIT_ARRAYS),
    },
    "algorithm_label_merges": ALGORITHM_MERGE_MAP,
    "removed_constant_targets": constant_targets,
    "categorical_cols": categorical_cols,
    "loss_weights": {
        "default": float(CAT_LOSS_WEIGHT_DEFAULT),
        "algorithm": float(CAT_LOSS_WEIGHT_ALGORITHM),
        "style": float(CAT_LOSS_WEIGHT_STYLE),
        "env_curve": float(CAT_LOSS_WEIGHT_ENV_CURVE),
        "by_col": {col: float(categorical_loss_weight(col)) for col in categorical_cols},
    },
    "pretraining": {
        "metrics": {
            "test_log_mel_mae": float(pretrain_test_log_mel_mae),
            "test_stft_mae": float(pretrain_test_stft_mae),
            "test_hybrid_recomputed": float(pretrain_test_hybrid),
        },
        "history_last": pretrain_history_last,
        "best_val_loss": {
            "epoch": pretrain_best_val_loss_epoch,
            "value": pretrain_best_val_loss,
        },
    },
    "fine_tuning": {
        "metrics": {
            "categorical_accuracy_mean": categorical_accuracy_mean,
            "categorical_crossentropy_mean": categorical_crossentropy_mean,
        },
        "categorical_accuracy": categorical_accuracy,
        "categorical_crossentropy": categorical_crossentropy,
        "test_metrics_by_head": per_head,
        "history_last": history_last,
        "best_val_loss": {
            "epoch": best_val_loss_epoch,
            "value": best_val_loss,
        },
    },
    "categorical_maps": {k: categorical_maps[k] for k in categorical_cols},
    "artifacts": {
        "autoencoder": os.path.join(OUTPUT_DIR, f"{AUTOENCODER_NAME}.keras"),
        "pretrained_encoder": os.path.join(OUTPUT_DIR, f"pretrained_{ENCODER_NAME}.keras"),
        "classifier": os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.keras"),
        "encoder": os.path.join(OUTPUT_DIR, f"{ENCODER_NAME}.keras"),
        "preprocess": os.path.join(OUTPUT_DIR, f"preprocess_{MODEL_NAME}.save"),
        "target_preprocess": os.path.join(OUTPUT_DIR, f"target_preprocess_{MODEL_NAME}.save"),
        "results": os.path.join(OUTPUT_DIR, "results.json"),
        "pretrain_history": os.path.join(OUTPUT_DIR, "pretrain_history.csv"),
        "train_history": os.path.join(OUTPUT_DIR, "train_history.csv"),
        "pred_codes": os.path.join(OUTPUT_DIR, "params_pred_test_codes.csv"),
        "pred_labels": os.path.join(OUTPUT_DIR, "params_pred_test_labels.csv"),
        "latent_train": os.path.join(OUTPUT_DIR, "latent_train.npy") if SAVE_LATENT_ARRAYS else None,
        "latent_test": os.path.join(OUTPUT_DIR, "latent_test.npy") if SAVE_LATENT_ARRAYS else None,
    },
}

with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Treino em duas etapas concluído.")
print(f"Resultados em: {os.path.join(OUTPUT_DIR, 'results.json')}")
