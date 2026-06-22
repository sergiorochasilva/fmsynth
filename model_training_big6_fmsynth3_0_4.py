"""Light multi-head waveform regressor for `dataset_big6` (variant 0_4).

Architecture:
- Raw waveform input loaded from sharded `int16` audio or legacy caches
- Compact 1D residual CNN backbone with early downsampling and multi-scale pooled features
- Shared dense trunk plus specialized heads for grouped numeric targets and categorical targets

Data flow:
- Input: `parameters.csv` plus `audio_big6_manifest.json` shards or `sample_*.wav`
- Output: trained model `.keras`, target preprocessing bundle, predictions, plots, and `results.json`
"""

import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import soundfile as sf

# Use a non-interactive backend for matplotlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
    LayerNormalization,
    MaxPooling1D,
    SpatialDropout1D,
)
from keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import plot_model

import joblib

BASE_PATH = os.getenv("DATASET_PATH", "dataset_big6")
AUDIO_MANIFEST_PATH = os.path.join(BASE_PATH, "audio_big6_manifest.json")
AUDIO_LEGACY_CACHE_PATH = os.path.join(BASE_PATH, "audio_big6_int16.npy")
OUTPUT_DIR = "model_training_big6_fmsynth3_0_4"
MODEL_NAME = "model_training_big6_fmsynth3_0_4"

RANDOM_STATE = 0
TRAIN_FRAC = 0.75
VAL_FRAC = 0.2

# Per-head weights (objetivo: priorizar parâmetros críticos de timbre).
RATIO_HEAD_LOSS_WEIGHT = float(os.getenv("RATIO_HEAD_LOSS_WEIGHT", "2.4"))
INDEX_HEAD_LOSS_WEIGHT = float(os.getenv("INDEX_HEAD_LOSS_WEIGHT", "2.0"))
DETUNE_HEAD_LOSS_WEIGHT = float(os.getenv("DETUNE_HEAD_LOSS_WEIGHT", "1.5"))
ENV_HEAD_LOSS_WEIGHT = float(os.getenv("ENV_HEAD_LOSS_WEIGHT", "0.9"))
PHASE_HEAD_LOSS_WEIGHT = float(os.getenv("PHASE_HEAD_LOSS_WEIGHT", "0.2"))
OTHER_HEAD_LOSS_WEIGHT = float(os.getenv("OTHER_HEAD_LOSS_WEIGHT", "0.8"))
FREQ_HEAD_LOSS_WEIGHT = float(os.getenv("FREQ_HEAD_LOSS_WEIGHT", "2.4"))

CAT_LOSS_WEIGHT_DEFAULT = float(os.getenv("CAT_LOSS_WEIGHT_DEFAULT", "0.05"))
CAT_LOSS_WEIGHT_ALGORITHM = float(os.getenv("CAT_LOSS_WEIGHT_ALGORITHM", "1.25"))
CAT_LOSS_WEIGHT_STYLE = float(os.getenv("CAT_LOSS_WEIGHT_STYLE", "0.2"))
CAT_LOSS_WEIGHT_ENV_CURVE = float(os.getenv("CAT_LOSS_WEIGHT_ENV_CURVE", "0.06"))

# Configuração de lote
# Batch pequeno para caber em 4 GB de VRAM com o dataset completo.
BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "2"))
PRED_BATCH_SIZE = int(os.getenv("PRED_BATCH_SIZE", "4"))

# Configuração para reduzir uso de VRAM
USE_MIXED_PRECISION = os.getenv("MIXED_PRECISION", "1") == "1"
DISABLE_XLA_JIT = os.getenv("DISABLE_XLA_JIT", "1") == "1"
AUDIO_DTYPE = os.getenv("AUDIO_DTYPE", "float16").strip().lower()
if AUDIO_DTYPE not in {"float16", "float32"}:
    raise ValueError("AUDIO_DTYPE deve ser 'float16' ou 'float32'.")

LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.00015"))
CLIPNORM = float(os.getenv("CLIPNORM", "1.0"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.00001"))
STYLE_RESAMPLE = os.getenv("STYLE_RESAMPLE", "1") == "1"
CAT_LABEL_SMOOTHING = float(os.getenv("CAT_LABEL_SMOOTHING", "0.02"))

ALGORITHM_MERGE_MAP = {
    # Em fm_synth3, dual_chain é implementado como alias de series2x2_parallel1.
    "dual_chain": "series2x2_parallel1",
}

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
    f"learning_rate={LEARNING_RATE}, clipnorm={CLIPNORM}, dataset_path={BASE_PATH}"
)


# -------------------------
# Utilitários
# -------------------------
def cat_output_name(column_name: str) -> str:
    return f"cat__{column_name}"


def reg_output_name(group_name: str) -> str:
    return f"{group_name}_head"


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


def normalize_audio_batch(batch: np.ndarray) -> np.ndarray:
    batch = np.asarray(batch, dtype=np.float32) / 32768.0
    if AUDIO_DTYPE == "float16":
        return batch.astype(np.float16, copy=False)
    return batch.astype(np.float32, copy=False)


def load_manifest() -> dict | None:
    if not os.path.exists(AUDIO_MANIFEST_PATH):
        return None
    try:
        with open(AUDIO_MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class DenseAudioStore:
    def __init__(self, audio: np.ndarray):
        audio = np.asarray(audio)
        if audio.ndim != 2:
            raise ValueError(f"Formato inesperado de áudio denso: {audio.shape}")
        self.audio = audio.astype(np.int16, copy=False)
        self.sample_len = int(self.audio.shape[1])

    def get_batch(self, sample_indices: np.ndarray) -> np.ndarray:
        sample_indices = np.asarray(sample_indices, dtype=np.int32)
        return normalize_audio_batch(self.audio[sample_indices]).reshape(
            sample_indices.shape[0], self.sample_len, 1
        )


class ShardedAudioStore:
    def __init__(self, base_path: str, manifest: dict):
        self.base_path = base_path
        self.manifest = manifest
        self.sample_len = int(manifest["audio_sample_len"])
        self.shard_size = int(manifest["audio_shard_size"])
        self.total_rows = int(manifest["total_rows"])
        self.shards = list(manifest.get("shards", []))
        self.max_cached_shards = int(os.getenv("AUDIO_SHARD_CACHE", "4"))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def _load_shard(self, shard_idx: int) -> np.ndarray:
        shard_idx = int(shard_idx)
        if shard_idx in self._cache:
            self._cache.move_to_end(shard_idx)
            return self._cache[shard_idx]

        shard_meta = self.shards[shard_idx]
        shard_path = os.path.join(self.base_path, shard_meta["file"])
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


def load_audio_store(base_path: str, sample_ids: list[int]):
    manifest = load_manifest()
    if manifest is not None:
        return ShardedAudioStore(base_path, manifest)

    if os.path.exists(AUDIO_LEGACY_CACHE_PATH):
        audio = np.load(AUDIO_LEGACY_CACHE_PATH, mmap_mode="r")
        if audio.ndim != 2:
            raise ValueError(f"Formato inesperado no cache legado: {audio.shape}")
        return DenseAudioStore(audio)

    print(
        "Aviso: cache shardado não encontrado. "
        "Fallback para leitura dos WAVs individuais."
    )
    samples = []
    for sample_id in sample_ids:
        wav_path = os.path.join(base_path, f"sample_{sample_id}.wav")
        signal, _sample_rate = sf.read(wav_path)
        samples.append(np.round(np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16))
    return DenseAudioStore(np.asarray(samples, dtype=np.int16))


class MultiHeadAudioSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        audio_store,
        sample_indices: np.ndarray,
        y: dict[str, np.ndarray] | None,
        batch_size: int,
        shuffle: bool = True,
        sample_weights: np.ndarray | None = None,
    ):
        self.audio_store = audio_store
        self.sample_indices = np.asarray(sample_indices, dtype=np.int32)
        self.y = y
        self.batch_size = int(max(batch_size, 1))
        self.shuffle = bool(shuffle)
        self.sample_weights = None
        if sample_weights is not None:
            weights = np.asarray(sample_weights, dtype=np.float64)
            if weights.shape[0] != self.sample_indices.shape[0]:
                raise ValueError("sample_weights deve ter o mesmo tamanho de sample_indices")
            weights = np.clip(weights, 1e-12, None)
            self.sample_weights = weights / weights.sum()
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
        if self.y is None:
            return x_batch
        y_batch = {name: values[batch_ids] for name, values in self.y.items()}
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.sample_weights is not None:
            self.indices = np.random.choice(
                np.arange(self.sample_indices.shape[0], dtype=np.int32),
                size=self.sample_indices.shape[0],
                replace=True,
                p=self.sample_weights,
            ).astype(np.int32)
        elif self.shuffle:
            np.random.shuffle(self.indices)


def build_numeric_groups(numeric_cols: list[str]) -> dict[str, list[str]]:
    ratio_base = ["ratio1", "ratio2", "ratio3", "ratio4", "ratio5", "ratio_carrier"]
    index_base = ["index_12", "index_23", "index_3c", "index_4c", "index_5c"]
    detune_base = ["detune1", "detune2", "detune3", "detune4", "detune5", "detune_carrier"]

    ratio_cols = [c for c in ratio_base if c in numeric_cols]
    index_cols = [c for c in index_base if c in numeric_cols]
    detune_cols = [c for c in detune_base if c in numeric_cols]

    used = set(ratio_cols + index_cols + detune_cols)

    env_cols = [c for c in numeric_cols if c.startswith("env") and c not in used]
    used.update(env_cols)

    phase_cols = [c for c in numeric_cols if c.startswith("phase") and c not in used]
    used.update(phase_cols)

    other_cols = [c for c in numeric_cols if c not in used]

    return {
        "ratio": ratio_cols,
        "index": index_cols,
        "detune": detune_cols,
        "env": env_cols,
        "phase": phase_cols,
        "other": other_cols,
    }


def categorical_loss_weight(col_name: str) -> float:
    if col_name == "algorithm":
        return CAT_LOSS_WEIGHT_ALGORITHM
    if col_name == "style":
        return CAT_LOSS_WEIGHT_STYLE
    if col_name.startswith("env") and "curve" in col_name:
        return CAT_LOSS_WEIGHT_ENV_CURVE
    return CAT_LOSS_WEIGHT_DEFAULT


def numeric_head_loss_weight(head_name: str) -> float:
    if head_name == "ratio_head":
        return RATIO_HEAD_LOSS_WEIGHT
    if head_name == "index_head":
        return INDEX_HEAD_LOSS_WEIGHT
    if head_name == "detune_head":
        return DETUNE_HEAD_LOSS_WEIGHT
    if head_name == "env_head":
        return ENV_HEAD_LOSS_WEIGHT
    if head_name == "phase_head":
        return PHASE_HEAD_LOSS_WEIGHT
    if head_name == "other_head":
        return OTHER_HEAD_LOSS_WEIGHT
    return 1.0


def sparse_categorical_crossentropy_smoothed(label_smoothing: float):
    label_smoothing = float(max(label_smoothing, 0.0))

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        num_classes = tf.shape(y_pred)[-1]
        y_true_oh = tf.one_hot(y_true, depth=num_classes, dtype=y_pred.dtype)
        if label_smoothing > 0.0:
            smooth = tf.cast(label_smoothing, y_pred.dtype)
            num_classes_f = tf.cast(num_classes, y_pred.dtype)
            y_true_oh = y_true_oh * (1.0 - smooth) + smooth / num_classes_f
        return tf.keras.losses.categorical_crossentropy(y_true_oh, y_pred)

    return loss_fn


def make_stratify_key(frame: pd.DataFrame) -> pd.Series | None:
    if "style" not in frame.columns or "algorithm" not in frame.columns:
        return None
    return frame["style"].astype(str) + "__" + frame["algorithm"].astype(str)


def stratified_split_indices(
    indices: np.ndarray,
    strata: pd.Series | None,
    test_size: float,
    random_state: int,
):
    if strata is None:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(indices)
        cut = int(round((1.0 - test_size) * len(perm)))
        return perm[:cut], perm[cut:]

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_pos, test_pos = next(splitter.split(np.zeros(len(indices)), strata))
    return indices[train_pos], indices[test_pos]


# -------------------------
# Leitura de metadados e parâmetros
# -------------------------
meta = {}
meta_path = os.path.join(BASE_PATH, "meta.json")
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

params_path = os.path.join(BASE_PATH, "parameters.csv")
if not os.path.exists(params_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {params_path}")

# Lendo parâmetros (targets)
target_raw = pd.read_csv(params_path)
meta_tamanho_dataset = meta.get("tamanho_dataset")
if meta_tamanho_dataset is None:
    tamanho_dataset = len(target_raw)
else:
    tamanho_dataset = int(meta_tamanho_dataset)
    if tamanho_dataset <= 0:
        tamanho_dataset = len(target_raw)

if tamanho_dataset and tamanho_dataset < len(target_raw):
    target_raw = target_raw.iloc[:tamanho_dataset].copy()
else:
    tamanho_dataset = len(target_raw)

# Ajuste: merge de rótulos equivalentes de algoritmo.
if "algorithm" in target_raw.columns:
    target_raw["algorithm"] = target_raw["algorithm"].replace(ALGORITHM_MERGE_MAP)

# Guardando ids para carregar os .wav corretos
if "id" in target_raw.columns:
    sample_ids = target_raw["id"].astype(int).tolist()
else:
    sample_ids = list(range(tamanho_dataset))

# Codificação categórica e casting de bool
categorical_maps = {}
target_encoded = target_raw.copy()
for col in target_encoded.columns:
    if target_encoded[col].dtype == object:
        cat = pd.Categorical(target_encoded[col])
        categorical_maps[col] = [str(x) for x in cat.categories]
        target_encoded[col] = cat.codes.astype(np.int32)
    elif target_encoded[col].dtype == bool:
        target_encoded[col] = target_encoded[col].astype(np.int32)

# Removendo coluna de id
if "id" in target_encoded.columns:
    target_all = target_encoded.drop(columns=["id"]).copy()
else:
    target_all = target_encoded.copy()

# Removendo targets constantes
constant_targets = {}
for col in list(target_all.columns):
    if target_all[col].nunique(dropna=False) <= 1:
        constant_targets[col] = to_json_scalar(target_all[col].iloc[0])

target_model = target_all.drop(columns=list(constant_targets.keys())).copy()
if target_model.empty:
    raise ValueError("Todos os targets foram removidos como constantes.")

categorical_cols = [col for col in target_model.columns if col in categorical_maps]
numeric_cols = [col for col in target_model.columns if col not in categorical_cols]

# frequencia_base em head dedicada com transformação log2
freq_col = None
if "frequencia_base" in numeric_cols:
    freq_col = "frequencia_base"
    numeric_cols.remove(freq_col)

numeric_groups = build_numeric_groups(numeric_cols)


# -------------------------
# Lendo o dataset de áudio
# -------------------------
audio_store = load_audio_store(BASE_PATH, sample_ids)
audio_len = int(audio_store.sample_len)
expected_audio_len = int(meta.get("audio_sample_len", audio_len))
if expected_audio_len and audio_len != expected_audio_len:
    raise ValueError(
        f"Tamanho de áudio inesperado: {audio_len} vs {expected_audio_len}"
    )


# -------------------------
# Split treino / teste
# -------------------------
dataset_indices = target_all.index.to_numpy(dtype=np.int32)
train_strata = make_stratify_key(target_all)
train_idx, test_idx = stratified_split_indices(
    dataset_indices,
    train_strata,
    test_size=1.0 - TRAIN_FRAC,
    random_state=RANDOM_STATE,
)

train_audio_indices = np.asarray(train_idx, dtype=np.int32)
test_audio_indices = np.asarray(test_idx, dtype=np.int32)

y_train_full = target_all.loc[train_idx].reset_index(drop=True)
y_test_full = target_all.loc[test_idx].reset_index(drop=True)

y_train_model = target_model.loc[train_idx].reset_index(drop=True)
y_test_model = target_model.loc[test_idx].reset_index(drop=True)

# Split treino / validação com estratificação sobre estilo + algoritmo.
val_strata = make_stratify_key(y_train_model)
fit_idx, val_idx = stratified_split_indices(
    np.arange(len(y_train_model), dtype=np.int32),
    val_strata,
    test_size=VAL_FRAC,
    random_state=RANDOM_STATE,
)

fit_audio_indices = train_audio_indices[fit_idx]
val_audio_indices = train_audio_indices[val_idx]

# Salvando índices e targets sem duplicar o áudio inteiro.
np.save(os.path.join(OUTPUT_DIR, "train_audio_indices.npy"), train_audio_indices)
np.save(os.path.join(OUTPUT_DIR, "val_audio_indices.npy"), val_audio_indices)
np.save(os.path.join(OUTPUT_DIR, "test_audio_indices.npy"), test_audio_indices)
np.save(
    os.path.join(OUTPUT_DIR, "y_train_big6.npy"),
    y_train_full.to_numpy(dtype=np.float32),
)
np.save(
    os.path.join(OUTPUT_DIR, "y_test_big6.npy"),
    y_test_full.to_numpy(dtype=np.float32),
)


# -------------------------
# Preparando y por head
# -------------------------
y_fit: dict[str, np.ndarray] = {}
y_val: dict[str, np.ndarray] = {}

numeric_head_specs: dict[str, dict] = {}
scaler_by_numeric_head: dict[str, StandardScaler] = {}

for group_name, cols in numeric_groups.items():
    if not cols:
        continue

    head_name = reg_output_name(group_name)
    values = y_train_model[cols].to_numpy(dtype=np.float32)

    log2_cols: list[str] = []
    if group_name == "ratio":
        if np.any(values <= 0):
            raise ValueError("Há valores de ratio <= 0; log2 não é aplicável.")
        values = np.log2(values)
        log2_cols = list(cols)

    scaler = StandardScaler()
    values_norm = scaler.fit_transform(values).astype(np.float32)

    y_fit[head_name] = values_norm[fit_idx]
    y_val[head_name] = values_norm[val_idx]

    scaler_by_numeric_head[head_name] = scaler
    numeric_head_specs[head_name] = {
        "group_name": group_name,
        "cols": list(cols),
        "log2_cols": log2_cols,
    }

scaler_freq = None
if freq_col is not None:
    y_freq = y_train_model[freq_col].to_numpy(dtype=np.float32)
    if np.any(y_freq <= 0):
        raise ValueError("frequencia_base contém valores <= 0; log2 não é aplicável.")

    y_freq_log2 = np.log2(y_freq).reshape(-1, 1)
    scaler_freq = StandardScaler()
    y_freq_norm = scaler_freq.fit_transform(y_freq_log2).astype(np.float32)
    y_fit["freq_head"] = y_freq_norm[fit_idx]
    y_val["freq_head"] = y_freq_norm[val_idx]

cat_dims = {}
for col in categorical_cols:
    head_name = cat_output_name(col)
    y_col = y_train_model[col].to_numpy(dtype=np.int32)
    y_fit[head_name] = y_col[fit_idx]
    y_val[head_name] = y_col[val_idx]
    cat_dims[col] = len(categorical_maps[col])

train_style_weights = None
if STYLE_RESAMPLE and "style" in y_train_model.columns:
    style_values = y_train_model.loc[fit_idx, "style"].to_numpy(dtype=np.int32)
    style_counts = np.bincount(style_values)
    style_weights = 1.0 / np.maximum(style_counts, 1)
    train_style_weights = style_weights[style_values]


# -------------------------
# Modelo
# -------------------------
# Configurando para não alocar diretamente toda a memória da GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def residual_conv_block(
    x,
    *,
    filters: int,
    kernel_size: int,
    activation: str,
    bias_cnn: bool,
    kernel_regularizer_cnn,
    block_name: str,
    dilation_rate: int = 1,
    dropout_rate: float = 0.0,
):
    shortcut = x
    input_filters = shortcut.shape[-1]
    if input_filters != filters:
        shortcut = Conv1D(
            filters,
            kernel_size=1,
            padding="same",
            use_bias=bias_cnn,
            name=f"{block_name}_proj",
        )(shortcut)

    y = Conv1D(
        filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        activation=None,
        kernel_regularizer=kernel_regularizer_cnn,
        use_bias=bias_cnn,
        name=f"{block_name}_conv1",
    )(x)
    y = BatchNormalization(name=f"{block_name}_bn1")(y)
    y = Activation(activation, name=f"{block_name}_act1")(y)
    if dropout_rate > 0:
        y = SpatialDropout1D(dropout_rate, name=f"{block_name}_drop1")(y)
    y = Conv1D(
        filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        activation=None,
        kernel_regularizer=kernel_regularizer_cnn,
        use_bias=bias_cnn,
        name=f"{block_name}_conv2",
    )(y)
    y = BatchNormalization(name=f"{block_name}_bn2")(y)
    y = Add(name=f"{block_name}_add")([shortcut, y])
    y = Activation(activation, name=f"{block_name}_out")(y)
    return y


def build_model(
    input_len,
    input_dims,
    activation,
    bias_cnn,
    kernel_regularizer_cnn,
    bias_head,
    dropout_head,
    numeric_output_dims,
    has_freq_head,
    categorical_output_dims,
):
    input_layer = Input(shape=(input_len, input_dims), name="audio_input")

    def pooled_1d(tensor, prefix):
        gap = GlobalAveragePooling1D(name=f"{prefix}_gap")(tensor)
        gmp = GlobalMaxPooling1D(name=f"{prefix}_gmp")(tensor)
        return Concatenate(name=f"{prefix}_pool_concat")([gap, gmp])

    # Compact waveform branch for local 4 GB GPUs.
    wave = BatchNormalization(name="wave_input_batch_norm")(input_layer)
    wave = Conv1D(
        filters=16,
        kernel_size=7,
        strides=1,
        padding="same",
        activation=activation,
        kernel_regularizer=kernel_regularizer_cnn,
        use_bias=bias_cnn,
        name="wave_conv_1",
    )(wave)
    wave = Conv1D(
        filters=16,
        kernel_size=7,
        strides=1,
        padding="same",
        activation=activation,
        kernel_regularizer=kernel_regularizer_cnn,
        use_bias=bias_cnn,
        name="wave_conv_2",
    )(wave)
    wave = SpatialDropout1D(0.03, name="wave_stem_dropout")(wave)
    wave = MaxPooling1D(pool_size=2, name="wave_pool_1")(wave)
    wave_scale1 = wave

    wave = residual_conv_block(
        wave,
        filters=24,
        kernel_size=5,
        activation=activation,
        bias_cnn=bias_cnn,
        kernel_regularizer_cnn=kernel_regularizer_cnn,
        block_name="wave_res_1",
        dropout_rate=0.03,
    )
    wave = MaxPooling1D(pool_size=2, name="wave_pool_2")(wave)
    wave_scale2 = wave

    wave = residual_conv_block(
        wave,
        filters=32,
        kernel_size=3,
        activation=activation,
        bias_cnn=bias_cnn,
        kernel_regularizer_cnn=kernel_regularizer_cnn,
        block_name="wave_res_2",
        dilation_rate=2,
        dropout_rate=0.05,
    )
    wave = MaxPooling1D(pool_size=2, name="wave_pool_3")(wave)
    wave_scale3 = wave

    wave = residual_conv_block(
        wave,
        filters=48,
        kernel_size=3,
        activation=activation,
        bias_cnn=bias_cnn,
        kernel_regularizer_cnn=kernel_regularizer_cnn,
        block_name="wave_res_3",
        dropout_rate=0.06,
    )
    wave = residual_conv_block(
        wave,
        filters=64,
        kernel_size=3,
        activation=activation,
        bias_cnn=bias_cnn,
        kernel_regularizer_cnn=kernel_regularizer_cnn,
        block_name="wave_res_4",
        dilation_rate=2,
        dropout_rate=0.08,
    )
    wave_scale4 = wave

    # Multi-scale feature fusion.
    features = Concatenate(name="feature_fusion")(
        [
            pooled_1d(wave_scale1, "wave_s1"),
            pooled_1d(wave_scale2, "wave_s2"),
            pooled_1d(wave_scale3, "wave_s3"),
            pooled_1d(wave_scale4, "wave_s4"),
        ]
    )
    features_norm = LayerNormalization(name="features_layer_norm")(features)

    shared_head = Dense(
        192,
        activation=activation,
        use_bias=bias_head,
        name="shared_head_dense_1",
    )(features_norm)
    shared_head = Dropout(dropout_head, name="shared_head_dropout")(shared_head)
    shared_head = Dense(
        128,
        activation=activation,
        use_bias=bias_head,
        name="shared_head_dense_2",
    )(shared_head)
    shared_head = Dropout(max(dropout_head * 0.5, 0.06), name="shared_head_dropout_2")(shared_head)
    shared_head = Dense(
        96,
        activation=activation,
        use_bias=bias_head,
        name="shared_head_dense_3",
    )(shared_head)
    shared_head = Dropout(max(dropout_head * 0.3, 0.05), name="shared_head_dropout_3")(shared_head)

    outputs = {}

    for head_name, output_dim in numeric_output_dims.items():
        reg_hidden = Dense(
            96,
            activation=activation,
            use_bias=bias_head,
            name=f"{head_name}_dense",
        )(shared_head)
        reg_hidden = Dropout(max(dropout_head * 0.35, 0.05), name=f"{head_name}_dropout")(reg_hidden)
        reg_hidden = Dense(
            64,
            activation=activation,
            use_bias=bias_head,
            name=f"{head_name}_dense_2",
        )(reg_hidden)
        outputs[head_name] = Dense(
            output_dim,
            activation=None,
            use_bias=bias_head,
            dtype="float32",
            name=head_name,
        )(reg_hidden)

    if has_freq_head:
        freq_hidden = Dense(
            64,
            activation=activation,
            use_bias=bias_head,
            name="freq_head_dense",
        )(shared_head)
        freq_hidden = Dropout(max(dropout_head * 0.25, 0.05), name="freq_head_dropout")(freq_hidden)
        outputs["freq_head"] = Dense(
            1,
            activation=None,
            use_bias=bias_head,
            dtype="float32",
            name="freq_head",
        )(freq_hidden)

    for col, n_classes in categorical_output_dims.items():
        head_name = cat_output_name(col)

        if col == "algorithm":
            hidden_units = 96
        elif col == "style":
            hidden_units = 64
        else:
            hidden_units = 48

        cat_hidden = Dense(
            hidden_units,
            activation=activation,
            use_bias=bias_head,
            name=f"{head_name}_dense",
        )(shared_head)
        cat_hidden = Dropout(max(dropout_head * 0.25, 0.04), name=f"{head_name}_dropout")(cat_hidden)

        outputs[head_name] = Dense(
            n_classes,
            activation="softmax",
            use_bias=bias_head,
            dtype="float32",
            name=head_name,
        )(cat_hidden)

    return Model(input_layer, outputs, name="complete_multihead_light_0_4")


numeric_output_dims = {
    head_name: len(spec["cols"]) for head_name, spec in numeric_head_specs.items()
}

model = build_model(
    input_len=audio_len,
    input_dims=1,
    activation="gelu",
    bias_cnn=True,
    kernel_regularizer_cnn=None,
    bias_head=True,
    dropout_head=0.25,
    numeric_output_dims=numeric_output_dims,
    has_freq_head=freq_col is not None,
    categorical_output_dims=cat_dims,
)

losses = {}
metrics = {}
loss_weights = {}

for head_name in numeric_head_specs:
    losses[head_name] = tf.keras.losses.Huber(delta=0.75)
    metrics[head_name] = ["mae", "mse"]
    loss_weights[head_name] = numeric_head_loss_weight(head_name)

if freq_col is not None:
    losses["freq_head"] = tf.keras.losses.Huber(delta=0.5)
    metrics["freq_head"] = ["mae", "mse"]
    loss_weights["freq_head"] = FREQ_HEAD_LOSS_WEIGHT

for col in categorical_cols:
    head_name = cat_output_name(col)
    losses[head_name] = sparse_categorical_crossentropy_smoothed(CAT_LABEL_SMOOTHING)
    metrics[head_name] = ["sparse_categorical_accuracy"]
    loss_weights[head_name] = categorical_loss_weight(col)

model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        clipnorm=CLIPNORM,
    ),
    loss=losses,
    metrics=metrics,
    loss_weights=loss_weights,
)

plot_model(
    model,
    to_file=os.path.join(OUTPUT_DIR, "model.png"),
    show_shapes=True,
    expand_nested=True,
    rankdir="TB",
    dpi=250,
)


# -------------------------
# Treino
# -------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
    tf.keras.callbacks.TerminateOnNaN(),
]

train_sequence = MultiHeadAudioSequence(
    audio_store,
    fit_audio_indices,
    y_fit,
    batch_size=BATCH_SIZE,
    shuffle=True,
    sample_weights=train_style_weights,
)
val_sequence = MultiHeadAudioSequence(
    audio_store,
    val_audio_indices,
    y_val,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

history = model.fit(
    train_sequence,
    epochs=220,
    validation_data=val_sequence,
    callbacks=callbacks,
)

hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch

numeric_head_names = list(numeric_head_specs.keys())

# Agregados para facilitar leitura de treino por grupo de heads
train_num_mse_cols = [f"{h}_mse" for h in numeric_head_names if f"{h}_mse" in hist.columns]
val_num_mse_cols = [f"val_{h}_mse" for h in numeric_head_names if f"val_{h}_mse" in hist.columns]

if train_num_mse_cols:
    hist["num_group_mse_mean"] = hist[train_num_mse_cols].mean(axis=1)
if val_num_mse_cols:
    hist["val_num_group_mse_mean"] = hist[val_num_mse_cols].mean(axis=1)

train_cat_acc_cols = [
    c
    for c in hist.columns
    if c.startswith("cat__") and c.endswith("sparse_categorical_accuracy")
]
val_cat_acc_cols = [
    c
    for c in hist.columns
    if c.startswith("val_cat__") and c.endswith("sparse_categorical_accuracy")
]
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

# Salvando histórico
hist.to_csv(os.path.join(OUTPUT_DIR, "train_history.csv"), index=False)

plot_metric(hist, "loss", "val_loss", "Loss", "train_history_loss.png")
plot_metric(
    hist,
    "num_group_mse_mean",
    "val_num_group_mse_mean",
    "Numeric Group MSE (mean)",
    "train_history_num_group_mse_mean.png",
)
plot_metric(
    hist,
    "freq_head_mse",
    "val_freq_head_mse",
    "Freq Head MSE",
    "train_history_freq_mse.png",
)
plot_metric(
    hist,
    "cat_acc_mean",
    "val_cat_acc_mean",
    "Categorical Accuracy (mean)",
    "train_history_cat_acc_mean.png",
)
plot_metric(
    hist,
    "cat_loss_mean",
    "val_cat_loss_mean",
    "Categorical Loss (mean)",
    "train_history_cat_loss_mean.png",
)

# Curvas por head crítica
for head_name in numeric_head_names:
    plot_metric(
        hist,
        f"{head_name}_mse",
        f"val_{head_name}_mse",
        f"{head_name} MSE",
        f"train_history_{head_name}_mse.png",
    )

algo_acc_key = f"{cat_output_name('algorithm')}_sparse_categorical_accuracy"
val_algo_acc_key = f"val_{algo_acc_key}"
plot_metric(
    hist,
    algo_acc_key,
    val_algo_acc_key,
    "Algorithm Accuracy",
    "train_history_algorithm_acc.png",
)


# -------------------------
# Teste e métricas
# -------------------------
test_sequence = MultiHeadAudioSequence(
    audio_store,
    test_audio_indices,
    None,
    batch_size=PRED_BATCH_SIZE,
    shuffle=False,
)
raw_pred = model.predict(test_sequence, verbose=0)

if isinstance(raw_pred, dict):
    pred_map = raw_pred
elif isinstance(raw_pred, list):
    pred_map = {name: pred for name, pred in zip(model.output_names, raw_pred, strict=False)}
else:
    pred_map = {model.output_names[0]: raw_pred}

pred_model = pd.DataFrame(index=y_test_model.index)

for head_name, spec in numeric_head_specs.items():
    if head_name not in pred_map:
        raise KeyError(f"Saída numérica ausente no modelo: {head_name}")

    pred_norm = np.asarray(pred_map[head_name], dtype=np.float32)
    pred_values = scaler_by_numeric_head[head_name].inverse_transform(pred_norm)

    cols = spec["cols"]
    log2_cols = set(spec["log2_cols"])

    for col_idx, col in enumerate(cols):
        values = pred_values[:, col_idx]
        if col in log2_cols:
            values = np.power(2.0, values)
        pred_model[col] = values

if freq_col is not None:
    pred_freq_norm = np.asarray(pred_map["freq_head"]).reshape(-1, 1)
    pred_freq_log2 = scaler_freq.inverse_transform(pred_freq_norm)[:, 0]
    pred_freq_hz = np.power(2.0, pred_freq_log2)
    pred_model[freq_col] = pred_freq_hz

for col in categorical_cols:
    head_name = cat_output_name(col)
    pred_logits = np.asarray(pred_map[head_name])
    pred_cls = np.argmax(pred_logits, axis=1).astype(np.int32)
    pred_model[col] = pred_cls

pred_model = pred_model[target_model.columns]

pred_full = pred_model.copy()
for col, value in constant_targets.items():
    pred_full[col] = value

pred_full = pred_full[target_all.columns]
y_true_full = y_test_full[target_all.columns]

y_true_np = y_true_full.to_numpy(dtype=np.float32)
y_pred_np = pred_full.to_numpy(dtype=np.float32)

mse = tf.keras.losses.MSE(y_true_np, y_pred_np).numpy().mean()
mae = tf.keras.losses.MAE(y_true_np, y_pred_np).numpy().mean()
rmse = np.sqrt(mse)

print(f"RMSE Test: {rmse}")
print(f"MSE Test: {mse}")
print(f"MAE Test: {mae}")

metrics_extra = {}
test_metrics_by_head = {}

if freq_col is not None:
    freq_true = y_true_full[freq_col].to_numpy(dtype=np.float64)
    freq_pred = pred_full[freq_col].to_numpy(dtype=np.float64)
    abs_err_hz = np.abs(freq_pred - freq_true)
    rel_err = abs_err_hz / np.maximum(freq_true, 1e-9)
    cents_err = np.abs(
        1200.0
        * np.log2(np.clip(freq_pred, 1e-3, None) / np.clip(freq_true, 1e-3, None))
    )

    metrics_extra.update(
        {
            "freq_mae_hz": float(abs_err_hz.mean()),
            "freq_rmse_hz": float(np.sqrt(np.mean((freq_pred - freq_true) ** 2))),
            "freq_mape": float(rel_err.mean()),
            "freq_mae_cents": float(cents_err.mean()),
        }
    )
    test_metrics_by_head["freq_head"] = {
        "mae_hz": float(abs_err_hz.mean()),
        "rmse_hz": float(np.sqrt(np.mean((freq_pred - freq_true) ** 2))),
        "mse_hz": float(np.mean((freq_pred - freq_true) ** 2)),
        "mape": float(rel_err.mean()),
        "mae_cents": float(cents_err.mean()),
    }

all_num_mse = []
all_num_mae = []

for head_name, spec in numeric_head_specs.items():
    cols = spec["cols"]
    num_true = y_true_full[cols].to_numpy(dtype=np.float64)
    num_pred = pred_full[cols].to_numpy(dtype=np.float64)
    num_err = num_pred - num_true
    num_mse_cols = np.mean(num_err**2, axis=0)
    num_mae_cols = np.mean(np.abs(num_err), axis=0)

    test_metrics_by_head[head_name] = {
        "group_name": spec["group_name"],
        "mse_mean": float(np.mean(num_mse_cols)),
        "mae_mean": float(np.mean(num_mae_cols)),
        "mse_by_col": {
            col: float(val) for col, val in zip(cols, num_mse_cols, strict=False)
        },
        "mae_by_col": {
            col: float(val) for col, val in zip(cols, num_mae_cols, strict=False)
        },
    }

    all_num_mse.extend(num_mse_cols.tolist())
    all_num_mae.extend(num_mae_cols.tolist())

if all_num_mse:
    test_metrics_by_head["numeric_heads_aggregate"] = {
        "mse_mean": float(np.mean(all_num_mse)),
        "mae_mean": float(np.mean(all_num_mae)),
    }

categorical_accuracy = {}
categorical_crossentropy = {}
for col in categorical_cols:
    acc = (
        pred_full[col].to_numpy(dtype=np.int32)
        == y_true_full[col].to_numpy(dtype=np.int32)
    ).mean()
    categorical_accuracy[col] = float(acc)

    head_name = cat_output_name(col)
    y_true_cat = y_true_full[col].to_numpy(dtype=np.int32)
    y_pred_prob = np.asarray(pred_map[head_name], dtype=np.float64)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_cat, y_pred_prob).numpy()
    categorical_crossentropy[col] = float(np.mean(ce))

if categorical_accuracy:
    metrics_extra["categorical_accuracy_mean"] = float(
        np.mean(list(categorical_accuracy.values()))
    )
    test_metrics_by_head["categorical_heads"] = {
        "accuracy_mean": float(np.mean(list(categorical_accuracy.values()))),
        "crossentropy_mean": float(np.mean(list(categorical_crossentropy.values()))),
        "accuracy_by_col": categorical_accuracy,
        "crossentropy_by_col": categorical_crossentropy,
    }

print("===== Test Metrics by Head =====")
if "numeric_heads_aggregate" in test_metrics_by_head:
    print(
        "numeric_heads_aggregate: "
        f"mse_mean={test_metrics_by_head['numeric_heads_aggregate']['mse_mean']:.6f} "
        f"mae_mean={test_metrics_by_head['numeric_heads_aggregate']['mae_mean']:.6f}"
    )
if "freq_head" in test_metrics_by_head:
    print(
        "freq_head: "
        f"mae_hz={test_metrics_by_head['freq_head']['mae_hz']:.3f} "
        f"rmse_hz={test_metrics_by_head['freq_head']['rmse_hz']:.3f} "
        f"mape={test_metrics_by_head['freq_head']['mape']:.6f} "
        f"mae_cents={test_metrics_by_head['freq_head']['mae_cents']:.3f}"
    )
if "categorical_heads" in test_metrics_by_head:
    print(
        "categorical_heads: "
        f"accuracy_mean={test_metrics_by_head['categorical_heads']['accuracy_mean']:.6f} "
        f"crossentropy_mean={test_metrics_by_head['categorical_heads']['crossentropy_mean']:.6f}"
    )


# -------------------------
# Salvando modelo e preprocessadores
# -------------------------
model.save(os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.keras"))

preprocess_bundle = {
    "numeric_heads": numeric_head_specs,
    "freq_col": freq_col,
    "categorical_cols": categorical_cols,
    "constant_targets": constant_targets,
    "scaler_by_numeric_head": scaler_by_numeric_head,
    "scaler_freq": scaler_freq,
    "algorithm_merge_map": ALGORITHM_MERGE_MAP,
    "audio_manifest_path": AUDIO_MANIFEST_PATH,
    "audio_shard_dir": "audio_big6_shards",
    "audio_shard_size": int(meta.get("audio_shard_size", 256)),
    "audio_dtype_disk": "int16",
    "audio_dtype_runtime": AUDIO_DTYPE,
    "audio_sample_len": audio_len,
    "split_index_files": {
        "train": "train_audio_indices.npy",
        "val": "val_audio_indices.npy",
        "test": "test_audio_indices.npy",
    },
}

joblib.dump(
    preprocess_bundle,
    os.path.join(OUTPUT_DIR, f"target_preprocess_{MODEL_NAME}.save"),
)

# Guardando predição do conjunto de teste
pred_full.to_csv(os.path.join(OUTPUT_DIR, "params_pred_test.csv"), index=False)


# -------------------------
# Resultados em JSON
# -------------------------
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
    "dataset": BASE_PATH,
    "tamanho_dataset": int(tamanho_dataset),
    "audio_source": {
        "manifest_path": AUDIO_MANIFEST_PATH,
        "manifest_present": bool(os.path.exists(AUDIO_MANIFEST_PATH)),
        "shard_dir": "audio_big6_shards",
        "cache_kind": "sharded_int16",
        "shard_size": int(meta.get("audio_shard_size", 256)),
        "audio_sample_len": int(audio_len),
        "audio_dtype_disk": "int16",
        "audio_dtype_runtime": AUDIO_DTYPE,
    },
    "split": {
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "train_frac": float(TRAIN_FRAC),
        "val_frac_of_train": float(VAL_FRAC),
        "train_audio_indices_file": "train_audio_indices.npy",
        "val_audio_indices_file": "val_audio_indices.npy",
        "test_audio_indices_file": "test_audio_indices.npy",
    },
    "runtime_config": {
        "batch_size": int(BATCH_SIZE),
        "pred_batch_size": int(PRED_BATCH_SIZE),
        "mixed_precision": bool(USE_MIXED_PRECISION),
        "mixed_precision_policy": mixed_precision.global_policy().name,
        "disable_xla_jit": bool(DISABLE_XLA_JIT),
        "audio_dtype": AUDIO_DTYPE,
        "sequence_training": True,
    },
    "target_groups": {
        "numeric_groups": {
            spec["group_name"]: spec["cols"] for spec in numeric_head_specs.values()
        },
        "ratio_log2_cols": numeric_groups.get("ratio", []),
        "freq_col": freq_col,
        "categorical_cols": categorical_cols,
    },
    "algorithm_label_merges": ALGORITHM_MERGE_MAP,
    "removed_constant_targets": constant_targets,
    "loss_weights": {
        "numeric": {
            k: float(v)
            for k, v in {
                "ratio_head": RATIO_HEAD_LOSS_WEIGHT,
                "index_head": INDEX_HEAD_LOSS_WEIGHT,
                "detune_head": DETUNE_HEAD_LOSS_WEIGHT,
                "env_head": ENV_HEAD_LOSS_WEIGHT,
                "phase_head": PHASE_HEAD_LOSS_WEIGHT,
                "other_head": OTHER_HEAD_LOSS_WEIGHT,
            }.items()
        },
        "freq_head": float(FREQ_HEAD_LOSS_WEIGHT),
        "categorical": {
            "default": float(CAT_LOSS_WEIGHT_DEFAULT),
            "algorithm": float(CAT_LOSS_WEIGHT_ALGORITHM),
            "style": float(CAT_LOSS_WEIGHT_STYLE),
            "env_curve": float(CAT_LOSS_WEIGHT_ENV_CURVE),
        },
        "categorical_by_col": {col: float(categorical_loss_weight(col)) for col in categorical_cols},
    },
    "metrics": {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
    },
    "metrics_extra": metrics_extra,
    "categorical_accuracy": categorical_accuracy,
    "categorical_crossentropy": categorical_crossentropy,
    "test_metrics_by_head": test_metrics_by_head,
    "history_last": history_last,
    "best_val_loss": {
        "epoch": best_val_loss_epoch,
        "value": best_val_loss,
    },
    "categorical_maps": categorical_maps,
}

with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
