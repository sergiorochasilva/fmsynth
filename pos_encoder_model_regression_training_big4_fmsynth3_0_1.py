"""Train a multi-head regressor from latent vectors on `dataset_big4`.

Architecture:
- Dense regression trunk over normalized latent vectors
- Separate heads for ratio, index, detune, envelope, phase, and other numeric groups
- Optional frequency head for absolute frequency prediction

Data flow:
- Input: `dataset_big4/parameters.csv` and `dataset_big4_encoded/latent_big4_fmsynth3_0_1.npy`
- Output: regression `.keras`, scaler/preprocess bundle, test predictions, history, and `results.json`
"""

import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import BatchNormalization, Dense, Dropout, Input
from keras.models import Model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.utils import plot_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_PATH = os.getenv("REG_BASE_PATH", "dataset_big4")
LATENT_PATH = os.getenv("REG_LATENT_PATH", "dataset_big4_encoded/latent_big4_fmsynth3_0_1.npy")
OUTPUT_DIR = os.getenv(
    "REG_OUTPUT_DIR",
    "pos_encoder_model_regression_training_big4_fmsynth3_0_1",
)
MODEL_NAME = "pos_encoder_model_regression_training_big4_fmsynth3_0_1"

RANDOM_STATE = int(os.getenv("REG_RANDOM_STATE", "0"))
TRAIN_FRAC = float(os.getenv("REG_TRAIN_FRAC", "0.75"))
VAL_FRAC = float(os.getenv("REG_VAL_FRAC", "0.2"))
MAX_SAMPLES = int(os.getenv("REG_MAX_SAMPLES", "0"))

BATCH_SIZE = int(os.getenv("REG_BATCH_SIZE", "64"))
EPOCHS = int(os.getenv("REG_EPOCHS", "240"))
LEARNING_RATE = float(os.getenv("REG_LEARNING_RATE", "1e-3"))

DROPOUT_1 = float(os.getenv("REG_DROPOUT_1", "0.20"))
DROPOUT_2 = float(os.getenv("REG_DROPOUT_2", "0.15"))

# Pesos por head (espelhando filosofia do 0_6)
RATIO_HEAD_LOSS_WEIGHT = float(os.getenv("REG_RATIO_HEAD_LOSS_WEIGHT", "2.4"))
INDEX_HEAD_LOSS_WEIGHT = float(os.getenv("REG_INDEX_HEAD_LOSS_WEIGHT", "2.0"))
DETUNE_HEAD_LOSS_WEIGHT = float(os.getenv("REG_DETUNE_HEAD_LOSS_WEIGHT", "1.5"))
ENV_HEAD_LOSS_WEIGHT = float(os.getenv("REG_ENV_HEAD_LOSS_WEIGHT", "0.9"))
PHASE_HEAD_LOSS_WEIGHT = float(os.getenv("REG_PHASE_HEAD_LOSS_WEIGHT", "0.2"))
OTHER_HEAD_LOSS_WEIGHT = float(os.getenv("REG_OTHER_HEAD_LOSS_WEIGHT", "0.8"))
FREQ_HEAD_LOSS_WEIGHT = float(os.getenv("REG_FREQ_HEAD_LOSS_WEIGHT", "2.0"))

CONDITION_COLS_RAW = os.getenv("REG_CONDITION_COLS", "algorithm,style")
CONDITION_COLS_REQUESTED = [x.strip() for x in CONDITION_COLS_RAW.split(",") if x.strip()]

USE_MIXED_PRECISION = os.getenv("REG_MIXED_PRECISION", "0") == "1"
DISABLE_XLA_JIT = os.getenv("REG_DISABLE_XLA_JIT", "1") == "1"

ALGORITHM_MERGE_MAP = {
    "dual_chain": "series2x2_parallel1",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

if USE_MIXED_PRECISION:
    mixed_precision.set_global_policy("mixed_float16")
if DISABLE_XLA_JIT:
    tf.config.optimizer.set_jit(False)

print(
    "Runtime config: "
    f"batch_size={BATCH_SIZE}, epochs={EPOCHS}, lr={LEARNING_RATE}, "
    f"mixed_precision={USE_MIXED_PRECISION}, policy={mixed_precision.global_policy().name}, "
    f"disable_xla_jit={DISABLE_XLA_JIT}, max_samples={MAX_SAMPLES}, "
    f"condition_cols={CONDITION_COLS_REQUESTED}"
)


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

    plt.figure(dpi=350)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.plot(history_df["epoch"], history_df[train_key], label=f"{ylabel} Training")
    plt.plot(history_df["epoch"], history_df[val_key], label=f"{ylabel} Validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=350)
    plt.close()


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


def build_model(input_dim: int, numeric_output_dims: dict[str, int], has_freq_head: bool) -> Model:
    input_layer = Input(shape=(input_dim,), name="latent_cond_input")

    x = BatchNormalization(name="input_batch_norm")(input_layer)
    x = Dense(384, activation="gelu", name="shared_dense_1")(x)
    x = Dropout(DROPOUT_1, name="shared_dropout_1")(x)
    x = Dense(256, activation="gelu", name="shared_dense_2")(x)
    x = Dropout(DROPOUT_2, name="shared_dropout_2")(x)
    shared = Dense(160, activation="gelu", name="shared_dense_3")(x)

    outputs = {}

    for head_name, output_dim in numeric_output_dims.items():
        h = Dense(128, activation="gelu", name=f"{head_name}_dense")(shared)
        h = Dropout(0.08, name=f"{head_name}_dropout")(h)
        outputs[head_name] = Dense(
            output_dim,
            activation=None,
            dtype="float32",
            name=head_name,
        )(h)

    if has_freq_head:
        hf = Dense(96, activation="gelu", name="freq_head_dense")(shared)
        outputs["freq_head"] = Dense(
            1,
            activation=None,
            dtype="float32",
            name="freq_head",
        )(hf)

    return Model(input_layer, outputs, name="post_encoder_regressor_0_1")


def build_condition_matrix(
    target_df: pd.DataFrame,
    condition_cols: list[str],
    categorical_maps: dict[str, list[str]],
) -> tuple[np.ndarray, dict[str, dict]]:
    mats = []
    spec = {}
    offset = 0

    for col in condition_cols:
        if col not in target_df.columns:
            continue
        if col not in categorical_maps:
            continue

        n_classes = len(categorical_maps[col])
        if n_classes <= 0:
            continue

        codes = target_df[col].to_numpy(dtype=np.int32)
        codes = np.clip(codes, 0, n_classes - 1)
        one_hot = np.eye(n_classes, dtype=np.float32)[codes]

        start = offset
        end = offset + n_classes
        spec[col] = {
            "n_classes": int(n_classes),
            "slice_start": int(start),
            "slice_end": int(end),
            "categories": list(categorical_maps[col]),
        }

        mats.append(one_hot)
        offset = end

    if mats:
        return np.concatenate(mats, axis=1).astype(np.float32), spec

    return np.zeros((len(target_df), 0), dtype=np.float32), spec


# -------------------------
# Carga de dados
# -------------------------
params_path = os.path.join(BASE_PATH, "parameters.csv")
if not os.path.exists(params_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {params_path}")
if not os.path.exists(LATENT_PATH):
    raise FileNotFoundError(f"Arquivo não encontrado: {LATENT_PATH}")

latent_all = np.load(LATENT_PATH).astype(np.float32)
if latent_all.ndim != 2:
    raise ValueError(f"Latent com shape inesperado: {latent_all.shape}")

params_raw = pd.read_csv(params_path)
if "algorithm" in params_raw.columns:
    params_raw["algorithm"] = params_raw["algorithm"].replace(ALGORITHM_MERGE_MAP)

if "id" in params_raw.columns:
    params_raw = params_raw.sort_values("id").reset_index(drop=True)

n_params = len(params_raw)
n_latent = latent_all.shape[0]

if MAX_SAMPLES > 0:
    n = min(n_params, n_latent, MAX_SAMPLES)
else:
    if n_params != n_latent:
        raise ValueError(
            "Quantidade de linhas em parameters.csv e latent.npy diverge: "
            f"{n_params} vs {n_latent}."
        )
    n = n_params

params_raw = params_raw.iloc[:n].reset_index(drop=True)
latent_all = latent_all[:n]

# Codificação categórica
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
numeric_cols = [col for col in target_model.columns if col not in categorical_cols]

freq_col = None
if "frequencia_base" in numeric_cols:
    freq_col = "frequencia_base"
    numeric_cols.remove(freq_col)

numeric_groups = build_numeric_groups(numeric_cols)

condition_cols = [c for c in CONDITION_COLS_REQUESTED if c in categorical_cols]
missing_condition_cols = [c for c in CONDITION_COLS_REQUESTED if c not in condition_cols]
if missing_condition_cols:
    print(f"Aviso: colunas de condição ausentes/inelegíveis: {missing_condition_cols}")

cond_all, condition_spec = build_condition_matrix(target_model, condition_cols, categorical_maps)

# -------------------------
# Split treino/val/test
# -------------------------
all_idx = np.arange(n)
rng = np.random.default_rng(RANDOM_STATE)
rng.shuffle(all_idx)

train_size = int(TRAIN_FRAC * n)
train_idx = all_idx[:train_size]
test_idx = all_idx[train_size:]

val_size = int(VAL_FRAC * len(train_idx))
val_idx = train_idx[:val_size]
fit_idx = train_idx[val_size:]

# Features: latent escalado + condições one-hot
latent_train_raw = latent_all[train_idx]
latent_test_raw = latent_all[test_idx]

latent_scaler = StandardScaler()
latent_train = latent_scaler.fit_transform(latent_train_raw).astype(np.float32)
latent_test = latent_scaler.transform(latent_test_raw).astype(np.float32)

cond_train = cond_all[train_idx]
cond_test = cond_all[test_idx]

x_train = np.concatenate([latent_train, cond_train], axis=1).astype(np.float32)
x_test = np.concatenate([latent_test, cond_test], axis=1).astype(np.float32)

fit_rel_idx = np.arange(len(train_idx))[val_size:]
val_rel_idx = np.arange(len(train_idx))[:val_size]

x_fit = x_train[fit_rel_idx]
x_val = x_train[val_rel_idx]

y_train_model = target_model.iloc[train_idx].reset_index(drop=True)
y_test_model = target_model.iloc[test_idx].reset_index(drop=True)

# -------------------------
# Targets por head
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

    y_fit[head_name] = values_norm[fit_rel_idx]
    y_val[head_name] = values_norm[val_rel_idx]

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
    y_fit["freq_head"] = y_freq_norm[fit_rel_idx]
    y_val["freq_head"] = y_freq_norm[val_rel_idx]

if not y_fit:
    raise ValueError("Nenhuma saída numérica/frequência configurada para regressão.")

# -------------------------
# Modelo
# -------------------------
numeric_output_dims = {
    head_name: len(spec["cols"]) for head_name, spec in numeric_head_specs.items()
}

model = build_model(
    input_dim=x_train.shape[1],
    numeric_output_dims=numeric_output_dims,
    has_freq_head=freq_col is not None,
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

model.compile(
    optimizer=Nadam(learning_rate=LEARNING_RATE),
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
    dpi=220,
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=24, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
]

history = model.fit(
    x_fit,
    y_fit,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=1,
)

hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch

numeric_head_names = list(numeric_head_specs.keys())
train_num_mse_cols = [f"{h}_mse" for h in numeric_head_names if f"{h}_mse" in hist.columns]
val_num_mse_cols = [f"val_{h}_mse" for h in numeric_head_names if f"val_{h}_mse" in hist.columns]

if train_num_mse_cols:
    hist["num_group_mse_mean"] = hist[train_num_mse_cols].mean(axis=1)
if val_num_mse_cols:
    hist["val_num_group_mse_mean"] = hist[val_num_mse_cols].mean(axis=1)

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

for head_name in numeric_head_names:
    plot_metric(
        hist,
        f"{head_name}_mse",
        f"val_{head_name}_mse",
        f"{head_name} MSE",
        f"train_history_{head_name}_mse.png",
    )

# -------------------------
# Avaliação
# -------------------------
raw_pred = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
if isinstance(raw_pred, dict):
    pred_map = raw_pred
elif isinstance(raw_pred, list):
    pred_map = {name: pred for name, pred in zip(model.output_names, raw_pred, strict=False)}
else:
    pred_map = {model.output_names[0]: raw_pred}

pred_model = pd.DataFrame(index=np.arange(len(test_idx)))

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
    pred_model[freq_col] = np.power(2.0, pred_freq_log2)

pred_cols = []
for spec in numeric_head_specs.values():
    pred_cols.extend(spec["cols"])
if freq_col is not None:
    pred_cols.append(freq_col)

# garantindo ordem sem duplicatas
pred_cols = list(dict.fromkeys(pred_cols))
y_true_num = y_test_model[pred_cols].to_numpy(dtype=np.float32)
y_pred_num = pred_model[pred_cols].to_numpy(dtype=np.float32)

mse = float(tf.keras.losses.MSE(y_true_num, y_pred_num).numpy().mean())
mae = float(tf.keras.losses.MAE(y_true_num, y_pred_num).numpy().mean())
rmse = float(np.sqrt(mse))

metrics_extra = {}
test_metrics_by_head = {}

if freq_col is not None:
    freq_true = y_test_model[freq_col].to_numpy(dtype=np.float64)
    freq_pred = pred_model[freq_col].to_numpy(dtype=np.float64)
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
    num_true = y_test_model[cols].to_numpy(dtype=np.float64)
    num_pred = pred_model[cols].to_numpy(dtype=np.float64)
    num_err = num_pred - num_true
    num_mse_cols = np.mean(num_err**2, axis=0)
    num_mae_cols = np.mean(np.abs(num_err), axis=0)

    test_metrics_by_head[head_name] = {
        "group_name": spec["group_name"],
        "mse_mean": float(np.mean(num_mse_cols)),
        "mae_mean": float(np.mean(num_mae_cols)),
        "mse_by_col": {col: float(val) for col, val in zip(cols, num_mse_cols, strict=False)},
        "mae_by_col": {col: float(val) for col, val in zip(cols, num_mae_cols, strict=False)},
    }

    all_num_mse.extend(num_mse_cols.tolist())
    all_num_mae.extend(num_mae_cols.tolist())

if all_num_mse:
    test_metrics_by_head["numeric_heads_aggregate"] = {
        "mse_mean": float(np.mean(all_num_mse)),
        "mae_mean": float(np.mean(all_num_mae)),
    }

print("===== Regression Test Metrics =====")
print(f"RMSE: {rmse:.6f}")
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
if "freq_head" in test_metrics_by_head:
    print(
        "freq_head: "
        f"mae_hz={test_metrics_by_head['freq_head']['mae_hz']:.3f} "
        f"mape={test_metrics_by_head['freq_head']['mape']:.6f}"
    )

# -------------------------
# Salvando artefatos
# -------------------------
model.save(os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.keras"))

preprocess_bundle = {
    "numeric_heads": numeric_head_specs,
    "freq_col": freq_col,
    "scaler_by_numeric_head": scaler_by_numeric_head,
    "scaler_freq": scaler_freq,
    "constant_targets": constant_targets,
    "algorithm_merge_map": ALGORITHM_MERGE_MAP,
    "latent_scaler": latent_scaler,
    "condition_cols": condition_cols,
    "condition_spec": condition_spec,
    "categorical_maps": {col: categorical_maps[col] for col in condition_cols},
}
joblib.dump(
    preprocess_bundle,
    os.path.join(OUTPUT_DIR, f"target_preprocess_{MODEL_NAME}.save"),
)

pred_out = pred_model[pred_cols].copy()
if "id" in params_raw.columns:
    pred_out.insert(0, "id", params_raw.iloc[test_idx]["id"].to_numpy(dtype=np.int32))
pred_out.to_csv(os.path.join(OUTPUT_DIR, "params_pred_test.csv"), index=False)

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
    "latent_dataset": LATENT_PATH,
    "n_samples": int(n),
    "split": {
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_rel_idx)),
        "test_size": int(len(test_idx)),
        "train_frac": float(TRAIN_FRAC),
        "val_frac_of_train": float(VAL_FRAC),
    },
    "runtime_config": {
        "batch_size": int(BATCH_SIZE),
        "epochs": int(EPOCHS),
        "learning_rate": float(LEARNING_RATE),
        "mixed_precision": bool(USE_MIXED_PRECISION),
        "mixed_precision_policy": mixed_precision.global_policy().name,
        "disable_xla_jit": bool(DISABLE_XLA_JIT),
        "condition_cols_requested": CONDITION_COLS_REQUESTED,
        "condition_cols_used": condition_cols,
    },
    "algorithm_label_merges": ALGORITHM_MERGE_MAP,
    "removed_constant_targets": constant_targets,
    "target_groups": {
        "numeric_groups": {spec["group_name"]: spec["cols"] for spec in numeric_head_specs.values()},
        "ratio_log2_cols": numeric_groups.get("ratio", []),
        "freq_col": freq_col,
    },
    "conditioning": {
        "condition_cols": condition_cols,
        "condition_spec": condition_spec,
    },
    "loss_weights": {
        "numeric": {
            "ratio_head": float(RATIO_HEAD_LOSS_WEIGHT),
            "index_head": float(INDEX_HEAD_LOSS_WEIGHT),
            "detune_head": float(DETUNE_HEAD_LOSS_WEIGHT),
            "env_head": float(ENV_HEAD_LOSS_WEIGHT),
            "phase_head": float(PHASE_HEAD_LOSS_WEIGHT),
            "other_head": float(OTHER_HEAD_LOSS_WEIGHT),
        },
        "freq_head": float(FREQ_HEAD_LOSS_WEIGHT),
    },
    "metrics": {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
    },
    "metrics_extra": metrics_extra,
    "test_metrics_by_head": test_metrics_by_head,
    "history_last": history_last,
    "best_val_loss": {
        "epoch": best_val_loss_epoch,
        "value": best_val_loss,
    },
}

with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
