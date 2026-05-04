"""Train a multi-head classifier from latent vectors on `dataset_big4`.

Architecture:
- Dense network over normalized latent vectors
- One softmax head per categorical synthesis parameter
- Shared trunk with per-head capacity for algorithm/style/envelope curves

Data flow:
- Input: `dataset_big4/parameters.csv` and `dataset_big4_encoded/latent_big4_fmsynth3_0_1.npy`
- Output: classifier `.keras`, target preprocessing bundle, test predictions, history, and `results.json`
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

BASE_PATH = os.getenv("CLASS_BASE_PATH", "dataset_big4")
LATENT_PATH = os.getenv("CLASS_LATENT_PATH", "dataset_big4_encoded/latent_big4_fmsynth3_0_1.npy")
OUTPUT_DIR = os.getenv(
    "CLASS_OUTPUT_DIR",
    "pos_encoder_model_classification_training_big4_fmsynth3_0_1",
)
MODEL_NAME = "pos_encoder_model_classification_training_big4_fmsynth3_0_1"

RANDOM_STATE = int(os.getenv("CLASS_RANDOM_STATE", "0"))
TRAIN_FRAC = float(os.getenv("CLASS_TRAIN_FRAC", "0.75"))
VAL_FRAC = float(os.getenv("CLASS_VAL_FRAC", "0.2"))
MAX_SAMPLES = int(os.getenv("CLASS_MAX_SAMPLES", "0"))

BATCH_SIZE = int(os.getenv("CLASS_BATCH_SIZE", "64"))
EPOCHS = int(os.getenv("CLASS_EPOCHS", "220"))
LEARNING_RATE = float(os.getenv("CLASS_LEARNING_RATE", "1e-3"))

DROPOUT_1 = float(os.getenv("CLASS_DROPOUT_1", "0.20"))
DROPOUT_2 = float(os.getenv("CLASS_DROPOUT_2", "0.15"))

CAT_LOSS_WEIGHT_DEFAULT = float(os.getenv("CLASS_CAT_LOSS_WEIGHT_DEFAULT", "0.05"))
CAT_LOSS_WEIGHT_ALGORITHM = float(os.getenv("CLASS_CAT_LOSS_WEIGHT_ALGORITHM", "1.0"))
CAT_LOSS_WEIGHT_STYLE = float(os.getenv("CLASS_CAT_LOSS_WEIGHT_STYLE", "0.20"))
CAT_LOSS_WEIGHT_ENV_CURVE = float(os.getenv("CLASS_CAT_LOSS_WEIGHT_ENV_CURVE", "0.06"))

USE_MIXED_PRECISION = os.getenv("CLASS_MIXED_PRECISION", "0") == "1"
DISABLE_XLA_JIT = os.getenv("CLASS_DISABLE_XLA_JIT", "1") == "1"

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
    f"disable_xla_jit={DISABLE_XLA_JIT}, max_samples={MAX_SAMPLES}"
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


def build_model(input_dim: int, categorical_output_dims: dict[str, int]) -> Model:
    input_layer = Input(shape=(input_dim,), name="latent_input")

    x = BatchNormalization(name="latent_batch_norm")(input_layer)
    x = Dense(384, activation="gelu", name="shared_dense_1")(x)
    x = Dropout(DROPOUT_1, name="shared_dropout_1")(x)
    x = Dense(256, activation="gelu", name="shared_dense_2")(x)
    x = Dropout(DROPOUT_2, name="shared_dropout_2")(x)
    shared = Dense(128, activation="gelu", name="shared_dense_3")(x)

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

    return Model(input_layer, outputs, name="post_encoder_classifier_0_1")


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
if not categorical_cols:
    raise ValueError("Nenhuma coluna categórica encontrada para classificação.")

cat_dims = {col: len(categorical_maps[col]) for col in categorical_cols}

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

x_train_raw = latent_all[train_idx]
x_test_raw = latent_all[test_idx]

y_train_model = target_model.iloc[train_idx].reset_index(drop=True)
y_test_model = target_model.iloc[test_idx].reset_index(drop=True)

fit_rel_idx = np.arange(len(train_idx))[val_size:]
val_rel_idx = np.arange(len(train_idx))[:val_size]

# Normalização dos vetores latentes
latent_scaler = StandardScaler()
x_train = latent_scaler.fit_transform(x_train_raw).astype(np.float32)
x_test = latent_scaler.transform(x_test_raw).astype(np.float32)

x_fit = x_train[fit_rel_idx]
x_val = x_train[val_rel_idx]

y_fit = {}
y_val = {}
for col in categorical_cols:
    head_name = cat_output_name(col)
    y_col = y_train_model[col].to_numpy(dtype=np.int32)
    y_fit[head_name] = y_col[fit_rel_idx]
    y_val[head_name] = y_col[val_rel_idx]

# -------------------------
# Modelo
# -------------------------
model = build_model(input_dim=x_train.shape[1], categorical_output_dims=cat_dims)

losses = {}
metrics = {}
loss_weights = {}
for col in categorical_cols:
    head_name = cat_output_name(col)
    losses[head_name] = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics[head_name] = ["sparse_categorical_accuracy"]
    loss_weights[head_name] = categorical_loss_weight(col)

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

hist.to_csv(os.path.join(OUTPUT_DIR, "train_history.csv"), index=False)

plot_metric(hist, "loss", "val_loss", "Loss", "train_history_loss.png")
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
# Avaliação
# -------------------------
raw_pred = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
if isinstance(raw_pred, dict):
    pred_map = raw_pred
elif isinstance(raw_pred, list):
    pred_map = {name: pred for name, pred in zip(model.output_names, raw_pred, strict=False)}
else:
    pred_map = {model.output_names[0]: raw_pred}

categorical_accuracy = {}
categorical_crossentropy = {}
per_head = {}
pred_codes_df = pd.DataFrame(index=np.arange(len(test_idx)))

if "id" in params_raw.columns:
    pred_codes_df["id"] = params_raw.iloc[test_idx]["id"].to_numpy(dtype=np.int32)

for col in categorical_cols:
    head_name = cat_output_name(col)
    y_true = y_test_model[col].to_numpy(dtype=np.int32)
    y_prob = np.asarray(pred_map[head_name], dtype=np.float64)
    y_pred = np.argmax(y_prob, axis=1).astype(np.int32)

    acc = float((y_pred == y_true).mean())
    ce = float(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_prob).numpy().mean())

    categorical_accuracy[col] = acc
    categorical_crossentropy[col] = ce
    per_head[col] = {
        "accuracy": acc,
        "crossentropy": ce,
        "n_classes": int(cat_dims[col]),
    }

    pred_codes_df[f"true__{col}"] = y_true
    pred_codes_df[f"pred__{col}"] = y_pred

# versão decodificada para inspeção humana
pred_labels_df = pred_codes_df.copy()
for col in categorical_cols:
    categories = categorical_maps[col]
    pred_labels_df[f"true_label__{col}"] = pred_codes_df[f"true__{col}"].map(
        lambda x: categories[int(x)] if 0 <= int(x) < len(categories) else "<UNK>"
    )
    pred_labels_df[f"pred_label__{col}"] = pred_codes_df[f"pred__{col}"].map(
        lambda x: categories[int(x)] if 0 <= int(x) < len(categories) else "<UNK>"
    )

metrics_extra = {
    "categorical_accuracy_mean": float(np.mean(list(categorical_accuracy.values()))),
    "categorical_crossentropy_mean": float(np.mean(list(categorical_crossentropy.values()))),
}

print("===== Classification Test Metrics =====")
print(f"categorical_accuracy_mean: {metrics_extra['categorical_accuracy_mean']:.6f}")
print(f"categorical_crossentropy_mean: {metrics_extra['categorical_crossentropy_mean']:.6f}")
if "algorithm" in per_head:
    print(f"algorithm_accuracy: {per_head['algorithm']['accuracy']:.6f}")

# -------------------------
# Salvando artefatos
# -------------------------
model.save(os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.keras"))

preprocess_bundle = {
    "categorical_cols": categorical_cols,
    "categorical_maps": {k: categorical_maps[k] for k in categorical_cols},
    "constant_targets": constant_targets,
    "algorithm_merge_map": ALGORITHM_MERGE_MAP,
    "latent_scaler": latent_scaler,
}
joblib.dump(
    preprocess_bundle,
    os.path.join(OUTPUT_DIR, f"target_preprocess_{MODEL_NAME}.save"),
)

pred_codes_df.to_csv(os.path.join(OUTPUT_DIR, "params_pred_test_codes.csv"), index=False)
pred_labels_df.to_csv(os.path.join(OUTPUT_DIR, "params_pred_test_labels.csv"), index=False)

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
    "metrics": metrics_extra,
    "categorical_accuracy": categorical_accuracy,
    "categorical_crossentropy": categorical_crossentropy,
    "test_metrics_by_head": per_head,
    "history_last": history_last,
    "best_val_loss": {
        "epoch": best_val_loss_epoch,
        "value": best_val_loss,
    },
    "categorical_maps": {k: categorical_maps[k] for k in categorical_cols},
}

with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
