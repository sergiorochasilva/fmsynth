"""Temporal Convolutional Network regressor for `dataset_big4`.

Architecture:
- Raw waveform input
- TCN-style convolutional stem with dilated temporal blocks
- Shared dense trunk with numeric and frequency heads

Data flow:
- Input: `dataset_big4/parameters.csv` and `sample_*.wav`
- Output: trained TCN model `.keras`, preprocessing bundle, predictions, plots, and `results.json`
"""

import json
import os

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
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
)
from keras.models import Model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import plot_model

import joblib

BASE_PATH = "dataset_big4"
OUTPUT_DIR = "tcn_training_big4_fmsynth3_0_1"
MODEL_NAME = "tcn_training_big4_fmsynth3_0_1"

RANDOM_STATE = 0
TRAIN_FRAC = 0.75
VAL_FRAC = 0.2
CAT_LOSS_WEIGHT = 0.15
FREQ_HEAD_LOSS_WEIGHT = 2.0

TCN_FILTERS = 32
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2, 4, 8, 16, 32]
TCN_DROPOUT = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------
# Utilitários
# -------------------------
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

    plt.figure(dpi=400)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.plot(history_df["epoch"], history_df[train_key], label=f"{ylabel} Training")
    plt.plot(history_df["epoch"], history_df[val_key], label=f"{ylabel} Validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=400)
    plt.close()


# -------------------------
# Leitura de metadados e parâmetros
# -------------------------
meta = {}
meta_path = os.path.join(BASE_PATH, "meta.json")
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

tamanho_dataset = int(meta.get("tamanho_dataset", 5000))

params_path = os.path.join(BASE_PATH, "parameters.csv")
if not os.path.exists(params_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {params_path}")

# Lendo parâmetros (targets)
target_raw = pd.read_csv(params_path)
if tamanho_dataset and tamanho_dataset < len(target_raw):
    target_raw = target_raw.iloc[:tamanho_dataset].copy()
else:
    tamanho_dataset = len(target_raw)

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

# Ajuste 4: removendo targets constantes
constant_targets = {}
for col in list(target_all.columns):
    if target_all[col].nunique(dropna=False) <= 1:
        constant_targets[col] = to_json_scalar(target_all[col].iloc[0])

target_model = target_all.drop(columns=list(constant_targets.keys())).copy()
if target_model.empty:
    raise ValueError("Todos os targets foram removidos como constantes.")

categorical_cols = [col for col in target_model.columns if col in categorical_maps]
numeric_cols = [col for col in target_model.columns if col not in categorical_cols]

# Ajuste 3: tratar frequencia_base em head dedicada com transformação log2
freq_col = None
if "frequencia_base" in numeric_cols:
    freq_col = "frequencia_base"
    numeric_cols.remove(freq_col)


# -------------------------
# Lendo o dataset de áudio
# -------------------------
samples = []
for sample_id in sample_ids:
    wav_path = os.path.join(BASE_PATH, f"sample_{sample_id}.wav")
    signal, _sample_rate = sf.read(wav_path)
    samples.append(signal.astype(np.float32))

samples = np.asarray(samples, dtype=np.float32)

if samples.ndim != 2:
    raise ValueError(f"Formato de áudio inesperado: {samples.shape}")

if samples.shape[0] != len(target_all):
    raise ValueError(
        "Quantidade de áudios e targets não confere: "
        f"{samples.shape[0]} vs {len(target_all)}"
    )


# -------------------------
# Split treino / teste
# -------------------------
train_idx = target_all.sample(frac=TRAIN_FRAC, random_state=RANDOM_STATE).index
test_idx = target_all.drop(index=train_idx).index

x_train_raw = samples[train_idx.to_numpy()]
x_test_raw = samples[test_idx.to_numpy()]

y_train_full = target_all.loc[train_idx].reset_index(drop=True)
y_test_full = target_all.loc[test_idx].reset_index(drop=True)

y_train_model = target_model.loc[train_idx].reset_index(drop=True)
y_test_model = target_model.loc[test_idx].reset_index(drop=True)

# Split treino / validação (determinístico)
val_idx = y_train_model.sample(frac=VAL_FRAC, random_state=RANDOM_STATE).index.to_numpy()
fit_idx = y_train_model.drop(index=val_idx).index.to_numpy()

# Ajustando dimensão de X
x_train = x_train_raw.reshape((x_train_raw.shape[0], x_train_raw.shape[1], 1))
x_test = x_test_raw.reshape((x_test_raw.shape[0], x_test_raw.shape[1], 1))

x_fit = x_train[fit_idx]
x_val = x_train[val_idx]

# Salvando arrays
audio_len_suffix = "big4"
np.save(os.path.join(OUTPUT_DIR, f"x_train_{audio_len_suffix}.npy"), x_train)
np.save(os.path.join(OUTPUT_DIR, f"x_test_{audio_len_suffix}.npy"), x_test)
np.save(
    os.path.join(OUTPUT_DIR, f"y_train_{audio_len_suffix}.npy"),
    y_train_full.to_numpy(dtype=np.float32),
)
np.save(
    os.path.join(OUTPUT_DIR, f"y_test_{audio_len_suffix}.npy"),
    y_test_full.to_numpy(dtype=np.float32),
)


# -------------------------
# Preparando y por head
# -------------------------
y_fit = {}
y_val = {}

scaler_num = None
if numeric_cols:
    scaler_num = StandardScaler()
    y_num = y_train_model[numeric_cols].to_numpy(dtype=np.float32)
    y_num_norm = scaler_num.fit_transform(y_num).astype(np.float32)
    y_fit["num_head"] = y_num_norm[fit_idx]
    y_val["num_head"] = y_num_norm[val_idx]

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


# -------------------------
# Modelo
# -------------------------
# Configurando para não alocar diretamente toda a memória da GPU
# (alocar conforme necessário)
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def tcn_residual_block(
    x,
    filters,
    kernel_size,
    dilation_rate,
    activation,
    dropout_rate,
    bias_tcn,
    kernel_regularizer_tcn,
    block_name,
):
    residual = x

    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        use_bias=bias_tcn,
        kernel_regularizer=kernel_regularizer_tcn,
        name=f"{block_name}_conv_1",
    )(x)
    x = BatchNormalization(name=f"{block_name}_bn_1")(x)
    x = Activation(activation, name=f"{block_name}_act_1")(x)
    x = Dropout(dropout_rate, name=f"{block_name}_drop_1")(x)

    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        use_bias=bias_tcn,
        kernel_regularizer=kernel_regularizer_tcn,
        name=f"{block_name}_conv_2",
    )(x)
    x = BatchNormalization(name=f"{block_name}_bn_2")(x)

    if residual.shape[-1] != filters:
        residual = Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            use_bias=bias_tcn,
            kernel_regularizer=kernel_regularizer_tcn,
            name=f"{block_name}_res_proj",
        )(residual)

    x = Add(name=f"{block_name}_add")([x, residual])
    x = Activation(activation, name=f"{block_name}_act_out")(x)
    return x


def build_model(
    input_len,
    input_dims,
    activation,
    bias_tcn,
    kernel_regularizer_tcn,
    dropout_tcn,
    bias_head,
    dropout_head,
    numeric_output_dim,
    has_freq_head,
    categorical_output_dims,
):
    input_layer = Input(shape=(input_len, input_dims), name="audio_input")

    x_n = BatchNormalization(name="input_batch_norm")(input_layer)

    # Stem para reduzir custo temporal antes dos blocos dilatados.
    x_n = Conv1D(
        filters=TCN_FILTERS,
        kernel_size=TCN_KERNEL_SIZE,
        strides=2,
        padding="causal",
        use_bias=bias_tcn,
        kernel_regularizer=kernel_regularizer_tcn,
        name="tcn_stem_conv",
    )(x_n)
    x_n = BatchNormalization(name="tcn_stem_bn")(x_n)
    x_n = Activation(activation, name="tcn_stem_act")(x_n)

    for idx, dilation in enumerate(TCN_DILATIONS, start=1):
        x_n = tcn_residual_block(
            x=x_n,
            filters=TCN_FILTERS,
            kernel_size=TCN_KERNEL_SIZE,
            dilation_rate=dilation,
            activation=activation,
            dropout_rate=dropout_tcn,
            bias_tcn=bias_tcn,
            kernel_regularizer_tcn=kernel_regularizer_tcn,
            block_name=f"tcn_block_{idx}",
        )

    x_n = Conv1D(
        filters=TCN_FILTERS * 2,
        kernel_size=1,
        padding="same",
        activation=activation,
        use_bias=bias_tcn,
        kernel_regularizer=kernel_regularizer_tcn,
        name="tcn_post_conv",
    )(x_n)

    features_flat = GlobalAveragePooling1D(name="features_gap")(x_n)

    # Ajuste 1: garantir uso do tensor normalizado nas heads
    features_norm = BatchNormalization(name="features_batch_norm")(features_flat)

    hidden_dim = int(features_norm.shape[-1])
    hidden_1 = max(hidden_dim // 2, 64)
    hidden_2 = max(hidden_dim // 4, 32)

    shared_head = Dense(
        hidden_1,
        activation=activation,
        use_bias=bias_head,
        name="shared_head_dense_1",
    )(features_norm)
    shared_head = Dropout(dropout_head, name="shared_head_dropout")(shared_head)
    shared_head = Dense(
        hidden_2,
        activation=activation,
        use_bias=bias_head,
        name="shared_head_dense_2",
    )(shared_head)

    outputs = {}

    if numeric_output_dim > 0:
        outputs["num_head"] = Dense(
            numeric_output_dim,
            activation=None,
            use_bias=bias_head,
            name="num_head",
        )(shared_head)

    if has_freq_head:
        outputs["freq_head"] = Dense(
            1,
            activation=None,
            use_bias=bias_head,
            name="freq_head",
        )(shared_head)

    for col, n_classes in categorical_output_dims.items():
        head_name = cat_output_name(col)
        cat_hidden = Dense(
            64,
            activation=activation,
            use_bias=bias_head,
            name=f"{head_name}_dense",
        )(features_norm)
        outputs[head_name] = Dense(
            n_classes,
            activation="softmax",
            use_bias=bias_head,
            name=head_name,
        )(cat_hidden)

    return Model(input_layer, outputs, name="complete_multihead_tcn")


model = build_model(
    input_len=x_train.shape[1],
    input_dims=x_train.shape[2],
    activation="gelu",
    bias_tcn=True,
    kernel_regularizer_tcn=None,
    dropout_tcn=TCN_DROPOUT,
    bias_head=True,
    dropout_head=0.2,
    numeric_output_dim=len(numeric_cols),
    has_freq_head=freq_col is not None,
    categorical_output_dims=cat_dims,
)

losses = {}
metrics = {}
loss_weights = {}

if numeric_cols:
    losses["num_head"] = "mse"
    metrics["num_head"] = ["mae", "mse"]
    loss_weights["num_head"] = 1.0

if freq_col is not None:
    losses["freq_head"] = tf.keras.losses.Huber(delta=0.5)
    metrics["freq_head"] = ["mae", "mse"]
    loss_weights["freq_head"] = FREQ_HEAD_LOSS_WEIGHT

for col in categorical_cols:
    head_name = cat_output_name(col)
    losses[head_name] = "sparse_categorical_crossentropy"
    metrics[head_name] = ["sparse_categorical_accuracy"]
    loss_weights[head_name] = CAT_LOSS_WEIGHT

model.compile(
    optimizer="Nadam",
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
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
]

history = model.fit(
    x_fit,
    y_fit,
    epochs=200,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
)

hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch

# Agregados para facilitar leitura de treino por grupo de heads
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
    "num_head_mse",
    "val_num_head_mse",
    "Numeric MSE",
    "train_history_num_mse.png",
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


# -------------------------
# Teste e métricas
# -------------------------
raw_pred = model.predict(x_test, verbose=0)

if isinstance(raw_pred, dict):
    pred_map = raw_pred
elif isinstance(raw_pred, list):
    pred_map = {name: pred for name, pred in zip(model.output_names, raw_pred)}
else:
    pred_map = {model.output_names[0]: raw_pred}

pred_model = pd.DataFrame(index=y_test_model.index)

if numeric_cols:
    pred_num = scaler_num.inverse_transform(pred_map["num_head"])
    pred_model[numeric_cols] = pred_num

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

if numeric_cols:
    num_true = y_true_full[numeric_cols].to_numpy(dtype=np.float64)
    num_pred = pred_full[numeric_cols].to_numpy(dtype=np.float64)
    num_err = num_pred - num_true
    num_mse_cols = np.mean(num_err**2, axis=0)
    num_mae_cols = np.mean(np.abs(num_err), axis=0)
    test_metrics_by_head["num_head"] = {
        "mse_mean": float(np.mean(num_mse_cols)),
        "mae_mean": float(np.mean(num_mae_cols)),
        "mse_by_col": {
            col: float(val) for col, val in zip(numeric_cols, num_mse_cols, strict=False)
        },
        "mae_by_col": {
            col: float(val) for col, val in zip(numeric_cols, num_mae_cols, strict=False)
        },
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
if "num_head" in test_metrics_by_head:
    print(
        "num_head: "
        f"mse_mean={test_metrics_by_head['num_head']['mse_mean']:.6f} "
        f"mae_mean={test_metrics_by_head['num_head']['mae_mean']:.6f}"
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
    "numeric_cols": numeric_cols,
    "freq_col": freq_col,
    "categorical_cols": categorical_cols,
    "constant_targets": constant_targets,
    "scaler_num": scaler_num,
    "scaler_freq": scaler_freq,
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
for key in [
    "loss",
    "val_loss",
    "num_head_mse",
    "val_num_head_mse",
    "num_head_mae",
    "val_num_head_mae",
    "freq_head_mse",
    "val_freq_head_mse",
    "freq_head_mae",
    "val_freq_head_mae",
]:
    if key in hist:
        history_last[key] = float(hist[key].iloc[-1])

if "cat_acc_mean" in hist:
    history_last["cat_acc_mean"] = float(hist["cat_acc_mean"].iloc[-1])
if "val_cat_acc_mean" in hist:
    history_last["val_cat_acc_mean"] = float(hist["val_cat_acc_mean"].iloc[-1])
if "cat_loss_mean" in hist:
    history_last["cat_loss_mean"] = float(hist["cat_loss_mean"].iloc[-1])
if "val_cat_loss_mean" in hist:
    history_last["val_cat_loss_mean"] = float(hist["val_cat_loss_mean"].iloc[-1])

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
    "split": {
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "train_frac": float(TRAIN_FRAC),
        "val_frac_of_train": float(VAL_FRAC),
    },
    "target_groups": {
        "numeric_cols": numeric_cols,
        "freq_col": freq_col,
        "categorical_cols": categorical_cols,
    },
    "removed_constant_targets": constant_targets,
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
