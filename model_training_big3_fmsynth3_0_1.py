"""CNN regressor for `dataset_big3` synthesis parameters.

Architecture:
- Raw waveform input
- 1D CNN feature extractor with pooling
- Flatten + dense regression head

Data flow:
- Input: `dataset_big3/parameters.csv` and `sample_*.wav`
- Output: regression model `.keras`, target scaler, train/test arrays, plots, and `results.json`
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
from keras.callbacks import EarlyStopping
from keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
)
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import plot_model

import joblib

BASE_PATH = "dataset_big3"
OUTPUT_DIR = "model_training_big3_fmsynth3_0_1"
MODEL_NAME = "model_training_big3_fmsynth3_0_1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Leitura de metadados e parametros
# -------------------------
meta = {}
meta_path = os.path.join(BASE_PATH, "meta.json")
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

tamanho_dataset = int(meta.get("tamanho_dataset", 5000))

params_path = os.path.join(BASE_PATH, "parameters.csv")
if not os.path.exists(params_path):
    raise FileNotFoundError(f"Arquivo nao encontrado: {params_path}")

# Lendo parametros (targets)
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

# Definindo chave de estratificacao (balancear classes)
stratify_key = None
if "style" in target_raw.columns and "algorithm" in target_raw.columns:
    stratify_key = (
        target_raw["style"].astype(str) + "|" + target_raw["algorithm"].astype(str)
    )
elif "style" in target_raw.columns:
    stratify_key = target_raw["style"].astype(str)
elif "algorithm" in target_raw.columns:
    stratify_key = target_raw["algorithm"].astype(str)

# Codificando colunas categoricas
categorical_maps = {}
for col in target_raw.columns:
    if target_raw[col].dtype == object:
        cat = pd.Categorical(target_raw[col])
        categorical_maps[col] = [str(x) for x in cat.categories]
        target_raw[col] = cat.codes

# Removendo coluna de id
if "id" in target_raw.columns:
    target = target_raw.drop(columns=["id"])
else:
    target = target_raw

# -------------------------
# Lendo o dataset de audio
# -------------------------
samples = []
for sample_id in sample_ids:
    wav_path = os.path.join(BASE_PATH, f"sample_{sample_id}.wav")
    signal = sf.read(wav_path)
    samples.append(signal)

samples = pd.DataFrame(samples)
# Removendo a coluna do sample rate
samples = samples.drop(columns=[1])

# -------------------------
# Dataset final e split
# -------------------------
ds = pd.concat([samples, target], axis=1)

all_idx = ds.index.to_numpy()
if stratify_key is not None:
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=0.25,
        random_state=0,
        stratify=stratify_key.to_numpy(),
    )
    train = ds.loc[train_idx]
    test = ds.loc[test_idx]
else:
    train = ds.sample(frac=0.75, random_state=0)
    test = ds.drop(train.index)

# Split treino/validacao com estratificacao
if stratify_key is not None:
    stratify_train = stratify_key.loc[train.index].to_numpy()
    train_idx2, val_idx = train_test_split(
        train.index.to_numpy(),
        test_size=0.2,
        random_state=0,
        stratify=stratify_train,
    )
    train = ds.loc[train_idx2]
    val = ds.loc[val_idx]
else:
    train_idx2, val_idx = train_test_split(
        train.index.to_numpy(), test_size=0.2, random_state=0
    )
    train = ds.loc[train_idx2]
    val = ds.loc[val_idx]

# -------------------------
# Preparando X e Y
# -------------------------
x_train = pd.DataFrame(train[0])
x_train = np.array(x_train[0].values.tolist())

y_train = train.drop(columns=[0])

scaler_y = StandardScaler()
y_train_norm = scaler_y.fit_transform(y_train)

# Ajustando dimensao de X
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

# Validacao
x_val = pd.DataFrame(val[0])
x_val = np.array(x_val[0].values.tolist())
x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))

y_val = val.drop(columns=[0])
y_val_norm = scaler_y.transform(y_val)

# Salvando arrays de treino
np.save(os.path.join(OUTPUT_DIR, "x_train_big3.npy"), x_train)
np.save(os.path.join(OUTPUT_DIR, "y_train_big3.npy"), y_train_norm)

# -------------------------
# Modelo
# -------------------------
# Configurando para nao alocar diretamente toda a memoria da GPU
# (alocar conforme necessario)
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def regressor(
    input_dims,
    output_dims,
    activation,
    bias_regressor,
    dropout_regressor: float,
    kernel_regularizer_regressor: str,
):
    input_layer = Input(shape=[input_dims])

    x_0 = Dense(
        int(input_dims / 2),
        activation=activation,
        use_bias=bias_regressor,
        kernel_regularizer=kernel_regularizer_regressor,
    )(input_layer)
    x_0 = Dropout(dropout_regressor)(x_0)
    x_2 = Dense(int(input_dims / 4), activation=activation, use_bias=bias_regressor)(
        x_0
    )
    saidas = Dense(
        output_dims, activation=None, name="regressor_saidas", use_bias=bias_regressor
    )(x_2)

    return Model(input_layer, saidas, name="regressor")


def build_models(
    input_len,
    input_dims,
    output_dims,
    activation,
    bias_cnn,
    kernel_regularizer_cnn,
    bias_regressor,
    dropout_regressor,
    kernel_regularizer_regressor,
):
    # Camadas de entrada
    input_layer = Input(shape=(input_len, input_dims))

    x_n = BatchNormalization()(input_layer)

    # Features 1
    extrator1 = Conv1D(
        filters=8,
        kernel_size=4,
        strides=1,
        activation=activation,
        input_shape=(input_len, 1),
        kernel_regularizer=kernel_regularizer_cnn,
        use_bias=bias_cnn,
    )
    pooling1 = MaxPooling1D(pool_size=16)

    features1 = extrator1(x_n)
    features1 = pooling1(features1)

    extrator1_2 = Conv1D(
        filters=16,
        kernel_size=8,
        strides=1,
        activation=activation,
        input_shape=(input_len, 1),
        kernel_regularizer=kernel_regularizer_cnn,
        use_bias=bias_cnn,
    )
    pooling1_2 = MaxPooling1D(pool_size=16)

    features1_2 = extrator1_2(features1)
    features1_2 = pooling1_2(features1_2)

    extrator1_3 = Conv1D(
        filters=32,
        kernel_size=16,
        strides=1,
        activation=activation,
        input_shape=(input_len, 1),
        kernel_regularizer=kernel_regularizer_cnn,
        use_bias=bias_cnn,
    )
    pooling1_3 = MaxPooling1D(pool_size=16)

    features1_3 = extrator1_3(features1_2)
    features1_3 = pooling1_3(features1_3)

    features1_flatten1 = Flatten()(features1_3)

    features1_flatten1_normalized = BatchNormalization()(features1_flatten1)

    # Regressao
    regressao = regressor(
        features1_flatten1_normalized.shape[1],
        output_dims,
        activation,
        bias_regressor,
        dropout_regressor,
        kernel_regularizer_regressor,
    )

    saida = regressao(features1_flatten1)

    return (
        Model(input_layer, saida, name="complete"),
        Model(input_layer, features1_flatten1, name="projecao"),
        Model(features1_flatten1, saida, name="regressao"),
    )


model, features, regression = build_models(
    input_len=x_train.shape[1],
    input_dims=x_train.shape[2],
    output_dims=y_train_norm.shape[1],
    activation="relu",
    bias_cnn=True,
    kernel_regularizer_cnn=None,
    bias_regressor=True,
    dropout_regressor=0.5,
    kernel_regularizer_regressor="l2",
)

model.compile(optimizer="Nadam", loss="mse", metrics=["mae", "mse"])

# -------------------------
# Imagens dos modelos
# -------------------------
plot_model(
    features,
    to_file=os.path.join(OUTPUT_DIR, "projecao.png"),
    show_shapes=True,
    rankdir="TB",
    dpi=400,
)
plot_model(
    regression,
    to_file=os.path.join(OUTPUT_DIR, "regressao.png"),
    show_shapes=True,
    expand_nested=True,
    rankdir="TB",
    dpi=400,
)

# -------------------------
# Treino
# -------------------------
callback = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

history = model.fit(
    x_train,
    y_train_norm,
    epochs=200,
    validation_data=(x_val, y_val_norm),
    callbacks=[callback],
)

hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch

# Salvando historico
hist.to_csv(os.path.join(OUTPUT_DIR, "train_history.csv"), index=False)

# Grafico MSE
plt.figure(dpi=400)
plt.xlabel("Epoch")
plt.ylabel("Mean Square Error")
plt.plot(hist["epoch"], hist["mse"], label="MSE Training")
plt.plot(hist["epoch"], hist["val_mse"], label="MSE Validation")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "train_history_mse.png"), dpi=400)
plt.close()

# Grafico MAE
plt.figure(dpi=400)
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.plot(hist["epoch"], hist["mae"], label="MAE Training")
plt.plot(hist["epoch"], hist["val_mae"], label="MAE Validation")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "train_history_mae.png"), dpi=400)
plt.close()

# -------------------------
# Teste e metricas
# -------------------------
x_t = pd.DataFrame(test[0])
x_t = np.array(x_t[0].values.tolist())

# Ajustando dimensao de X de teste
x_t = x_t.reshape((x_t.shape[0], x_t.shape[1], 1))

y_t = test.drop(columns=[0])

np.save(os.path.join(OUTPUT_DIR, "x_test_big3.npy"), x_t)
np.save(os.path.join(OUTPUT_DIR, "y_test_big3.npy"), y_t)

y_pred_norm = model.predict(x_t)
y_pred = scaler_y.inverse_transform(y_pred_norm)
y_pred = pd.DataFrame(y_pred, columns=y_t.columns)

mse = tf.keras.losses.MSE(y_t, y_pred).numpy().mean()
mae = tf.keras.losses.MAE(y_t, y_pred).numpy().mean()
rmse = np.sqrt(mse)

print(f"RMSE Test: {rmse}")
print(f"MSE Test: {mse}")
print(f"MAE Test: {mae}")

# -------------------------
# Salvando modelo e scaler
# -------------------------
model.save(os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.keras"))
joblib.dump(scaler_y, os.path.join(OUTPUT_DIR, f"scaler_y_{MODEL_NAME}.save"))

# -------------------------
# Resultados em JSON
# -------------------------
results = {
    "model_name": MODEL_NAME,
    "dataset": BASE_PATH,
    "tamanho_dataset": int(tamanho_dataset),
    "metrics": {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
    },
    "history_last": {
        key: float(hist[key].iloc[-1])
        for key in ["loss", "val_loss", "mse", "val_mse", "mae", "val_mae"]
        if key in hist
    },
    "categorical_maps": categorical_maps,
}

with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
