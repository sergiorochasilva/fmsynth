import IPython.display as ipd
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import soundfile as sf
import pandas as pd

import tensorflow as tf
import keras

# Carregando o modelo salvo
model = keras.models.load_model("model_conv_1_sec_v1_0.keras")

# Carregando o scaler_y
import joblib

scaler_y = joblib.load("scaler_y_conv_1_sec_v1_0.save")

## Carregando o dataset (lendo todos os arquivos .wav da pasta 'nsynth-test/audio')
import os

base_path = "nsynth-test/audio"
file_list = [f for f in os.listdir(base_path) if f.endswith(".wav")]
samples = []
for file_name in file_list:
    signal = sf.read(os.path.join(base_path, file_name))
    samples.append(signal)

samples = pd.DataFrame(samples)

samples = samples.drop(columns=[1])

print(samples.shape)
x = np.array(samples[0].values.tolist())
print(x.shape)

# Normalizando o pico de todas as amostras
for i in range(x.shape[0]):
    signal = x[i]
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = 0.891 * signal / peak
    x[i] = signal

# Ajustando o tamanho das amostras para 1 segundo (16000 amostras)
x = x.reshape((x.shape[0], x.shape[1], 1))
print(x.shape)

print(x[2])

## Chamando o modelo para predição do dataset ajustado
y_pred_norm = model.predict(x)
y_pred = scaler_y.inverse_transform(y_pred_norm)

y_pred = pd.DataFrame(y_pred)

# Convertendo o nome das colunas para os nomes dos parâmetros do sintetizador
column_names = [
    "frequencia_base",
    "frequency1",
    "beta2",
    "frequency2",
    "beta3",
    "frequency3",
    "beta4",
    "frequency4",
    "beta5",
    "frequency5",
    "beta_carrier",
    "amplitude_carrier",
    "attack",
    "decay",
    "sustain",
    "release",
]
y_pred.columns = column_names

## Escrevendo os parâmetros preditos em arquivo JSON
params_list = y_pred.to_dict(orient="records")
with open("nsynth-pred/_params_pred.json", "w") as f:
    json.dump(params_list, f, indent=4)
