import json
import numpy as np
import os
import soundfile as sf

base_path = "nsynth-test/audio"
file_list = [f for f in os.listdir(base_path) if f.endswith(".wav")]

## Recarregando os parâmetros preditos do arquivo JSON
with open("nsynth-pred/params_pred.json", "r") as f:
    params_list = json.load(f)

## Chamando o sintetizador para cada parâmetro predito pelo modelo
from fm_synth2 import FMSynth

# Iterando sobre todas as predições
for i in range(len(params_list)):
    # Extraindo os parâmetros preditos
    params = params_list[i]

    # # Criando uma instância do sintetizador FM
    fm_synth = FMSynth(
        amplitude1=params["amplitude1"],
        frequency1=params["frequency1"],
        beta2=params["beta2"],
        amplitude2=params["amplitude2"],
        frequency2=params["frequency2"],
        beta3=params["beta3"],
        amplitude3=params["amplitude3"],
        frequency3=params["frequency3"],
        beta4=params["beta4"],
        amplitude4=params["amplitude4"],
        frequency4=params["frequency4"],
        beta5=params["beta5"],
        amplitude5=params["amplitude5"],
        frequency5=params["frequency5"],
        beta_carrier=params["beta_carrier"],
        amplitude_carrier=params["amplitude_carrier"],
        attack=params["attack"],
        decay=params["decay"],
        sustain=params["sustain"],
        release=params["release"],
    )
    signal = fm_synth.synth_alg1(4, params["frequencia_base"])

    # Normalizando o pico
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = 0.891 * signal / peak

    # Salvando o áudio em arquivo
    sf.write(f"nsynth-pred/{file_list[i]}", signal, 16000)
