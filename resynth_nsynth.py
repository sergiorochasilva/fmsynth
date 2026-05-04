"""Resynthesize NSynth test audio from predicted FM parameters.

Architecture:
- Loads predicted parameter JSON
- Projects parameters into a feasible FM range
- Uses the FM synth engine to render audio back to waveform form

Data flow:
- Input: NSynth predictions plus `nsynth-test/examples.json`
- Output: synthesized `.wav` files in `nsynth-pred/`
"""

import json
import numpy as np
import os
import soundfile as sf

base_path = "nsynth-test/audio"
file_list = [f for f in os.listdir(base_path) if f.endswith(".wav")]

# limites do dataset
MIN_FC, MAX_FC = 20.0, 6000.0
MIN_RATIO, MAX_RATIO = 1 / 8, 8.0
MIN_BETA, MAX_BETA = 0.0, 8.0
SAFE_OUT = 0.75 * (16000 / 2)  # = 6000.0 Hz

RATIOS_DISCRETOS = [
    1 / 8,
    1 / 6,
    1 / 5,
    1 / 4,
    1 / 3,
    1 / 2,
    2 / 3,
    1,
    3 / 2,
    2,
    3,
    4,
    5,
    6,
    8,
]


def quantize_to_set(x, choices):
    return min(choices, key=lambda c: abs(np.log(x) - np.log(c)))


def project_params_to_feasible(p):
    # 1) clamp básicos
    p["frequencia_base"] = float(np.clip(p["frequencia_base"], MIN_FC, MAX_FC))
    for k in ["frequency1", "frequency2", "frequency3", "frequency4", "frequency5"]:
        p[k] = float(np.clip(p[k], MIN_RATIO, MAX_RATIO))
    for k in ["beta2", "beta3", "beta4", "beta5", "beta_carrier"]:
        p[k] = float(np.clip(p[k], MIN_BETA, MAX_BETA))
    p["amplitude_carrier"] = float(np.clip(p["amplitude_carrier"], 0.4, 1.0))

    # 2) (opcional, ajuda muito) quantizar razões para valores musicais
    for k in ["frequency2", "frequency3", "frequency4", "frequency5"]:
        p[k] = quantize_to_set(max(MIN_RATIO, min(MAX_RATIO, p[k])), RATIOS_DISCRETOS)

    # 3) orçamento espectral conservador no carrier (PM):
    fc = p["frequencia_base"]
    fm3 = p["frequency3"] * fc
    fm4 = p["frequency4"] * fc
    fm5 = p["frequency5"] * fc

    # pela regra de Carson (~ (β+1)*fm) e somando os três ramos
    B_chain = (p["beta_carrier"] + 1.0) * fm3
    B_p4 = (p["beta4"] + 1.0) * fm4
    B_p5 = (p["beta5"] + 1.0) * fm5
    f_max_est = fc + B_chain + B_p4 + B_p5

    if f_max_est > SAFE_OUT:
        scale = max((SAFE_OUT - fc) / (B_chain + B_p4 + B_p5 + 1e-9), 0.0)
        # reduzir β de forma proporcional (mantendo relações)
        p["beta_carrier"] *= scale
        p["beta4"] *= scale
        p["beta5"] *= scale

    return p


## Recarregando os parâmetros preditos do arquivo JSON
with open("nsynth-pred/_params_pred.json", "r") as f:
    params_list = json.load(f)

## Carregando as informações do próprio dataset NSynth
with open("nsynth-test/examples.json", "r") as f:
    params_nsynth = json.load(f)

## Chamando o sintetizador para cada parâmetro predito pelo modelo
from fm_synth2 import FMSynth

# Iterando sobre todas as predições
for i in range(len(params_list)):
    # Extraindo os parâmetros preditos
    params = params_list[i]

    params = project_params_to_feasible(params)

    # Recuperando o pitch original do dataset NSynth
    pitch = params_nsynth[file_list[i].replace(".wav", "")]["pitch"]
    frequencia_base = 440.0 * 2 * 2 ** ((pitch - 69) / 12.0)
    params["frequencia_base"] = frequencia_base

    # # Criando uma instância do sintetizador FM
    fm_synth = FMSynth(
        amplitude1=1.0,
        frequency1=params["frequency1"],
        beta2=params["beta2"],
        amplitude2=1.0,
        frequency2=params["frequency2"],
        beta3=params["beta3"],
        amplitude3=1.0,
        frequency3=params["frequency3"],
        beta4=params["beta4"],
        amplitude4=1.0,
        frequency4=params["frequency4"],
        beta5=params["beta5"],
        amplitude5=1.0,
        frequency5=params["frequency5"],
        beta_carrier=params["beta_carrier"],
        amplitude_carrier=params["amplitude_carrier"],
        attack=params["attack"],
        decay=params["decay"],
        sustain=params["sustain"],
        release=params["release"],
    )
    signal = fm_synth.synth_alg_series3_parallel2(4, params["frequencia_base"])

    # Normalizando o pico
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = 0.891 * signal / peak

    # Salvando o áudio em arquivo
    sf.write(f"nsynth-pred/{file_list[i]}", signal, 16000)
