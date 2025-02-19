import IPython.display as ipd
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import soundfile as sf
import pandas as pd

from fm_synth import FMSynth, SAMPLE_RATE

duracao_amostras = 1
tamanho_dataset = 5000
precisao_decimal = 3

with open("dataset/parameters.json", "w") as f:
    f.write("[\n")
    for i in range(tamanho_dataset):
        # Sorteando uma frequencia báse, na faixa audível humana
        frequencia_base = random.random() * 20000 + 20
        frequencia_base = round(frequencia_base, precisao_decimal)

        # Sorteando os demais parâmetros da síntese
        amplitude1 = round((random.random() * 0.999 + 0.001), precisao_decimal)
        frequency1 = round((random.random() * 0.999 + 0.001), precisao_decimal)
        beta2 = round((random.random() * 0.999 + 0.001), precisao_decimal)
        amplitude2 = round((random.random() * 0.999 + 0.001), precisao_decimal)
        frequency2 = round((random.random() * 0.999 + 0.001), precisao_decimal)
        beta3 = round((random.random() * 0.999 + 0.001), precisao_decimal)
        amplitude3 = round((random.random() * 0.999 + 0.001), precisao_decimal)
        frequency3 = round((random.random() * 0.999 + 0.001), precisao_decimal)
        beta4 = round((random.random() * 0.999 + 0.001), precisao_decimal)
        amplitude4 = round((random.random() * 0.999 + 0.001), precisao_decimal)
        frequency4 = round((random.random() * 0.999 + 0.001), precisao_decimal)
        beta5 = round((random.random() * 0.999 + 0.001), precisao_decimal)
        amplitude5 = round((random.random() * 0.999 + 0.001), precisao_decimal)
        frequency5 = round((random.random() * 0.999 + 0.001), precisao_decimal)
        beta_carrier = round((random.random() * 0.999 + 0.001), precisao_decimal)
        amplitude_carrier = round((random.random() * 0.999 + 0.001), precisao_decimal)
        attack = round((random.random() * 0.2 + 0.001), precisao_decimal)
        decay = round((random.random() * 0.1 + 0.001), precisao_decimal)
        sustain = round((random.random() * 0.5 + 0.001), precisao_decimal)
        release = round((random.random() * 0.2 + 0.001), precisao_decimal)
        if decay <= 0:
            print(decay)

        # Guardando os parâmetros em JSON
        data = {
            "id": i,
            "frequencia_base": frequencia_base,
            "amplitude1": amplitude1,
            "frequency1": frequency1,
            "beta2": beta2,
            "amplitude2": amplitude2,
            "frequency2": frequency2,
            "beta3": beta3,
            "amplitude3": amplitude3,
            "frequency3": frequency3,
            "beta4": beta4,
            "amplitude4": amplitude4,
            "frequency4": frequency4,
            "beta5": beta5,
            "amplitude5": amplitude5,
            "frequency5": frequency5,
            "beta_carrier": beta_carrier,
            "amplitude_carrier": amplitude_carrier,
            "attack": attack,
            "decay": decay,
            "sustain": sustain,
            "release": release,
        }

        # Codificando os parâmetros em json
        data = json.dumps(data)

        # Imprimindo na saída
        f.write(data)
        if i < tamanho_dataset - 1:
            f.write(",\n")
        else:
            f.write("\n")

        # Sintetizando o sinal
        fm_synth = FMSynth(
            amplitude1=amplitude1,
            frequency1=frequency1,
            beta2=beta2,
            amplitude2=amplitude2,
            frequency2=frequency2,
            beta3=beta3,
            amplitude3=amplitude3,
            frequency3=frequency3,
            beta4=beta4,
            amplitude4=amplitude4,
            frequency4=frequency4,
            beta5=beta5,
            amplitude5=amplitude5,
            frequency5=frequency5,
            beta_carrier=beta_carrier,
            amplitude_carrier=amplitude_carrier,
            attack=attack,
            decay=decay,
            sustain=sustain,
            release=release,
        )
        signal = fm_synth.synth_alg1(duracao_amostras, frequencia_base)

        # Escrevendo o áudio em arquivo
        sf.write(f"dataset/sample_{i}.wav", signal, SAMPLE_RATE)

    f.write("]")
