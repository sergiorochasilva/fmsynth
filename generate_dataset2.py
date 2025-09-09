import json
import math
import random
import soundfile as sf

from fm_synth2 import FMSynth, SAMPLE_RATE_OUT, SAMPLE_RATE_RENDER

duracao_amostras = 4
tamanho_dataset = 20000
precisao_decimal = 3

max_frequency = 6000
min_frequency = 20

min_effective_beta = 0
max_effective_beta = 6

min_ratio = 1 / 8  # Evitar LFO (que gera mais vibrado do que variação de timbre)
max_ratio = 8  # Evitar aliasing

min_beta = 0
max_beta = 8

min_amplitude = 0.4
max_amplitude = 1

output_dir = "dataset_big2"


def uniform_log(min, max):
    return math.exp(random.uniform(math.log(min), math.log(max)))


def sort_parameters(frequencia_base: float):
    while True:
        ratio = uniform_log(min_ratio, max_ratio)

        beta = random.uniform(min_beta, max_beta)
        amplitude = round(
            random.uniform(min_amplitude, max_amplitude), precisao_decimal
        )
        frequency = round(ratio * frequencia_base, precisao_decimal)

        # Garantindo frequência válida
        # (impedindo aliasing)
        if frequency >= SAMPLE_RATE_RENDER / 2 * 0.45:
            continue

        # Garantindo que o beta efetivo não estoure um limite razoável
        # (impedindo aliasing)
        effective_beta = beta * amplitude
        f_max_espectral = frequencia_base + (effective_beta + 1) * frequency
        if f_max_espectral < 0.9 * SAMPLE_RATE_OUT / 2:
            break

    return (
        round(beta, precisao_decimal),
        round(amplitude, precisao_decimal),
        round(frequency, precisao_decimal),
    )


with open(f"{output_dir}/parameters.json", "w") as f:
    f.write("[\n")
    for i in range(tamanho_dataset):
        print(f"Gerando amostra {i + 1} de {tamanho_dataset}...")
        # Sorteando uma frequencia báse, na faixa audível humana
        frequencia_base = uniform_log(min_frequency, max_frequency)
        frequencia_base = round(frequencia_base, precisao_decimal)

        # Sorteando os demais parâmetros da síntese
        _, amplitude1, frequency1 = sort_parameters(frequencia_base)
        beta2, amplitude2, frequency2 = sort_parameters(frequencia_base)
        beta3, amplitude3, frequency3 = sort_parameters(frequencia_base)
        beta4, amplitude4, frequency4 = sort_parameters(frequencia_base)
        beta5, amplitude5, frequency5 = sort_parameters(frequencia_base)

        beta_carrier, amplitude_carrier, _ = sort_parameters(frequencia_base)

        attack = random.uniform(0.005, 0.5)
        decay = random.uniform(0.02, 0.6)
        sustain = random.uniform(0.1, 0.9)
        release = random.uniform(0.05, 0.8)

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
        sf.write(f"{output_dir}/sample_{i}.wav", signal, SAMPLE_RATE_OUT)

    f.write("]")
