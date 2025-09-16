import json
import math
import numpy as np
import random
import soundfile as sf

from fm_synth2 import FMSynth, SAMPLE_RATE_OUT, SAMPLE_RATE_RENDER

NYQ_RENDER = SAMPLE_RATE_RENDER / 2.0  # 24000 (para 48k)
NYQ_OUT = SAMPLE_RATE_OUT / 2.0  # 8000  (para 16k)

duracao_amostras = 4
tamanho_dataset = 20000
precisao_decimal = 3

max_frequency = 6000
min_frequency = 20

min_effective_beta = 0
max_effective_beta = 4

min_ratio = 1 / 8  # Evitar LFO (que gera mais vibrado do que variação de timbre)
max_ratio = 8  # Evitar aliasing

min_beta = 0
max_beta = 8

min_amplitude = 0.4
max_amplitude = 1

output_dir = "dataset_big2"

SAFE_OUT = 0.75 * NYQ_OUT
SAFE_RENDER = 0.45 * NYQ_RENDER


def uniform_log(min, max):
    return math.exp(random.uniform(math.log(min), math.log(max)))


def is_valid(frequency, beta, amplitude, frequencia_base):
    freq_real = frequency * frequencia_base

    # Garantindo frequência válida
    # (impedindo aliasing)
    if freq_real >= SAFE_RENDER:
        return False

    # (B) beta efetivo dentro do envelope
    effective_beta = beta * amplitude
    if not (min_effective_beta <= effective_beta <= max_effective_beta):
        return False

    # Garantindo que o beta efetivo não estoure um limite razoável
    # (impedindo aliasing)
    f_max_espectral = frequencia_base + (effective_beta + 1) * freq_real
    if f_max_espectral >= SAFE_OUT:
        return False

    return True


def is_valid_carrier(ratio_last, beta_carrier, fc):
    fm_last = ratio_last * fc

    f_max = fc + (beta_carrier + 1.0) * fm_last

    return f_max < SAFE_OUT


def sort_parameters(frequencia_base: float):
    i = 0
    while True:
        i += 1

        if i > 200:
            raise RuntimeError("Não foi possível sortear parâmetros válidos")

        ratio = uniform_log(min_ratio, max_ratio)

        # beta = random.uniform(min_beta, max_beta)
        beta = 0.0 if random.random() < 0.3 else random.uniform(min_beta, max_beta)
        amplitude = random.uniform(min_amplitude, max_amplitude)
        # frequency = round(ratio * frequencia_base, precisao_decimal)
        frequency = ratio

        if not is_valid(frequency, beta, amplitude, frequencia_base):
            continue

        break

    return (beta, amplitude, frequency)


with open(f"{output_dir}/parameters.json", "w") as f:
    f.write("[\n")
    for i in range(tamanho_dataset):
        try:
            print(f"Gerando amostra {i + 1} de {tamanho_dataset}...")
            # Sorteando uma frequencia báse, na faixa audível humana
            frequencia_base = uniform_log(min_frequency, max_frequency)
            frequencia_base = frequencia_base

            # Sorteando os demais parâmetros da síntese
            _, amplitude1, frequency1 = sort_parameters(frequencia_base)
            beta2, amplitude2, frequency2 = sort_parameters(frequencia_base)
            beta3, amplitude3, frequency3 = sort_parameters(frequencia_base)
            beta4, amplitude4, frequency4 = sort_parameters(frequencia_base)
            beta5, amplitude5, frequency5 = sort_parameters(frequencia_base)

            j = 0
            while True:
                j += 1
                if j > 200:
                    raise RuntimeError("Não foi possível sortear parâmetros válidos")

                beta_carrier, amplitude_carrier, _ = sort_parameters(frequencia_base)

                if not is_valid_carrier(frequency5, beta_carrier, frequencia_base):
                    continue

                break

            attack = random.uniform(0.005, 0.5)
            decay = random.uniform(0.02, 0.6)
            sustain = random.uniform(0.1, 0.9)
            release = random.uniform(0.05, 0.8)

            if decay <= 0:
                print(decay)

            # Guardando os parâmetros em JSON
            data = {
                "id": i,
                "frequencia_base": round(frequencia_base, precisao_decimal),
                "amplitude1": round(amplitude1, precisao_decimal),
                "frequency1": round(frequency1, precisao_decimal),
                "beta2": round(beta2, precisao_decimal),
                "amplitude2": round(amplitude2, precisao_decimal),
                "frequency2": round(frequency2, precisao_decimal),
                "beta3": round(beta3, precisao_decimal),
                "amplitude3": round(amplitude3, precisao_decimal),
                "frequency3": round(frequency3, precisao_decimal),
                "beta4": round(beta4, precisao_decimal),
                "amplitude4": round(amplitude4, precisao_decimal),
                "frequency4": round(frequency4, precisao_decimal),
                "beta5": round(beta5, precisao_decimal),
                "amplitude5": round(amplitude5, precisao_decimal),
                "frequency5": round(frequency5, precisao_decimal),
                "beta_carrier": round(beta_carrier, precisao_decimal),
                "amplitude_carrier": round(amplitude_carrier, precisao_decimal),
                "attack": round(attack, precisao_decimal),
                "decay": round(decay, precisao_decimal),
                "sustain": round(sustain, precisao_decimal),
                "release": round(release, precisao_decimal),
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

            # Normalizando o pico
            peak = np.max(np.abs(signal))
            if peak > 0:
                signal = 0.891 * signal / peak

            # Escrevendo o áudio em arquivo
            sf.write(f"{output_dir}/sample_{i}.wav", signal, SAMPLE_RATE_OUT)
        except RuntimeError:
            print("  - Falhou. Tentando novamente...")
            i -= 1
            continue

    f.write("]")
