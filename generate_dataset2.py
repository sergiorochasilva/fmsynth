"""Generate the second FM-synthesis dataset used in the experiments.

Architecture:
- Samples FM parameters with a mix of musical and random distributions
- Renders audio with `fm_synth2`

Data flow:
- Input: configuration constants in the script
- Output: dataset files plus a JSON summary
"""

import json
import math
import numpy as np
import random
import soundfile as sf

from fm_synth2 import FMSynth, SAMPLE_RATE_OUT, SAMPLE_RATE_RENDER

NYQ_RENDER = SAMPLE_RATE_RENDER / 2.0  # 24000 (para 48k)
NYQ_OUT = SAMPLE_RATE_OUT / 2.0  # 8000  (para 16k)

duracao_amostras = 4
tamanho_dataset = 5000
precisao_decimal = 3

max_frequency = 6000
min_frequency = 20

min_effective_beta = 0
max_effective_beta = 5

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

    # Beta efetivo dentro do envelope
    # effective_beta = beta * amplitude
    effective_beta = beta
    if not (min_effective_beta <= effective_beta <= max_effective_beta):
        return False

    # Garantindo que o beta efetivo não estoure um limite razoável
    # (impedindo aliasing)
    f_max_espectral = frequencia_base + (effective_beta + 1) * freq_real
    if f_max_espectral >= SAFE_OUT:
        return False

    return True


def is_valid_carrier(r3, r4, r5, beta_carrier, beta4, beta5, fc):
    fm3, fm4, fm5 = r3 * fc, r4 * fc, r5 * fc
    f_max = fc + (beta_carrier + 1) * fm3 + (beta4 + 1) * fm4 + (beta5 + 1) * fm5
    return f_max < SAFE_OUT


# def is_valid_carrier(ratio_last, beta_carrier, fc):
#     fm_last = ratio_last * fc

#     f_max = fc + (beta_carrier + 1.0) * fm_last

#     return f_max < SAFE_OUT


choices = [
    (0.0, 0.5, 0.2),  # 20% chance: suave/quase puro
    (0.5, 3.0, 0.6),  # 60% chance: zona musical
    (3.0, 8.0, 0.2),  # 20% chance: agressivo/metálico
]


def sample_beta():
    x = random.random()
    if x < 0.2:
        return random.uniform(0.0, 0.5)
    elif x < 0.8:
        return random.uniform(0.5, 3.0)
    else:
        return random.uniform(3.0, 8.0)


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


def sample_ratio(min_ratio=1 / 8, max_ratio=8):
    """Mistura:
    - 70% de chance: ratios discretos musicais
    - 30% de chance: contínuo log-uniforme entre [1/8, 8]
    """
    if random.random() < 0.9:
        return random.choice(RATIOS_DISCRETOS)
    else:
        return uniform_log(min_ratio, max_ratio)


def keyscale_beta(
    beta_base: float,
    fc: float,
    fc_ref: float = 440.0,
    strength: float = 0.5,  # 0=no scaling, 1=total
    beta_min: float = 0.0,
    beta_max: float = 8.0,
) -> float:
    """
    Aplica key-scaling em β:
    β'(fc) = ((β_base + 1) * (fc_ref/fc)**strength) - 1
    Clamp em [beta_min, beta_max].
    """
    fc_eff = max(fc, 60.0)  # evita exagero abaixo de ~A1
    factor = (fc_ref / fc_eff) ** strength
    beta_scaled = (beta_base + 1.0) * factor - 1.0
    # clamp
    if beta_scaled < beta_min:
        beta_scaled = beta_min
    if beta_scaled > beta_max:
        beta_scaled = beta_max
    return beta_scaled


def sort_parameters(frequencia_base: float):
    i = 0
    while True:
        i += 1

        if i > 200:
            raise RuntimeError("Não foi possível sortear parâmetros válidos")

        # ratio = uniform_log(min_ratio, max_ratio)
        ratio = sample_ratio(min_ratio, max_ratio)

        beta_base = sample_beta()
        beta = keyscale_beta(
            beta_base,
            fc=frequencia_base,
            fc_ref=440.0,
            strength=0.55,  # ajuste fino aqui
            beta_min=min_beta,
            beta_max=max_beta,
        )

        # beta = random.uniform(min_beta, max_beta)
        # beta = 0.0 if random.random() < 0.3 else random.uniform(min_beta, max_beta)
        amplitude = random.uniform(min_amplitude, max_amplitude)
        # frequency = round(ratio * frequencia_base, precisao_decimal)
        frequency = ratio

        if not is_valid(frequency, beta, amplitude, frequencia_base):
            continue

        break

    return (beta, amplitude, frequency)


with open(f"{output_dir}/parameters.json", "w") as f:
    f.write("[\n")

    i = 0
    while True:
        if i >= tamanho_dataset:
            break
        i += 1

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

            active = random.choice([1, 2, 3])
            idxs = random.sample([2, 3, 4, 5], k=active)
            for j in [2, 3, 4, 5]:
                if j not in idxs:
                    # zere este beta (sintetizador vai ignorar esta modulação)
                    if j == 2:
                        beta2 = 0.0
                    if j == 3:
                        beta3 = 0.0
                    if j == 4:
                        beta4 = 0.0
                    if j == 5:
                        beta5 = 0.0

            j = 0
            while True:
                j += 1
                if j > 200:
                    raise RuntimeError("Não foi possível sortear parâmetros válidos")

                amplitude_carrier = random.uniform(min_amplitude, max_amplitude)
                beta_carrier_base = sample_beta()
                beta_carrier = keyscale_beta(
                    beta_carrier_base,
                    fc=frequencia_base,
                    fc_ref=440.0,
                    strength=0.55,
                    beta_min=min_beta,
                    beta_max=max_beta,
                )

                # if not is_valid_carrier(frequency5, beta_carrier, frequencia_base):
                if not is_valid_carrier(
                    frequency3,
                    frequency4,
                    frequency5,
                    beta_carrier,
                    beta4,
                    beta5,
                    frequencia_base,
                ):
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
                "frequency1": round(frequency1, precisao_decimal),
                "beta2": round(beta2, precisao_decimal),
                "frequency2": round(frequency2, precisao_decimal),
                "beta3": round(beta3, precisao_decimal),
                "frequency3": round(frequency3, precisao_decimal),
                "beta4": round(beta4, precisao_decimal),
                "frequency4": round(frequency4, precisao_decimal),
                "beta5": round(beta5, precisao_decimal),
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
            if i < tamanho_dataset:
                f.write(",\n")
            else:
                f.write("\n")

            # Sintetizando o sinal
            fm_synth = FMSynth(
                amplitude1=1.0,
                frequency1=frequency1,
                beta2=beta2,
                amplitude2=1.0,
                frequency2=frequency2,
                beta3=beta3,
                amplitude3=1.0,
                frequency3=frequency3,
                beta4=beta4,
                amplitude4=1.0,
                frequency4=frequency4,
                beta5=beta5,
                amplitude5=1.0,
                frequency5=frequency5,
                beta_carrier=beta_carrier,
                amplitude_carrier=amplitude_carrier,
                attack=attack,
                decay=decay,
                sustain=sustain,
                release=release,
            )
            signal = fm_synth.synth_alg_series3_parallel2(
                duracao_amostras, frequencia_base
            )

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
