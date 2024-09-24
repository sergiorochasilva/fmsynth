import math
import soundfile as sf
import numpy as np

# import matplotlib
# import matplotlib.pyplot as plt

# matplotlib.use("TkAgg")

SAMPLE_RATE = 22050


def synth_operator(
    beta_modulator: float,
    input_modulator: np.ndarray,
    audio_seconds: int,
    amplitude: float,
    frequency: float,
):
    # Calc total samples
    total_samples = SAMPLE_RATE * audio_seconds

    # Calc duration for each sample
    time_slice = 1.0 / SAMPLE_RATE
    times = np.arange(0, total_samples * time_slice, time_slice)

    # Carrier
    omega_carrier = 2 * math.pi * frequency
    samples = np.ones((total_samples))

    samples *= omega_carrier
    samples *= times

    # Modulating
    if input_modulator is not None:
        samples_modulator = input_modulator * beta_modulator
        samples += samples_modulator

    samples = np.sin(samples)
    samples *= amplitude

    return samples


def adsr(
    audio_seconds: int,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
    signal: np.ndarray,
):
    total_samples = SAMPLE_RATE * audio_seconds

    # Attack
    attack_samples_qtd = int(SAMPLE_RATE * attack)
    attack_coef = 1.0 / attack_samples_qtd
    attack_level_stop = attack_samples_qtd * attack_coef
    attack_samples = np.arange(0, attack_level_stop, attack_coef)

    # Decay
    decay_samples_qtd = int(SAMPLE_RATE * decay)
    decay_coef = (sustain - 1.0) / decay_samples_qtd
    decay_level_stop = attack_level_stop + decay_samples_qtd * decay_coef
    decay_samples = np.arange(attack_level_stop, decay_level_stop, decay_coef)

    # Release
    release_samples_qtd = int(SAMPLE_RATE * release)
    release_coef = (0.0 - sustain) / release_samples_qtd
    release_samples = np.arange(sustain, 0, release_coef)

    # Sustain
    sustain_samples_qtd = int(SAMPLE_RATE * (audio_seconds - attack - decay - release))
    sustain_samples_qtd += total_samples - (
        attack_samples_qtd
        + decay_samples_qtd
        + sustain_samples_qtd
        + release_samples_qtd
    )
    sustain_samples = np.ones((sustain_samples_qtd))
    sustain_samples *= sustain

    # Reultado
    result = np.concatenate(
        (attack_samples, decay_samples, sustain_samples, release_samples), axis=0
    )
    # plt.plot(result)
    # plt.show()
    result *= signal

    return result


modulator1 = synth_operator(0, None, 3, 1, 5 * 220.0)
modulator2 = synth_operator(0, modulator1, 3, 1, 880.0)
modulator3 = synth_operator(0.3, modulator2, 3, 1, 660.0)
carrier = synth_operator(2.5, modulator3, 3, 1, 440.0)
signal = adsr(3, 0.01, 0.2, 0.2, 0.3, carrier)
sf.write("output.wav", signal, SAMPLE_RATE)
