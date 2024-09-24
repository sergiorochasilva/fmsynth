import math
import soundfile as sf
import numpy as np

SAMPLE_RATE = 22050


def synth_fm(
    audio_seconds: int,
    amplitude: float,
    frequency: float,
    beta_modulator: float,
    modulator_frequency: float,
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

    # Modulator
    omega_modulator = 2 * math.pi * modulator_frequency
    samples_modulator = np.ones((total_samples))

    samples_modulator *= omega_modulator
    samples_modulator *= times

    samples_modulator = np.sin(samples_modulator)
    samples_modulator *= beta_modulator

    # Modulating
    samples += samples_modulator

    samples = np.sin(samples)
    samples *= amplitude

    sf.write("output.wav", samples, SAMPLE_RATE)


synth_fm(3, 0.5, 440.0, 10, 880.0)
