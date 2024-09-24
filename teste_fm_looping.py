import math
import soundfile as sf

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

    # Calc each sample
    samples = []
    for i in range(total_samples):
        time = i * time_slice
        omega_carrier = 2 * math.pi * frequency
        omega_modulator = 2 * math.pi * modulator_frequency
        sample = amplitude * math.sin(
            omega_carrier * time + beta_modulator * math.sin(omega_modulator * time)
        )
        samples.append(sample)

    # Converting to audio
    sf.write("output.wav", samples, SAMPLE_RATE)


synth_fm(3, 1.0, 220.0, 1, 440.0)
