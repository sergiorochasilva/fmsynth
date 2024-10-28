import math
import soundfile as sf
import numpy as np

# import matplotlib
# import matplotlib.pyplot as plt

# matplotlib.use("TkAgg")

SAMPLE_RATE = 22050


class FMSynth:

    def __init__(
        self,
        amplitude1: float,
        frequency1: float,
        beta2: float,
        amplitude2: float,
        frequency2: float,
        beta3: float,
        amplitude3: float,
        frequency3: float,
        beta4: float,
        amplitude4: float,
        frequency4: float,
        beta5: float,
        amplitude5: float,
        frequency5: float,
        beta_carrier: float,
        amplitude_carrier: float,
        attack: float,
        decay: float,
        sustain: float,
        release: float,
    ) -> None:
        self.amplitude1 = amplitude1
        self.frequency1 = frequency1
        self.beta2 = beta2
        self.amplitude2 = amplitude2
        self.frequency2 = frequency2
        self.beta3 = beta3
        self.amplitude3 = amplitude3
        self.frequency3 = frequency3
        self.beta4 = beta4
        self.amplitude4 = amplitude4
        self.frequency4 = frequency4
        self.beta5 = beta5
        self.amplitude5 = amplitude5
        self.frequency5 = frequency5
        self.beta_carrier = beta_carrier
        self.amplitude_carrier = amplitude_carrier
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release

    def synth_alg1(
        self,
        audio_seconds: int,
        frequency_carrier: float,
    ):
        """
        Algortimo de síntese sequencial (cada operador modula o seguinte até o sinal do portador).
        """

        modulator1 = self._synth_operator(
            0, None, audio_seconds, self.amplitude1, self.frequency1 * frequency_carrier
        )
        modulator2 = self._synth_operator(
            self.beta2,
            modulator1,
            audio_seconds,
            self.amplitude2,
            self.frequency2 * frequency_carrier,
        )
        modulator3 = self._synth_operator(
            self.beta3,
            modulator2,
            audio_seconds,
            self.amplitude3,
            self.frequency3 * frequency_carrier,
        )
        modulator4 = self._synth_operator(
            self.beta4,
            modulator3,
            audio_seconds,
            self.amplitude4,
            self.frequency4 * frequency_carrier,
        )
        modulator5 = self._synth_operator(
            self.beta5,
            modulator4,
            audio_seconds,
            self.amplitude5,
            self.frequency5 * frequency_carrier,
        )
        carrier = self._synth_operator(
            self.beta_carrier,
            modulator5,
            audio_seconds,
            self.amplitude_carrier,
            frequency_carrier,
        )

        return self._adsr(
            len(carrier),
            audio_seconds,
            self.attack,
            self.decay,
            self.sustain,
            self.release,
            carrier,
        )

    def _synth_operator(
        self,
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

    def _adsr(
        self,
        sample_length: int,
        audio_seconds: float,
        attack: float,
        decay: float,
        sustain: float,
        release: float,
        signal: np.ndarray,
    ):
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
        sustain_samples_qtd = sample_length - (
            attack_samples.shape[0] + decay_samples.shape[0] + release_samples.shape[0]
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


if __name__ == "__main__":
    # fm_synth = FMSynth(
    #     amplitude1=1.0,
    #     frequency1=2.2,
    #     beta2=0.0,
    #     amplitude2=1,
    #     frequency2=2.0,
    #     beta3=0.3,
    #     amplitude3=1,
    #     frequency3=1.5,
    #     beta4=0.3,
    #     amplitude4=1,
    #     frequency4=1.5,
    #     beta5=0.3,
    #     amplitude5=1,
    #     frequency5=1.5,
    #     beta_carrier=2.5,
    #     amplitude_carrier=0.8,
    #     attack=0.01,
    #     decay=0.2,
    #     sustain=0.2,
    #     release=0.3,
    # )
    data = {
        "amplitude1": 0.396,
        "frequency1": 0.07,
        "beta2": 0.273,
        "amplitude2": 0.672,
        "frequency2": 0.133,
        "beta3": 0.278,
        "amplitude3": 0.692,
        "frequency3": 0.135,
        "beta4": 0.293,
        "amplitude4": 0.083,
        "frequency4": 0.598,
        "beta5": 0.945,
        "amplitude5": 0.861,
        "frequency5": 0.13,
        "beta_carrier": 0.742,
        "amplitude_carrier": 0.93,
        "attack": 0.72,
        "decay": 0.865,
        "sustain": 0.545,
        "release": 0.453,
    }
    fm_synth = FMSynth(**data)
    signal = fm_synth.synth_alg1(4, 18618.554)
    sf.write("output.wav", signal, SAMPLE_RATE)
