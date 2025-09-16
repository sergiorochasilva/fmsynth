import math
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly

SAMPLE_RATE_RENDER = 48000  # onde calculamos (alta taxa para reduzir aliasing)
SAMPLE_RATE_OUT = 16000  # taxa alvo do arquivo final
DECIM = SAMPLE_RATE_RENDER // SAMPLE_RATE_OUT  # 48k -> 16k = 3
assert SAMPLE_RATE_RENDER % SAMPLE_RATE_OUT == 0


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

    def synth_alg1(self, audio_seconds: int, frequency_carrier: float):
        """
        Algoritmo de síntese sequencial (cada operador modula o seguinte até o sinal do portador).
        Tudo renderizado a 48 kHz e, no final, reamostrado para 16 kHz.
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

        # ADSR no SR de renderização
        out_render = self._adsr(
            len(carrier),
            audio_seconds,
            self.attack,
            self.decay,
            self.sustain,
            self.release,
            carrier,
        )

        # Anti-alias + downsample: 48 kHz -> 16 kHz (decimação por 3)
        # window=('kaiser', 8.6) ~ bom compromisso de ripple/atenuação
        out_16k = resample_poly(out_render, up=1, down=DECIM, window=("kaiser", 14))
        return out_16k

    def _synth_operator(
        self,
        beta_modulator: float,
        input_modulator: np.ndarray,
        audio_seconds: int,
        amplitude: float,
        frequency: float,
    ):
        # Tempo em SR de renderização
        total_samples = SAMPLE_RATE_RENDER * audio_seconds
        t = np.arange(total_samples) / SAMPLE_RATE_RENDER

        # Fase da portadora
        phase0 = np.random.uniform(0.0, 2 * np.pi)
        phase = 2 * math.pi * frequency * t + phase0

        # Modulação (beta * entrada)
        if input_modulator is not None:
            phase = phase + beta_modulator * input_modulator

        samples = np.sin(phase) * amplitude
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
        sr = SAMPLE_RATE_RENDER
        # Tratamento robusto para tempos = 0
        A = int(round(sr * max(attack, 0.0)))
        D = int(round(sr * max(decay, 0.0)))
        R = int(round(sr * max(release, 0.0)))
        S = max(sample_length - (A + D + R), 0)

        if A > 0:
            a = np.linspace(0.0, 1.0, num=A, endpoint=False)
        else:
            a = np.zeros(0)
        if D > 0:
            d = np.linspace(1.0, sustain, num=D, endpoint=False)
        else:
            d = np.zeros(0)
        s = np.full(S, sustain, dtype=np.float64)
        if R > 0:
            r = np.linspace(sustain, 0.0, num=R, endpoint=False)
        else:
            r = np.zeros(0)

        env = np.concatenate([a, d, s, r])

        # Ajuste de comprimento
        if env.size < sample_length:
            env = np.pad(env, (0, sample_length - env.size), mode="edge")
        elif env.size > sample_length:
            env = env[:sample_length]

        return env * signal


if __name__ == "__main__":
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

    audio_seconds = 4
    frequency_carrier = (
        440  # lembrete: ao reamostrar para 16 kHz, o que passar de ~8 kHz será filtrado
    )
    signal_16k = fm_synth.synth_alg1(audio_seconds, frequency_carrier)

    # Salva em 16 kHz
    sf.write("output2.wav", signal_16k, SAMPLE_RATE_OUT)
