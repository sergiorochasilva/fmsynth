"""Second-generation FM synth engine with improved rendering and decimation.

Architecture:
- FM/PM style synthesis engine with envelopes and anti-aliasing support
- Renders at a higher internal sample rate and downsamples to output

Data flow:
- Input: synthesis parameters from the prediction/resynthesis scripts
- Output: rendered waveform arrays or `.wav` files
"""

import math
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from scipy.signal import butter, lfilter

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
            0,
            None,
            audio_seconds,
            self.amplitude1,
            self.frequency1 * frequency_carrier,
            None,
        )
        modulator2 = self._synth_operator(
            self.beta2,
            modulator1,
            audio_seconds,
            self.amplitude2,
            self.frequency2 * frequency_carrier,
            self.frequency1 * frequency_carrier,
        )
        modulator3 = self._synth_operator(
            self.beta3,
            modulator2,
            audio_seconds,
            self.amplitude3,
            self.frequency3 * frequency_carrier,
            self.frequency2 * frequency_carrier,
        )
        modulator4 = self._synth_operator(
            self.beta4,
            modulator3,
            audio_seconds,
            self.amplitude4,
            self.frequency4 * frequency_carrier,
            self.frequency3 * frequency_carrier,
        )
        modulator5 = self._synth_operator(
            self.beta5,
            modulator4,
            audio_seconds,
            self.amplitude5,
            self.frequency5 * frequency_carrier,
            self.frequency4 * frequency_carrier,
        )
        carrier = self._synth_operator(
            self.beta_carrier,
            modulator5,
            audio_seconds,
            self.amplitude_carrier,
            frequency_carrier,
            self.frequency5 * frequency_carrier,
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
        out_16k = out_16k / (np.max(np.abs(out_16k)) + 1e-12)
        return out_16k

    def _tilt(self, x, sr=SAMPLE_RATE_RENDER, fc=5500.0):
        # 1 pólo baixo (Butter 1ª ordem) para “amaciar” a borda
        b, a = butter(1, fc / (sr / 2), btype="low", analog=False)
        return lfilter(b, a, x)

    def _index_env(self, N, sr, a=0.012, d=0.18, s=0.45, r=0.20):
        A = int(round(sr * a))
        D = int(round(sr * d))
        R = int(round(sr * r))
        S = max(N - (A + D + R), 0)

        # exponenciais simples (soam mais naturais que linear)
        def seg(n0, n1, y0, y1):
            n = n1 - n0
            if n <= 0:
                return np.zeros(0)
            x = np.linspace(0, 1, n, endpoint=False)
            # curva exponencial suave
            return y0 * ((y1 / y0) ** x)

        env = np.concatenate(
            [
                seg(0, A, 1e-3, 1.0),  # ataque rápido
                seg(0, D, 1.0, s),  # decay para sustain
                np.full(S, s, dtype=np.float64),
                seg(0, R, s, 1e-3),
            ]
        )
        return env

    def _prepare_mod_for_beta(self, m: np.ndarray, audio_seconds: float):
        """Remove DC, normaliza por pico (β clássico) e aplica envelope no índice."""
        m0 = m - np.mean(m)
        peak = np.max(np.abs(m0)) + 1e-12
        m_norm = m0 / peak
        # idx_env = self._adsr(
        #     len(m_norm),
        #     audio_seconds,
        #     self.attack,
        #     self.decay,
        #     self.sustain,
        #     self.release,
        #     np.ones_like(m_norm),
        # )
        idx_env = self._index_env(len(m_norm), SAMPLE_RATE_RENDER)
        m = m_norm * idx_env
        m -= np.mean(m)
        return m

    def synth_alg_series3_parallel2(self, audio_seconds: int, frequency_carrier: float):
        """
        Topologia: (op1 -> op2 -> op3)  ||  op4  ||  op5  --> carrier
        """
        N = int(SAMPLE_RATE_RENDER * audio_seconds)
        t = np.arange(N) / SAMPLE_RATE_RENDER

        # --- Série: op1 -> op2 -> op3
        m1 = self._synth_operator(
            0.0,
            None,
            audio_seconds,
            self.amplitude1,
            self.frequency1 * frequency_carrier,
            None,
        )
        m2 = self._synth_operator(
            self.beta2,
            m1,
            audio_seconds,
            self.amplitude2,
            self.frequency2 * frequency_carrier,
            self.frequency1 * frequency_carrier,
        )
        m3 = self._synth_operator(
            self.beta3,
            m2,
            audio_seconds,
            self.amplitude3,
            self.frequency3 * frequency_carrier,
            self.frequency2 * frequency_carrier,
        )

        # --- Paralelo: op4 e op5 (isolados, sem modulador de entrada)
        p4 = self._synth_operator(
            0.0,
            None,
            audio_seconds,
            self.amplitude4,
            self.frequency4 * frequency_carrier,
            None,
        )
        p5 = self._synth_operator(
            0.0,
            None,
            audio_seconds,
            self.amplitude5,
            self.frequency5 * frequency_carrier,
            None,
        )

        # --- Carrier: soma das três modulações (m3, p4, p5)
        phase = 2 * np.pi * frequency_carrier * t
        # phase += np.random.uniform(0.0, 2 * np.pi)

        # 1) cadeia (usa beta_carrier e f_m = f3)
        m3_prep = self._prepare_mod_for_beta(m3, audio_seconds)
        delta_f_chain = self.beta_carrier * (self.frequency3 * frequency_carrier)
        # phase += 2 * np.pi * (delta_f_chain / SAMPLE_RATE_RENDER) * np.cumsum(m3_prep)
        phase += self.beta_carrier * m3_prep

        # 2) paralelo op4 (usa beta4 e f_m = f4)
        p4_prep = self._prepare_mod_for_beta(p4, audio_seconds)
        delta_f_p4 = self.beta4 * (self.frequency4 * frequency_carrier)
        # phase += 2 * np.pi * (delta_f_p4 / SAMPLE_RATE_RENDER) * np.cumsum(p4_prep)
        phase += self.beta4 * p4_prep

        # 3) paralelo op5 (usa beta5 e f_m = f5)
        p5_prep = self._prepare_mod_for_beta(p5, audio_seconds)
        delta_f_p5 = self.beta5 * (self.frequency5 * frequency_carrier)
        # phase += 2 * np.pi * (delta_f_p5 / SAMPLE_RATE_RENDER) * np.cumsum(p5_prep)
        phase += self.beta5 * p5_prep

        # sinal da portadora
        carrier = np.sin(phase) * self.amplitude_carrier

        # ADSR na saída
        out_render = self._adsr(
            len(carrier),
            audio_seconds,
            self.attack,
            self.decay,
            self.sustain,
            self.release,
            carrier,
        )

        # Downsample + normalização de segurança
        out_render = self._tilt(out_render, SAMPLE_RATE_RENDER, fc=5500.0)
        out_16k = resample_poly(out_render, up=1, down=DECIM, window=("kaiser", 14))
        out_16k = out_16k / (np.max(np.abs(out_16k)) + 1e-12)
        return out_16k

    def _synth_operator(
        self,
        beta_modulator: float,
        input_modulator: np.ndarray | None,
        audio_seconds: int,
        amplitude: float,
        frequency: float,
        frequency_modulator: float | None,
    ):
        # Tempo em SR de renderização
        total_samples = SAMPLE_RATE_RENDER * audio_seconds
        t = np.arange(total_samples) / SAMPLE_RATE_RENDER

        # Fase da portadora
        phase = 2 * np.pi * frequency * t
        # phase0 = np.random.uniform(0.0, 2 * np.pi)
        # phase = phase + phase0

        # Modulação (beta * entrada)
        if input_modulator is not None:
            # Verificando se frequency_modulator foi informado
            if frequency_modulator is None:
                raise ValueError(
                    "frequency_modulator deve ser informado quando há modulador."
                )

            # Calculando o ganho de modulação em frequência (delta_f)
            delta_f = beta_modulator * frequency_modulator

            # Centraliza o modulador em zero (para evitar drift na fase)
            m0 = input_modulator - np.mean(input_modulator)

            # normaliza o modulador para que β seja estável (independente da amplitude real)
            peak = np.max(np.abs(m0)) + 1e-12
            m_norm = m0 / peak

            # phase = phase + beta_modulator * input_modulator
            idx_env = self._adsr(
                len(m_norm),
                audio_seconds,
                self.attack,
                self.decay,
                self.sustain,
                self.release,
                np.ones_like(m_norm),
            )

            # # Modulador final
            m = m_norm * idx_env

            # Integração discreta para obter a fase a partir de delta_f
            # phase += 2 * np.pi * (delta_f / SAMPLE_RATE_RENDER) * np.cumsum(m)
            phase += beta_modulator * m

        return np.sin(phase) * amplitude

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
    # data = {
    #     "amplitude1": 0.396,
    #     "frequency1": 0.07,
    #     "beta2": 0.273,
    #     "amplitude2": 0.672,
    #     "frequency2": 0.133,
    #     "beta3": 0.278,
    #     "amplitude3": 0.692,
    #     "frequency3": 0.135,
    #     "beta4": 0.293,
    #     "amplitude4": 0.083,
    #     "frequency4": 0.598,
    #     "beta5": 0.945,
    #     "amplitude5": 0.861,
    #     "frequency5": 0.13,
    #     "beta_carrier": 0.742,
    #     "amplitude_carrier": 0.93,
    #     "attack": 0.72,
    #     "decay": 0.865,
    #     "sustain": 0.545,
    #     "release": 0.453,
    # }
    data = {
        "amplitude1": 1.0,
        "frequency1": 440.0,
        "beta2": 0.0,
        "amplitude2": 1.0,
        "frequency2": 0.0,
        "beta3": 0.0,
        "amplitude3": 1.0,
        "frequency3": 0.0,
        "beta4": 0.0,
        "amplitude4": 1.0,
        "frequency4": 0.0,
        "beta5": 0.0,
        "amplitude5": 1.0,
        "frequency5": 0.0,
        "beta_carrier": 0.0,
        "amplitude_carrier": 1.0,
        "attack": 0.0,
        "decay": 0.0,
        "sustain": 1.0,
        "release": 0.0,
    }
    fm_synth = FMSynth(**data)

    audio_seconds = 4
    frequency_carrier = (
        440  # lembrete: ao reamostrar para 16 kHz, o que passar de ~8 kHz será filtrado
    )
    signal_16k = fm_synth.synth_alg_series3_parallel2(audio_seconds, frequency_carrier)

    # Salva em 16 kHz
    sf.write("output2.wav", signal_16k, SAMPLE_RATE_OUT)
