"""Third-generation FM synth engine used for `dataset_big3`, `big4`, and `big5`.

Architecture:
- Six-operator FM/PM-style synth with per-operator envelopes, detune, feedback, and LFO
- Renders at a high internal rate and optionally decimates to 16 kHz

Data flow:
- Input: structured synthesis parameters from dataset generation or prediction scripts
- Output: waveform arrays and `.wav` files compatible with the training datasets
"""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# Render em SR alto para reduzir aliasing; decimamos para 16 kHz se desejado.
SAMPLE_RATE_RENDER = 96000
SAMPLE_RATE_OUT = 16000
DECIM = SAMPLE_RATE_RENDER // SAMPLE_RATE_OUT
assert SAMPLE_RATE_RENDER % SAMPLE_RATE_OUT == 0


@dataclass
class Envelope:
    attack: float
    decay: float
    sustain: float
    release: float
    curve_attack: str = "exp"
    curve_decay: str = "exp"
    curve_release: str = "exp"


class FMSynth3:
    """
    Sintetizador FM simples e mais realista, mantendo baixa ambiguidade de parâmetros.

    Engine interna: PM (Phase Modulation), mas com parâmetros em termos de FM.

    Mapeamento FM -> PM (para moduladores senoidais):
      - Índice de modulação I = Δf / f_m
      - PM usa: phase += I * m(t)
      - FM usa: phase += 2π * (Δf / sr) * cumsum(m(t))

    Com m(t) = seno, FM e PM têm espectro equivalente quando I é o mesmo.
    """

    def __init__(
        self,
        # Ratios (em relação à frequência da nota/base)
        ratio1: float,
        ratio2: float,
        ratio3: float,
        ratio4: float,
        ratio5: float,
        ratio_carrier: float,
        # Detune por operador (cents)
        detune1: float = 0.0,
        detune2: float = 0.0,
        detune3: float = 0.0,
        detune4: float = 0.0,
        detune5: float = 0.0,
        detune_carrier: float = 0.0,
        # Índices de modulação por conexão (I = Δf / f_m)
        index_12: float = 0.0,
        index_23: float = 0.0,
        index_3c: float = 0.0,
        index_4c: float = 0.0,
        index_5c: float = 0.0,
        # Envelope global para moduladores e para a portadora (ADS R)
        env_mod: Envelope | None = None,
        env_car: Envelope | None = None,
        # Envelope por operador (se None, usa o envelope global correspondente)
        env1: Envelope | None = None,
        env2: Envelope | None = None,
        env3: Envelope | None = None,
        env4: Envelope | None = None,
        env5: Envelope | None = None,
        env_carrier: Envelope | None = None,
        # Escalas por operador (multiplica o envelope do operador)
        env_scale1: float = 1.0,
        env_scale2: float = 1.0,
        env_scale3: float = 1.0,
        env_scale4: float = 1.0,
        env_scale5: float = 1.0,
        env_scale_carrier: float = 1.0,
        # Nível da portadora (amplitude)
        carrier_level: float = 1.0,
        # Feedback (aplicado em op1; em radianos)
        feedback: float = 0.0,
        # LFO (vibrato)
        lfo_rate: float = 5.0,
        lfo_depth_cents: float = 0.0,
        # Key scaling do índice (0 = desliga)
        key_scaling: float = 0.0,
        key_scaling_ref_hz: float = 440.0,
        # Limites e segurança anti-alias
        index_max: float | None = 12.0,
        anti_alias: bool = True,
        # Fase inicial aleatória por operador
        random_phase: bool = True,
        # Fases iniciais explícitas por operador (rad). Se fornecidas, ignoram random_phase.
        phase1: float | None = None,
        phase2: float | None = None,
        phase3: float | None = None,
        phase4: float | None = None,
        phase5: float | None = None,
        phase_carrier: float | None = None,
        # Normalização de pico no final
        normalize: bool = True,
        # Downsample para 16 kHz
        downsample_16k: bool = True,
    ) -> None:
        self.ratio1 = ratio1
        self.ratio2 = ratio2
        self.ratio3 = ratio3
        self.ratio4 = ratio4
        self.ratio5 = ratio5
        self.ratio_carrier = ratio_carrier

        self.detune1 = detune1
        self.detune2 = detune2
        self.detune3 = detune3
        self.detune4 = detune4
        self.detune5 = detune5
        self.detune_carrier = detune_carrier

        self.index_12 = index_12
        self.index_23 = index_23
        self.index_3c = index_3c
        self.index_4c = index_4c
        self.index_5c = index_5c

        self.env_mod = env_mod if env_mod is not None else Envelope(0.01, 0.2, 0.4, 0.2)
        self.env_car = env_car if env_car is not None else Envelope(0.01, 0.2, 0.7, 0.2)

        self.env1 = env1
        self.env2 = env2
        self.env3 = env3
        self.env4 = env4
        self.env5 = env5
        self.env_carrier = env_carrier

        self.env_scale1 = env_scale1
        self.env_scale2 = env_scale2
        self.env_scale3 = env_scale3
        self.env_scale4 = env_scale4
        self.env_scale5 = env_scale5
        self.env_scale_carrier = env_scale_carrier

        self.carrier_level = carrier_level
        self.feedback = feedback
        self.lfo_rate = lfo_rate
        self.lfo_depth_cents = lfo_depth_cents
        self.key_scaling = key_scaling
        self.key_scaling_ref_hz = key_scaling_ref_hz
        self.index_max = index_max
        self.anti_alias = anti_alias
        self.random_phase = random_phase
        self.phase1 = phase1
        self.phase2 = phase2
        self.phase3 = phase3
        self.phase4 = phase4
        self.phase5 = phase5
        self.phase_carrier = phase_carrier
        self.normalize = normalize
        self.downsample_16k = downsample_16k

    # ----------------------------
    # API principal
    # ----------------------------
    def synth_alg_series3_parallel2(self, audio_seconds: float, frequency_carrier: float):
        """
        Topologia:
          (op1 -> op2 -> op3)  ||  op4  ||  op5  --> carrier
        """
        return self.synth(audio_seconds, frequency_carrier, algorithm="series3_parallel2")

    def list_algorithms(self) -> Tuple[str, ...]:
        """
        Algoritmos disponíveis e mapeamento dos índices:
        - series3_parallel2: index_12 (op1->op2), index_23 (op2->op3),
          index_3c (op3->carrier), index_4c (op4->carrier), index_5c (op5->carrier)
        - series3: index_12 (op1->op2), index_23 (op2->op3), index_3c (op3->carrier)
        - parallel5: index_12 (op1->carrier), index_23 (op2->carrier), index_3c (op3->carrier),
          index_4c (op4->carrier), index_5c (op5->carrier)
        - series2x2_parallel1: index_12 (op1->op2), index_23 (op3->op4),
          index_3c (op2->carrier), index_4c (op4->carrier), index_5c (op5->carrier)
        - series5: index_12 (op1->op2), index_23 (op2->op3), index_3c (op3->op4),
          index_4c (op4->op5), index_5c (op5->carrier)
        - series4_parallel1: index_12 (op1->op2), index_23 (op2->op3), index_3c (op3->op4),
          index_4c (op4->carrier), index_5c (op5->carrier)
        - series2_parallel3: index_12 (op1->op2), index_23 (op2->carrier), index_3c (op3->carrier),
          index_4c (op4->carrier), index_5c (op5->carrier)
        - series3_parallel1_plus1: index_12 (op1->op2), index_23 (op2->op3), index_3c (op3->carrier),
          index_4c (op4->op5), index_5c (op5->carrier)
        - dual_chain: alias de series2x2_parallel1
        """
        return (
            "series3_parallel2",
            "series3",
            "parallel5",
            "series2x2_parallel1",
            "series5",
            "series4_parallel1",
            "series2_parallel3",
            "series3_parallel1_plus1",
            "dual_chain",
        )

    def synth(self, audio_seconds: float, frequency_carrier: float, algorithm: str = "series3_parallel2"):
        return self._synth(audio_seconds, frequency_carrier, algorithm)

    # ----------------------------
    # Internals
    # ----------------------------
    def _synth(self, audio_seconds: float, frequency_carrier: float, algorithm: str):
        common = self._prepare_common(audio_seconds, frequency_carrier)
        if common["n_samples"] <= 0:
            return np.zeros(0, dtype=np.float64)

        algo = algorithm or "series3_parallel2"
        if algo == "series3_parallel2":
            out = self._alg_series3_parallel2(common)
        elif algo == "series3":
            out = self._alg_series3(common)
        elif algo == "parallel5":
            out = self._alg_parallel5(common)
        elif algo == "series2x2_parallel1":
            out = self._alg_series2x2_parallel1(common)
        elif algo == "series5":
            out = self._alg_series5(common)
        elif algo == "series4_parallel1":
            out = self._alg_series4_parallel1(common)
        elif algo == "series2_parallel3":
            out = self._alg_series2_parallel3(common)
        elif algo == "series3_parallel1_plus1":
            out = self._alg_series3_parallel1_plus1(common)
        elif algo == "dual_chain":
            out = self._alg_series2x2_parallel1(common)
        else:
            raise ValueError(f"Algoritmo desconhecido: {algorithm}. Use um destes: {self.list_algorithms()}")

        # Remoção de DC
        out = out - np.mean(out)

        # Downsample opcional
        if self.downsample_16k:
            out = resample_poly(out, up=1, down=DECIM, window=("kaiser", 14))

        if self.normalize:
            peak = np.max(np.abs(out)) + 1e-12
            out = 0.99 * (out / peak)

        return out.astype(np.float64, copy=False)

    def _prepare_common(self, audio_seconds: float, frequency_carrier: float) -> dict:
        sr = SAMPLE_RATE_RENDER
        n_samples = int(sr * audio_seconds)
        if n_samples <= 0:
            return {"n_samples": 0}

        t = np.arange(n_samples) / sr

        # LFO global (vibrato)
        if self.lfo_depth_cents != 0.0 and self.lfo_rate > 0.0:
            lfo = np.sin(2.0 * np.pi * self.lfo_rate * t)
            pitch_mod = 2.0 ** ((self.lfo_depth_cents / 1200.0) * lfo)
        else:
            pitch_mod = np.ones_like(t)

        f_base = frequency_carrier * pitch_mod

        # Frequências instantâneas por operador
        f1 = f_base * self.ratio1 * self._detune_factor(self.detune1)
        f2 = f_base * self.ratio2 * self._detune_factor(self.detune2)
        f3 = f_base * self.ratio3 * self._detune_factor(self.detune3)
        f4 = f_base * self.ratio4 * self._detune_factor(self.detune4)
        f5 = f_base * self.ratio5 * self._detune_factor(self.detune5)
        f_car = f_base * self.ratio_carrier * self._detune_factor(self.detune_carrier)

        # Envelopes (por operador, com fallback no global)
        env_mod_base = self._adsr_env(n_samples, sr, self.env_mod)
        env_car_base = self._adsr_env(n_samples, sr, self.env_car)
        env_mod1 = (self._adsr_env(n_samples, sr, self.env1) if self.env1 else env_mod_base) * self.env_scale1
        env_mod2 = (self._adsr_env(n_samples, sr, self.env2) if self.env2 else env_mod_base) * self.env_scale2
        env_mod3 = (self._adsr_env(n_samples, sr, self.env3) if self.env3 else env_mod_base) * self.env_scale3
        env_mod4 = (self._adsr_env(n_samples, sr, self.env4) if self.env4 else env_mod_base) * self.env_scale4
        env_mod5 = (self._adsr_env(n_samples, sr, self.env5) if self.env5 else env_mod_base) * self.env_scale5
        env_car = (
            (self._adsr_env(n_samples, sr, self.env_carrier) if self.env_carrier else env_car_base)
            * self.env_scale_carrier
        )

        # Índices com key scaling e clamp anti-alias
        i12 = self._scale_index(
            self.index_12,
            frequency_carrier,
            self.ratio1 * self._detune_factor(self.detune1),
            self.ratio2 * self._detune_factor(self.detune2),
        )
        i23 = self._scale_index(
            self.index_23,
            frequency_carrier,
            self.ratio2 * self._detune_factor(self.detune2),
            self.ratio3 * self._detune_factor(self.detune3),
        )
        i3c = self._scale_index(
            self.index_3c,
            frequency_carrier,
            self.ratio3 * self._detune_factor(self.detune3),
            self.ratio_carrier * self._detune_factor(self.detune_carrier),
        )
        i4c = self._scale_index(
            self.index_4c,
            frequency_carrier,
            self.ratio4 * self._detune_factor(self.detune4),
            self.ratio_carrier * self._detune_factor(self.detune_carrier),
        )
        i5c = self._scale_index(
            self.index_5c,
            frequency_carrier,
            self.ratio5 * self._detune_factor(self.detune5),
            self.ratio_carrier * self._detune_factor(self.detune_carrier),
        )

        return {
            "n_samples": n_samples,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4,
            "f5": f5,
            "f_car": f_car,
            "env_mod1": env_mod1,
            "env_mod2": env_mod2,
            "env_mod3": env_mod3,
            "env_mod4": env_mod4,
            "env_mod5": env_mod5,
            "env_car": env_car,
            "i12": i12,
            "i23": i23,
            "i3c": i3c,
            "i4c": i4c,
            "i5c": i5c,
        }

    def _alg_series3_parallel2(self, c: dict) -> np.ndarray:
        op1 = self._oscillator(
            c["f1"],
            mod=None,
            index=0.0,
            env=None,
            feedback=self.feedback,
            feedback_env=c["env_mod1"],
            phase0=self._phase_or_random(self.phase1),
        )
        op2 = self._oscillator(
            c["f2"],
            mod=self._normalize_mod(op1),
            index=c["i12"],
            env=c["env_mod2"],
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase2),
        )
        op3 = self._oscillator(
            c["f3"],
            mod=self._normalize_mod(op2),
            index=c["i23"],
            env=c["env_mod3"],
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase3),
        )
        op4 = self._oscillator(
            c["f4"],
            mod=None,
            index=0.0,
            env=None,
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase4),
        )
        op5 = self._oscillator(
            c["f5"],
            mod=None,
            index=0.0,
            env=None,
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase5),
        )
        mod_sum = (
            c["i3c"] * c["env_mod3"] * self._normalize_mod(op3)
            + c["i4c"] * c["env_mod4"] * self._normalize_mod(op4)
            + c["i5c"] * c["env_mod5"] * self._normalize_mod(op5)
        )
        carrier = self._oscillator(
            c["f_car"],
            mod=mod_sum,
            index=1.0,
            env=np.ones_like(c["env_car"]),
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase_carrier),
        )
        return carrier * c["env_car"] * self.carrier_level

    def _alg_series3(self, c: dict) -> np.ndarray:
        op1 = self._oscillator(
            c["f1"],
            mod=None,
            index=0.0,
            env=None,
            feedback=self.feedback,
            feedback_env=c["env_mod1"],
            phase0=self._phase_or_random(self.phase1),
        )
        op2 = self._oscillator(
            c["f2"],
            mod=self._normalize_mod(op1),
            index=c["i12"],
            env=c["env_mod2"],
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase2),
        )
        op3 = self._oscillator(
            c["f3"],
            mod=self._normalize_mod(op2),
            index=c["i23"],
            env=c["env_mod3"],
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase3),
        )
        mod_sum = c["i3c"] * c["env_mod3"] * self._normalize_mod(op3)
        carrier = self._oscillator(
            c["f_car"],
            mod=mod_sum,
            index=1.0,
            env=np.ones_like(c["env_car"]),
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase_carrier),
        )
        return carrier * c["env_car"] * self.carrier_level

    def _alg_parallel5(self, c: dict) -> np.ndarray:
        op1 = self._oscillator(
            c["f1"],
            mod=None,
            index=0.0,
            env=None,
            feedback=self.feedback,
            feedback_env=c["env_mod1"],
            phase0=self._phase_or_random(self.phase1),
        )
        op2 = self._oscillator(
            c["f2"],
            mod=None,
            index=0.0,
            env=None,
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase2),
        )
        op3 = self._oscillator(
            c["f3"],
            mod=None,
            index=0.0,
            env=None,
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase3),
        )
        op4 = self._oscillator(
            c["f4"],
            mod=None,
            index=0.0,
            env=None,
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase4),
        )
        op5 = self._oscillator(
            c["f5"],
            mod=None,
            index=0.0,
            env=None,
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase5),
        )
        mod_sum = (
            c["i12"] * c["env_mod1"] * self._normalize_mod(op1)
            + c["i23"] * c["env_mod2"] * self._normalize_mod(op2)
            + c["i3c"] * c["env_mod3"] * self._normalize_mod(op3)
            + c["i4c"] * c["env_mod4"] * self._normalize_mod(op4)
            + c["i5c"] * c["env_mod5"] * self._normalize_mod(op5)
        )
        carrier = self._oscillator(
            c["f_car"],
            mod=mod_sum,
            index=1.0,
            env=np.ones_like(c["env_car"]),
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase_carrier),
        )
        return carrier * c["env_car"] * self.carrier_level

    def _alg_series2x2_parallel1(self, c: dict) -> np.ndarray:
        op1 = self._oscillator(
            c["f1"],
            mod=None,
            index=0.0,
            env=None,
            feedback=self.feedback,
            feedback_env=c["env_mod1"],
            phase0=self._phase_or_random(self.phase1),
        )
        op2 = self._oscillator(
            c["f2"],
            mod=self._normalize_mod(op1),
            index=c["i12"],
            env=c["env_mod2"],
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase2),
        )
        op3 = self._oscillator(
            c["f3"],
            mod=None,
            index=0.0,
            env=None,
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase3),
        )
        op4 = self._oscillator(
            c["f4"],
            mod=self._normalize_mod(op3),
            index=c["i23"],
            env=c["env_mod4"],
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase4),
        )
        op5 = self._oscillator(
            c["f5"],
            mod=None,
            index=0.0,
            env=None,
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase5),
        )
        mod_sum = (
            c["i3c"] * c["env_mod2"] * self._normalize_mod(op2)
            + c["i4c"] * c["env_mod4"] * self._normalize_mod(op4)
            + c["i5c"] * c["env_mod5"] * self._normalize_mod(op5)
        )
        carrier = self._oscillator(
            c["f_car"],
            mod=mod_sum,
            index=1.0,
            env=np.ones_like(c["env_car"]),
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase_carrier),
        )
        return carrier * c["env_car"] * self.carrier_level

    def _alg_series5(self, c: dict) -> np.ndarray:
        op1 = self._oscillator(
            c["f1"],
            mod=None,
            index=0.0,
            env=None,
            feedback=self.feedback,
            feedback_env=c["env_mod1"],
            phase0=self._phase_or_random(self.phase1),
        )
        op2 = self._oscillator(
            c["f2"],
            mod=self._normalize_mod(op1),
            index=c["i12"],
            env=c["env_mod2"],
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase2),
        )
        op3 = self._oscillator(
            c["f3"],
            mod=self._normalize_mod(op2),
            index=c["i23"],
            env=c["env_mod3"],
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase3),
        )
        op4 = self._oscillator(
            c["f4"],
            mod=self._normalize_mod(op3),
            index=c["i3c"],
            env=c["env_mod4"],
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase4),
        )
        op5 = self._oscillator(
            c["f5"],
            mod=self._normalize_mod(op4),
            index=c["i4c"],
            env=c["env_mod5"],
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase5),
        )
        mod_sum = c["i5c"] * c["env_mod5"] * self._normalize_mod(op5)
        carrier = self._oscillator(
            c["f_car"],
            mod=mod_sum,
            index=1.0,
            env=np.ones_like(c["env_car"]),
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase_carrier),
        )
        return carrier * c["env_car"] * self.carrier_level

    def _alg_series4_parallel1(self, c: dict) -> np.ndarray:
        op1 = self._oscillator(
            c["f1"],
            mod=None,
            index=0.0,
            env=None,
            feedback=self.feedback,
            feedback_env=c["env_mod1"],
            phase0=self._phase_or_random(self.phase1),
        )
        op2 = self._oscillator(
            c["f2"],
            mod=self._normalize_mod(op1),
            index=c["i12"],
            env=c["env_mod2"],
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase2),
        )
        op3 = self._oscillator(
            c["f3"],
            mod=self._normalize_mod(op2),
            index=c["i23"],
            env=c["env_mod3"],
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase3),
        )
        op4 = self._oscillator(
            c["f4"],
            mod=self._normalize_mod(op3),
            index=c["i3c"],
            env=c["env_mod4"],
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase4),
        )
        op5 = self._oscillator(
            c["f5"],
            mod=None,
            index=0.0,
            env=None,
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase5),
        )
        mod_sum = (c["i4c"] * c["env_mod4"] * self._normalize_mod(op4)) + (
            c["i5c"] * c["env_mod5"] * self._normalize_mod(op5)
        )
        carrier = self._oscillator(
            c["f_car"],
            mod=mod_sum,
            index=1.0,
            env=np.ones_like(c["env_car"]),
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase_carrier),
        )
        return carrier * c["env_car"] * self.carrier_level

    def _alg_series2_parallel3(self, c: dict) -> np.ndarray:
        op1 = self._oscillator(
            c["f1"],
            mod=None,
            index=0.0,
            env=None,
            feedback=self.feedback,
            feedback_env=c["env_mod1"],
            phase0=self._phase_or_random(self.phase1),
        )
        op2 = self._oscillator(
            c["f2"],
            mod=self._normalize_mod(op1),
            index=c["i12"],
            env=c["env_mod2"],
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase2),
        )
        op3 = self._oscillator(
            c["f3"],
            mod=None,
            index=0.0,
            env=None,
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase3),
        )
        op4 = self._oscillator(
            c["f4"],
            mod=None,
            index=0.0,
            env=None,
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase4),
        )
        op5 = self._oscillator(
            c["f5"],
            mod=None,
            index=0.0,
            env=None,
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase5),
        )
        mod_sum = (
            c["i23"] * c["env_mod2"] * self._normalize_mod(op2)
            + c["i3c"] * c["env_mod3"] * self._normalize_mod(op3)
            + c["i4c"] * c["env_mod4"] * self._normalize_mod(op4)
            + c["i5c"] * c["env_mod5"] * self._normalize_mod(op5)
        )
        carrier = self._oscillator(
            c["f_car"],
            mod=mod_sum,
            index=1.0,
            env=np.ones_like(c["env_car"]),
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase_carrier),
        )
        return carrier * c["env_car"] * self.carrier_level

    def _alg_series3_parallel1_plus1(self, c: dict) -> np.ndarray:
        op1 = self._oscillator(
            c["f1"],
            mod=None,
            index=0.0,
            env=None,
            feedback=self.feedback,
            feedback_env=c["env_mod1"],
            phase0=self._phase_or_random(self.phase1),
        )
        op2 = self._oscillator(
            c["f2"],
            mod=self._normalize_mod(op1),
            index=c["i12"],
            env=c["env_mod2"],
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase2),
        )
        op3 = self._oscillator(
            c["f3"],
            mod=self._normalize_mod(op2),
            index=c["i23"],
            env=c["env_mod3"],
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase3),
        )
        op4 = self._oscillator(
            c["f4"],
            mod=None,
            index=0.0,
            env=None,
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase4),
        )
        op5 = self._oscillator(
            c["f5"],
            mod=self._normalize_mod(op4),
            index=c["i4c"],
            env=c["env_mod5"],
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase5),
        )
        mod_sum = (c["i3c"] * c["env_mod3"] * self._normalize_mod(op3)) + (
            c["i5c"] * c["env_mod5"] * self._normalize_mod(op5)
        )
        carrier = self._oscillator(
            c["f_car"],
            mod=mod_sum,
            index=1.0,
            env=np.ones_like(c["env_car"]),
            feedback=0.0,
            feedback_env=None,
            phase0=self._phase_or_random(self.phase_carrier),
        )
        return carrier * c["env_car"] * self.carrier_level

    # ----------------------------
    # Utilitários
    # ----------------------------
    @staticmethod
    def index_from_delta_f(delta_f_hz: float, f_mod_hz: float) -> float:
        """Converte Δf (Hz) em índice I = Δf / f_m."""
        return delta_f_hz / max(f_mod_hz, 1e-12)

    @staticmethod
    def delta_f_from_index(index: float, f_mod_hz: float) -> float:
        """Converte índice I em Δf (Hz)."""
        return index * f_mod_hz

    def _rand_phase(self) -> float:
        if not self.random_phase:
            return 0.0
        return np.random.uniform(0.0, 2.0 * np.pi)

    def _phase_or_random(self, phase: float | None) -> float:
        if phase is not None:
            return phase
        return self._rand_phase()

    def _detune_factor(self, cents: float) -> float:
        return 2.0 ** (cents / 1200.0)

    def _adsr_env(self, n_samples: int, sr: int, env: Envelope) -> np.ndarray:
        a = int(round(sr * max(env.attack, 0.0)))
        d = int(round(sr * max(env.decay, 0.0)))
        r = int(round(sr * max(env.release, 0.0)))
        s = max(n_samples - (a + d + r), 0)

        def segment(n: int, y0: float, y1: float, curve: str) -> np.ndarray:
            if n <= 0:
                return np.zeros(0)
            x = np.linspace(0.0, 1.0, n, endpoint=False)
            c = curve.lower().strip()
            if c == "linear":
                y = y0 + (y1 - y0) * x
                return y
            if c == "exp":
                y0e = max(y0, 1e-4)
                y1e = max(y1, 1e-4)
                return y0e * ((y1e / y0e) ** x)
            if c in ("log", "logarithmic"):
                a = 9.0
                w = np.log1p(a * x) / np.log1p(a)
                return y0 + (y1 - y0) * w
            if c in ("s", "s_curve", "smooth"):
                w = x * x * (3.0 - 2.0 * x)
                return y0 + (y1 - y0) * w
            raise ValueError(f"Curva de envelope desconhecida: {curve}")

        seg_a = segment(a, 1e-3, 1.0, env.curve_attack)
        seg_d = segment(d, 1.0, max(env.sustain, 1e-3), env.curve_decay)
        seg_s = np.full(s, max(env.sustain, 0.0), dtype=np.float64)
        seg_r = segment(r, max(env.sustain, 1e-3), 1e-3, env.curve_release)

        out = np.concatenate([seg_a, seg_d, seg_s, seg_r])
        if out.size < n_samples:
            out = np.pad(out, (0, n_samples - out.size), mode="edge")
        elif out.size > n_samples:
            out = out[:n_samples]
        return out

    def _normalize_mod(self, m: np.ndarray) -> np.ndarray:
        if m.size == 0:
            return m
        m0 = m - np.mean(m)
        peak = np.max(np.abs(m0)) + 1e-12
        return m0 / peak

    def _scale_index(self, index: float, f_note: float, ratio_m: float, ratio_c: float) -> float:
        # Key scaling
        if self.key_scaling != 0.0 and f_note > 0.0:
            scale = (f_note / max(self.key_scaling_ref_hz, 1e-6)) ** self.key_scaling
        else:
            scale = 1.0

        idx = index * scale

        # Clamp por limite global
        if self.index_max is not None:
            idx = min(idx, self.index_max)

        # Clamp anti-alias aproximado: f_c + I * f_m < Nyquist
        if self.anti_alias and f_note > 0.0:
            f_m = f_note * max(ratio_m, 0.0)
            f_c = f_note * max(ratio_c, 0.0)
            if f_m > 0.0:
                nyq = SAMPLE_RATE_RENDER * 0.5 * 0.95
                i_max = max((nyq - f_c) / f_m, 0.0)
                idx = min(idx, i_max)
        return idx

    def _oscillator(
        self,
        f_inst: np.ndarray,
        mod: np.ndarray | None,
        index: float,
        env: np.ndarray | None,
        feedback: float,
        feedback_env: np.ndarray | None,
        phase0: float,
    ) -> np.ndarray:
        # Se a frequência é zero, retorna zeros
        if np.allclose(f_inst, 0.0):
            return np.zeros_like(f_inst)

        # Pré-cálculo da fase base
        phase_base = 2.0 * np.pi * np.cumsum(f_inst) / SAMPLE_RATE_RENDER + phase0

        if mod is None or index == 0.0:
            mod_term = None
        else:
            mod_term = index * mod
            if env is not None:
                mod_term = mod_term * env

        if feedback == 0.0:
            if mod_term is None:
                phase = phase_base
            else:
                phase = phase_base + mod_term
            return np.sin(phase)

        # Feedback exige loop simples (op1)
        out = np.empty_like(f_inst)
        prev = 0.0
        for i in range(f_inst.size):
            fb = feedback * prev
            if feedback_env is not None:
                fb = fb * feedback_env[i]
            phase_i = phase_base[i] + fb
            if mod_term is not None:
                phase_i += mod_term[i]
            out[i] = math.sin(phase_i)
            prev = out[i]
        return out


if __name__ == "__main__":
    # Exemplo com envelopes por operador
    synth = FMSynth3(
        ratio1=1.0,
        ratio2=2.0,
        ratio3=1.0,
        ratio4=3.0,
        ratio5=0.5,
        ratio_carrier=1.0,
        index_12=2.0,
        index_23=1.5,
        index_3c=2.0,
        index_4c=0.8,
        index_5c=0.6,
        feedback=0.2,
        lfo_rate=5.5,
        lfo_depth_cents=8.0,
        env1=Envelope(0.002, 0.12, 0.25, 0.08, curve_attack="exp", curve_decay="exp", curve_release="exp"),
        env2=Envelope(0.004, 0.18, 0.30, 0.12, curve_attack="log", curve_decay="exp", curve_release="exp"),
        env3=Envelope(0.006, 0.25, 0.35, 0.16, curve_attack="exp", curve_decay="s_curve", curve_release="exp"),
        env4=Envelope(0.003, 0.15, 0.20, 0.10, curve_attack="linear", curve_decay="exp", curve_release="exp"),
        env5=Envelope(0.008, 0.30, 0.40, 0.18, curve_attack="exp", curve_decay="log", curve_release="exp"),
        env_carrier=Envelope(0.01, 0.35, 0.75, 0.25, curve_attack="exp", curve_decay="s_curve", curve_release="exp"),
        downsample_16k=True,
    )

    audio_seconds = 4.0
    note_hz = 440.0
    y = synth.synth(audio_seconds, note_hz, algorithm="series3_parallel2")
    sf.write("output3.wav", y, SAMPLE_RATE_OUT if synth.downsample_16k else SAMPLE_RATE_RENDER)
