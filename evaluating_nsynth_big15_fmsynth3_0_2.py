"""Evaluate split big15 `0_2` NSynth resynthesis against stronger baselines.

Architecture:
- Loads paired NSynth originals and split-model resynthesized outputs
- Computes FFT, STFT, and log-mel distances
- Compares the model output against:
  - fixed 440 Hz sine baseline
  - pitch-matched sine baseline
  - pitch-matched harmonic baseline

Data flow:
- Input: `nsynth-test/audio`, `nsynth-test/examples.json`, and `nsynth-pred-big15_0_2/`
- Output: JSON summaries with per-file and aggregate metrics
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

SR_REF = 16000
EPS = 1e-9


def load_audio_mono(file_path: Path, target_sr: int = SR_REF) -> np.ndarray:
    y, sr = sf.read(file_path, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != target_sr:
        raise AssertionError(f"Unexpected sample rate for {file_path}: {sr} (expected {target_sr})")
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y)))
    if peak > 0.0:
        y = y / peak
    return y.astype(np.float32, copy=False)


def normalize_audio(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.0:
        y = y / peak
    return y.astype(np.float32, copy=False)


def match_lengths(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = min(len(a), len(b))
    return a[:m], b[:m]


def midi_to_hz(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((int(midi_note) - 69) / 12.0))


def sine_wave(freq_hz: float, n_samples: int, sr: int = SR_REF) -> np.ndarray:
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n_samples, dtype=np.float32) / float(sr)
    return np.sin(2.0 * math.pi * float(freq_hz) * t).astype(np.float32, copy=False)


def harmonic_wave(freq_hz: float, n_samples: int, harmonics: int = 5, sr: int = SR_REF) -> np.ndarray:
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n_samples, dtype=np.float32) / float(sr)
    y = np.zeros(n_samples, dtype=np.float32)
    for k in range(1, harmonics + 1):
        y += (1.0 / float(k)) * np.sin(2.0 * math.pi * float(freq_hz) * float(k) * t).astype(np.float32, copy=False)
    return y.astype(np.float32, copy=False)


def fft_distance(y1: np.ndarray, y2: np.ndarray) -> float:
    y1, y2 = match_lengths(normalize_audio(y1), normalize_audio(y2))
    return float(np.linalg.norm(np.fft.rfft(y1) - np.fft.rfft(y2)))


def stft_distance(y1: np.ndarray, y2: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> float:
    y1, y2 = match_lengths(normalize_audio(y1), normalize_audio(y2))
    stft1 = np.abs(librosa.stft(y=y1, n_fft=n_fft, hop_length=hop_length))
    stft2 = np.abs(librosa.stft(y=y2, n_fft=n_fft, hop_length=hop_length))
    m = min(stft1.shape[1], stft2.shape[1])
    return float(np.linalg.norm(stft1[:, :m] - stft2[:, :m]))


def log_mel_spectrogram_distance(
    y1: np.ndarray,
    y2: np.ndarray,
    sr_ref: int = SR_REF,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> tuple[float, float, int]:
    y1, y2 = match_lengths(normalize_audio(y1), normalize_audio(y2))

    s1 = librosa.feature.melspectrogram(y=y1, sr=sr_ref, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    s2 = librosa.feature.melspectrogram(y=y2, sr=sr_ref, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)

    log1 = np.log(s1 + EPS)
    log2 = np.log(s2 + EPS)
    diff = log1 - log2
    raw = float(np.linalg.norm(diff))
    denom = float(np.linalg.norm(log1) + np.linalg.norm(log2) + EPS)
    normalized = float(raw / denom)
    return raw, normalized, int(log1.shape[1])


def compare_pair(reference: np.ndarray, candidate: np.ndarray) -> dict:
    raw, norm, frames = log_mel_spectrogram_distance(reference, candidate)
    return {
        "fft_distance": fft_distance(reference, candidate),
        "stft_distance": stft_distance(reference, candidate),
        "log_mel_spectrogram_distance_raw": raw,
        "log_mel_spectrogram_distance_normalized": norm,
        "frames": frames,
    }


def summarize(rows: list[dict], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([float(row[key]) for row in rows]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate split big15 resynthesis against NSynth baselines.")
    parser.add_argument("--audio-dir", type=Path, default=Path("nsynth-test/audio"))
    parser.add_argument("--examples-json", type=Path, default=Path("nsynth-test/examples.json"))
    parser.add_argument("--pred-dir", type=Path, default=Path("nsynth-pred-big15_0_2"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_nsynth_big15_0_2"))
    parser.add_argument("--include-440-baseline", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.examples_json, "r", encoding="utf-8") as f:
        examples = json.load(f)

    pred_files = sorted(args.pred_dir.glob("*.wav"))
    model_rows = []
    pitch_rows = []
    sine440_rows = []
    harmonic_rows = []

    for pred_path in pred_files:
        audio_name = pred_path.name
        note_key = audio_name.replace(".wav", "")
        if note_key not in examples:
            continue
        pitch = int(examples[note_key]["pitch"])
        pitch_freq = midi_to_hz(pitch)
        reference = load_audio_mono(args.audio_dir / audio_name)
        candidate = load_audio_mono(pred_path)
        pitch_baseline = sine_wave(pitch_freq, len(reference))
        harmonic_baseline = harmonic_wave(pitch_freq, len(reference))

        model_rows.append({"file_name": audio_name, "pitch": pitch, "pitch_hz": pitch_freq, **compare_pair(reference, candidate)})
        pitch_rows.append({"file_name": audio_name, "pitch": pitch, "pitch_hz": pitch_freq, **compare_pair(reference, pitch_baseline)})
        harmonic_rows.append({"file_name": audio_name, "pitch": pitch, "pitch_hz": pitch_freq, **compare_pair(reference, harmonic_baseline)})

        if args.include_440_baseline:
            sine440_rows.append({"file_name": audio_name, "pitch": pitch, "pitch_hz": pitch_freq, **compare_pair(reference, sine_wave(440.0, len(reference)))})

    summary = {
        "model": {
            "count": len(model_rows),
            "fft_distance_mean": summarize(model_rows, "fft_distance"),
            "stft_distance_mean": summarize(model_rows, "stft_distance"),
            "log_mel_spectrogram_distance_raw_mean": summarize(model_rows, "log_mel_spectrogram_distance_raw"),
            "log_mel_spectrogram_distance_normalized_mean": summarize(model_rows, "log_mel_spectrogram_distance_normalized"),
        },
        "pitch_baseline": {
            "count": len(pitch_rows),
            "fft_distance_mean": summarize(pitch_rows, "fft_distance"),
            "stft_distance_mean": summarize(pitch_rows, "stft_distance"),
            "log_mel_spectrogram_distance_raw_mean": summarize(pitch_rows, "log_mel_spectrogram_distance_raw"),
            "log_mel_spectrogram_distance_normalized_mean": summarize(pitch_rows, "log_mel_spectrogram_distance_normalized"),
        },
        "baseline_440": None,
        "harmonic_baseline": {
            "count": len(harmonic_rows),
            "fft_distance_mean": summarize(harmonic_rows, "fft_distance"),
            "stft_distance_mean": summarize(harmonic_rows, "stft_distance"),
            "log_mel_spectrogram_distance_raw_mean": summarize(harmonic_rows, "log_mel_spectrogram_distance_raw"),
            "log_mel_spectrogram_distance_normalized_mean": summarize(harmonic_rows, "log_mel_spectrogram_distance_normalized"),
        },
    }

    if args.include_440_baseline:
        summary["baseline_440"] = {
            "count": len(sine440_rows),
            "fft_distance_mean": summarize(sine440_rows, "fft_distance"),
            "stft_distance_mean": summarize(sine440_rows, "stft_distance"),
            "log_mel_spectrogram_distance_raw_mean": summarize(sine440_rows, "log_mel_spectrogram_distance_raw"),
            "log_mel_spectrogram_distance_normalized_mean": summarize(sine440_rows, "log_mel_spectrogram_distance_normalized"),
        }

    with open(args.output_dir / "nsynth_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame(model_rows).to_csv(args.output_dir / "model_metrics.csv", index=False)
    pd.DataFrame(pitch_rows).to_csv(args.output_dir / "pitch_baseline_metrics.csv", index=False)
    pd.DataFrame(harmonic_rows).to_csv(args.output_dir / "harmonic_baseline_metrics.csv", index=False)
    if args.include_440_baseline:
        pd.DataFrame(sine440_rows).to_csv(args.output_dir / "baseline_440_metrics.csv", index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
