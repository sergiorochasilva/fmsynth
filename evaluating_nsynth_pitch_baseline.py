"""Evaluate NSynth resynthesis against two baselines.

Architecture:
- Loads paired NSynth originals and resynthesized outputs
- Computes FFT, STFT, and log-mel distances
- Compares the model output against two baselines:
  - a fixed 440 Hz sine wave
  - a pitch-matched sine wave derived from NSynth metadata

Data flow:
- Input: `nsynth-test/audio`, `nsynth-test/examples.json`, and `nsynth-pred/`
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

    s1 = librosa.feature.melspectrogram(
        y=y1,
        sr=sr_ref,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    s2 = librosa.feature.melspectrogram(
        y=y2,
        sr=sr_ref,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )

    log_s1 = np.log1p(s1)
    log_s2 = np.log1p(s2)
    m = min(log_s1.shape[1], log_s2.shape[1])
    log_s1 = log_s1[:, :m]
    log_s2 = log_s2[:, :m]
    raw_dist = float(np.linalg.norm(log_s1 - log_s2))
    num_elements = int(log_s1.size)
    norm_dist = raw_dist / max(num_elements, 1)
    return raw_dist, float(norm_dist), num_elements


def compare_pair(reference: np.ndarray, candidate: np.ndarray) -> dict:
    logmel_raw, logmel_norm, num_elements = log_mel_spectrogram_distance(reference, candidate)
    return {
        "fft_distance": fft_distance(reference, candidate),
        "stft_distance": stft_distance(reference, candidate),
        "log_mel_spectrogram_distance_raw": logmel_raw,
        "log_mel_spectrogram_distance_normalized": logmel_norm,
        "log_mel_num_elements": num_elements,
    }


def summarize(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row]
    if not values:
        return None
    return float(np.mean(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate NSynth resynthesis and sine baselines.")
    parser.add_argument("--test-audio-dir", type=Path, default=Path("nsynth-test/audio"))
    parser.add_argument("--pred-audio-dir", type=Path, default=Path("nsynth-pred"))
    parser.add_argument("--examples-json", type=Path, default=Path("nsynth-test/examples.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_nsynth"))
    parser.add_argument(
        "--include-440-baseline",
        action="store_true",
        help="Also compute the fixed 440 Hz sine baseline.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.examples_json.exists():
        raise FileNotFoundError(f"Missing NSynth metadata: {args.examples_json}")

    with open(args.examples_json, "r", encoding="utf-8") as f:
        examples = json.load(f)

    test_files = sorted([f for f in os.listdir(args.test_audio_dir) if f.endswith(".wav")])
    model_rows: list[dict] = []
    pitch_rows: list[dict] = []
    sine440_rows: list[dict] = []

    for file_name in test_files:
        note_key = file_name[:-4]
        if note_key not in examples or "pitch" not in examples[note_key]:
            continue

        reference = load_audio_mono(args.test_audio_dir / file_name)
        pitch = int(examples[note_key]["pitch"])
        pitch_freq = midi_to_hz(pitch)
        pitch_baseline = sine_wave(pitch_freq, len(reference))

        model_path = args.pred_audio_dir / file_name
        if model_path.exists():
            candidate = load_audio_mono(model_path)
            model_rows.append(
                {
                    "file_name": file_name,
                    "pitch": pitch,
                    "pitch_hz": pitch_freq,
                    **compare_pair(reference, candidate),
                }
            )

        pitch_rows.append(
            {
                "file_name": file_name,
                "pitch": pitch,
                "pitch_hz": pitch_freq,
                **compare_pair(reference, pitch_baseline),
            }
        )

        if args.include_440_baseline:
            sine440_rows.append(
                {
                    "file_name": file_name,
                    "pitch": pitch,
                    "pitch_hz": pitch_freq,
                    **compare_pair(reference, sine_wave(440.0, len(reference))),
                }
            )

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
    }

    if args.include_440_baseline:
        summary["baseline_440"] = {
            "count": len(sine440_rows),
            "fft_distance_mean": summarize(sine440_rows, "fft_distance"),
            "stft_distance_mean": summarize(sine440_rows, "stft_distance"),
            "log_mel_spectrogram_distance_raw_mean": summarize(sine440_rows, "log_mel_spectrogram_distance_raw"),
            "log_mel_spectrogram_distance_normalized_mean": summarize(sine440_rows, "log_mel_spectrogram_distance_normalized"),
        }

    payload = {
        "summary": summary,
        "model_vs_nsynth": model_rows,
        "pitch_baseline_vs_nsynth": pitch_rows,
    }
    if args.include_440_baseline:
        payload["baseline_440_vs_nsynth"] = sine440_rows

    output_path = args.output_dir / "nsynth_evaluation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Evaluation written to {output_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
