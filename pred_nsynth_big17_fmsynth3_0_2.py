"""Predict NSynth parameters using the split `big17` cascade with the pitch `0_2` upgrade.

Architecture:
- Reuses the trained `algorithm`, `timbre`, and `envelope` branches from `0_1`
- Uses the new `pitch` branch from `0_2`, which adds an auxiliary class head for discrete `ratio_carrier`
- Runs the cascade in sequence and merges the outputs into one parameter table

Data flow:
- Input: `nsynth-test/audio`, `nsynth-test/examples.json`, and the versioned model directories
- Output: merged parameter table JSON/CSV for the resynthesis stage
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

import os

os.environ.pop("TARGET_GROUP", None)

from model_training_big17_fmsynth3_0_1 import (
    GROUP_SPECS as GROUP_SPECS_01,
    MODEL_PREFIX as MODEL_PREFIX_01,
    build_model as build_model_01,
    inverse_transform_series,
)
from model_training_big17_fmsynth3_0_2 import build_model as build_pitch_model

MODEL_NAME = "model_training_big17_fmsynth3_0_2"
DEFAULT_AUDIO_DIR = Path("nsynth-test/audio")
DEFAULT_EXAMPLES_JSON = Path("nsynth-test/examples.json")
DEFAULT_META_JSON = Path("dataset_big13/meta.json")
DEFAULT_OUTPUT_DIR = Path("nsynth-pred-big17_0_2")

CLIP_RANGES = {
    "ratio_carrier": (0.05, 8.0),
    "frequencia_base_pred": (40.0, 1200.0),
    "index_12": (0.0, 6.0),
    "index_23": (0.0, 6.0),
    "index_3c": (0.0, 6.0),
    "index_4c": (0.0, 6.0),
    "index_5c": (0.0, 6.0),
    "detune_carrier": (-15.0, 15.0),
    "feedback": (0.0, 0.65),
    "lfo_rate": (0.0, 12.0),
    "lfo_depth_cents": (0.0, 30.0),
    "key_scaling": (0.0, 1.0),
    "env_mod_attack": (0.001, 0.20),
    "env_mod_decay": (0.01, 0.80),
    "env_mod_sustain": (0.05, 0.98),
    "env_mod_release": (0.01, 1.0),
    "env_car_attack": (0.001, 0.20),
    "env_car_decay": (0.01, 0.90),
    "env_car_sustain": (0.05, 0.98),
    "env_car_release": (0.01, 1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict NSynth parameters with the conditioned split big17 cascade.")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--examples-json", type=Path, default=DEFAULT_EXAMPLES_JSON)
    parser.add_argument("--meta-json", type=Path, default=DEFAULT_META_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def load_audio_file(audio_path: Path, expected_len: int) -> np.ndarray:
    signal, sr = sf.read(str(audio_path))
    if sr != 16000:
        raise ValueError(f"Sample rate inesperado em {audio_path}: {sr}")
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    signal = np.asarray(signal, dtype=np.float32)
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 0:
        signal = 0.891 * signal / peak
    if signal.shape[0] > expected_len:
        signal = signal[:expected_len]
    elif signal.shape[0] < expected_len:
        signal = np.pad(signal, (0, expected_len - signal.shape[0]), mode="constant")
    return signal.reshape(1, expected_len, 1)


def load_model_01(group: str, audio_len: int, n_algorithm_classes: int):
    model_dir = Path(f"{MODEL_PREFIX_01}_{group}")
    weights_path = model_dir / "checkpoints" / "best.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights for group `{group}`: {weights_path}")
    model = build_model_01(audio_len, n_algorithm_classes, group, GROUP_SPECS_01[group])
    model.load_weights(str(weights_path))
    return model, model_dir


def load_pitch_model(audio_len: int, n_algorithm_classes: int):
    model_dir = Path(f"{MODEL_NAME}_pitch")
    weights_path = model_dir / "checkpoints" / "best.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights for pitch group: {weights_path}")
    ratio_classes_path = model_dir / "ratio_carrier_classes.npy"
    if not ratio_classes_path.exists():
        raise FileNotFoundError(f"Missing ratio classes file: {ratio_classes_path}")
    ratio_classes = np.load(ratio_classes_path).astype(np.float32)
    model = build_pitch_model(audio_len, n_algorithm_classes, len(ratio_classes))
    model.load_weights(str(weights_path))
    return model, model_dir, ratio_classes


def midi_to_hz(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((int(midi_note) - 69) / 12.0))


def main() -> None:
    args = parse_args()

    with open(args.examples_json, "r", encoding="utf-8") as f:
        examples = json.load(f)
    with open(args.meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    audio_len = int(meta.get("audio_sample_len", 64000))

    algo_dir = Path(f"{MODEL_PREFIX_01}_algorithm")
    with open(algo_dir / "results.json", "r", encoding="utf-8") as f:
        algo_results = json.load(f)
    algorithm_classes = algo_results.get("algorithm_classes") or []
    if not algorithm_classes:
        raise RuntimeError("Could not load algorithm classes from the algorithm model results.")

    algo_model, _ = load_model_01("algorithm", audio_len, len(algorithm_classes))
    pitch_model, pitch_dir, ratio_classes = load_pitch_model(audio_len, len(algorithm_classes))
    timbre_model, timbre_dir = load_model_01("timbre", audio_len, len(algorithm_classes))
    env_model, env_dir = load_model_01("envelope", audio_len, len(algorithm_classes))

    pitch_scalers = {
        "ratio_log2_head": joblib.load(pitch_dir / "ratio_log2_head_scaler.joblib"),
        "freq_log2_head": joblib.load(pitch_dir / "freq_log2_head_scaler.joblib"),
    }
    timbre_scalers = {spec["head"]: joblib.load(timbre_dir / f"{spec['head']}_scaler.joblib") for spec in GROUP_SPECS_01["timbre"]}
    env_scalers = {spec["head"]: joblib.load(env_dir / f"{spec['head']}_scaler.joblib") for spec in GROUP_SPECS_01["envelope"]}

    wav_files = sorted(args.audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"Nenhum arquivo .wav encontrado em {args.audio_dir}")

    rows: list[dict] = []
    for start in range(0, len(wav_files), args.batch_size):
        batch_files = wav_files[start : start + args.batch_size]
        x_batch = np.concatenate([load_audio_file(path, audio_len) for path in batch_files], axis=0)

        algo_pred = np.asarray(algo_model.predict(x_batch, batch_size=len(batch_files), verbose=0), dtype=np.float32)
        algo_idx = np.argmax(algo_pred, axis=1)
        algo_onehot = tf.keras.utils.to_categorical(algo_idx, num_classes=len(algorithm_classes)).astype(np.float32)

        pitch_pred = pitch_model.predict({"audio_input": x_batch, "algorithm_condition_input": algo_onehot}, batch_size=len(batch_files), verbose=0)
        pitch_pred = [np.asarray(arr, dtype=np.float32) for arr in pitch_pred]
        ratio_log2_scaled = pitch_pred[0]
        ratio_class_pred = pitch_pred[1]
        freq_log2_scaled = pitch_pred[2]
        pitch_condition = np.concatenate([ratio_log2_scaled, freq_log2_scaled], axis=1)

        timbre_pred = timbre_model.predict(
            {
                "audio_input": x_batch,
                "algorithm_condition_input": algo_onehot,
                "pitch_condition_input": pitch_condition,
            },
            batch_size=len(batch_files),
            verbose=0,
        )
        timbre_pred = [np.asarray(arr, dtype=np.float32) for arr in timbre_pred]
        timbre_condition = np.concatenate(timbre_pred, axis=1)

        env_pred = env_model.predict(
            {
                "audio_input": x_batch,
                "algorithm_condition_input": algo_onehot,
                "pitch_condition_input": pitch_condition,
                "timbre_condition_input": timbre_condition,
            },
            batch_size=len(batch_files),
            verbose=0,
        )
        env_pred = [np.asarray(arr, dtype=np.float32) for arr in env_pred]

        algo_name = [algorithm_classes[i] for i in algo_idx]
        ratio_class_idx = np.argmax(ratio_class_pred, axis=1)
        ratio_class_value = ratio_classes[ratio_class_idx]

        for idx, wav_path in enumerate(batch_files):
            name = wav_path.name
            note_key = name[:-4]
            if note_key not in examples or "pitch" not in examples[note_key]:
                continue
            row = {
                "audio_file": name,
                "note_key": note_key,
                "pitch": int(examples[note_key]["pitch"]),
                "pitch_hz": midi_to_hz(int(examples[note_key]["pitch"])),
                "algorithm_idx": int(algo_idx[idx]),
                "algorithm": algo_name[idx],
                "algorithm_prob": float(np.max(algo_pred[idx])),
                "ratio_log2_pred": float(ratio_log2_scaled[idx, 0]),
                "ratio_class_idx_pred": int(ratio_class_idx[idx]),
                "ratio_class_prob": float(np.max(ratio_class_pred[idx])),
                "ratio_carrier": float(np.clip(ratio_class_value[idx], *CLIP_RANGES["ratio_carrier"])),
            }

            ratio_scaler = pitch_scalers["ratio_log2_head"]
            freq_scaler = pitch_scalers["freq_log2_head"]
            ratio_transformed = ratio_scaler.inverse_transform(ratio_log2_scaled[idx : idx + 1]).reshape(-1)
            freq_transformed = freq_scaler.inverse_transform(freq_log2_scaled[idx : idx + 1]).reshape(-1)
            row["ratio_log2_head_pred"] = float(inverse_transform_series(ratio_transformed, "log2")[0])
            row["frequencia_base_pred"] = float(np.clip(inverse_transform_series(freq_transformed, "log2")[0], *CLIP_RANGES["frequencia_base_pred"]))

            for spec_idx, spec in enumerate(GROUP_SPECS_01["timbre"]):
                scaler = timbre_scalers[spec["head"]]
                pred_scaled = np.asarray(timbre_pred[spec_idx], dtype=np.float32)
                pred_transformed = scaler.inverse_transform(pred_scaled[idx : idx + 1]).reshape(-1)
                pred_raw = inverse_transform_series(pred_transformed, spec["transform"])[0]
                clip_min, clip_max = CLIP_RANGES[spec["column"]]
                row[spec["column"]] = float(np.clip(pred_raw, clip_min, clip_max))

            for spec_idx, spec in enumerate(GROUP_SPECS_01["envelope"]):
                scaler = env_scalers[spec["head"]]
                pred_scaled = np.asarray(env_pred[spec_idx], dtype=np.float32)
                pred_transformed = scaler.inverse_transform(pred_scaled[idx : idx + 1]).reshape(-1)
                pred_raw = inverse_transform_series(pred_transformed, spec["transform"])[0]
                clip_min, clip_max = CLIP_RANGES[spec["column"]]
                row[spec["column"]] = float(np.clip(pred_raw, clip_min, clip_max))

            rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / f"params_pred_nsynth_{MODEL_NAME}.json"
    output_csv = args.output_dir / f"params_pred_nsynth_{MODEL_NAME}.csv"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Predições salvas em {output_json}")
    print(f"Predições salvas em {output_csv}")


if __name__ == "__main__":
    main()
