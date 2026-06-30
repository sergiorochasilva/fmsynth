"""Predict FM parameters for NSynth using the hierarchical `big16` algorithm models.

Architecture:
- Loads a two-stage hierarchical algorithm predictor:
  - `family`: coarse `series` vs `parallel`
  - `exact`: family-conditioned exact classifier with dedicated `series` and `parallel` subheads
- Reuses the previously validated `big15_0_2` `carrier` and `modulators` submodels
- Predicts the subspaces in cascade and merges the outputs into a single parameter table

Data flow:
- Input: `nsynth-test/audio`, `nsynth-test/examples.json`, and the `big16` algorithm hierarchy plus the `big15_0_2` split submodels
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

from model_training_big16_fmsynth3_0_2 import GROUP_SPECS as HIER_GROUP_SPECS
from model_training_big16_fmsynth3_0_2 import MODEL_PREFIX as HIER_MODEL_PREFIX
from model_training_big16_fmsynth3_0_2 import build_model as build_hier_model
from model_training_big15_fmsynth3_0_2 import (
    GROUP_SPECS as SPLIT_GROUP_SPECS,
    MODEL_PREFIX as SPLIT_MODEL_PREFIX,
    build_model,
    inverse_transform_series,
)

MODEL_NAME = "model_training_big16_fmsynth3_0_2"
DEFAULT_AUDIO_DIR = Path("nsynth-test/audio")
DEFAULT_EXAMPLES_JSON = Path("nsynth-test/examples.json")
DEFAULT_META_JSON = Path("dataset_big16/meta.json")
DEFAULT_OUTPUT_DIR = Path("nsynth-pred-big16_0_2")

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
    parser = argparse.ArgumentParser(description="Predict NSynth parameters with the hierarchical big16 model cascade.")
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


def load_hier_model(stage: str, audio_len: int):
    model_dir = Path(f"{HIER_MODEL_PREFIX}_{stage}")
    weights_path = model_dir / "checkpoints" / "best.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights for stage `{stage}`: {weights_path}")
    n_classes = 2
    model = build_hier_model(audio_len, n_classes, stage)
    model.load_weights(str(weights_path))
    return model, model_dir


def load_split_model(group: str, audio_len: int, n_algorithm_classes: int):
    model_dir = Path(f"{SPLIT_MODEL_PREFIX}_{group}")
    weights_path = model_dir / "checkpoints" / "best.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights for group `{group}`: {weights_path}")
    model = build_model(audio_len, n_algorithm_classes, group, SPLIT_GROUP_SPECS[group])
    model.load_weights(str(weights_path))
    return model, model_dir


def midi_to_hz(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((int(midi_note) - 69) / 12.0))


def main() -> None:
    args = parse_args()

    if not args.examples_json.exists():
        raise FileNotFoundError(f"examples.json não encontrado: {args.examples_json}")
    with open(args.examples_json, "r", encoding="utf-8") as f:
        examples = json.load(f)

    if not args.meta_json.exists():
        raise FileNotFoundError(f"meta.json não encontrado: {args.meta_json}")
    with open(args.meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    audio_len = int(meta.get("audio_sample_len", 64000))

    family_results_path = Path(f"{HIER_MODEL_PREFIX}_family") / "results.json"
    exact_results_path = Path(f"{HIER_MODEL_PREFIX}_exact") / "results.json"
    with open(family_results_path, "r", encoding="utf-8") as f:
        family_results = json.load(f)
    with open(exact_results_path, "r", encoding="utf-8") as f:
        exact_results = json.load(f)
    family_classes = family_results.get("family_classes") or []
    exact_classes = exact_results.get("exact_classes") or []
    series_exact_classes = exact_results.get("series_exact_classes") or sorted([cls for cls in exact_classes if cls in {"series3", "series3_parallel2"}])
    parallel_exact_classes = exact_results.get("parallel_exact_classes") or sorted([cls for cls in exact_classes if cls in {"parallel5", "series2x2_parallel1"}])
    if not family_classes or not exact_classes:
        raise RuntimeError("Could not load hierarchical algorithm classes from the pretraining results.")

    family_model, _ = load_hier_model("family", audio_len)
    exact_model, _ = load_hier_model("exact", audio_len)
    carrier_model, carrier_dir = load_split_model("carrier", audio_len, len(exact_classes))
    mod_model, mod_dir = load_split_model("modulators", audio_len, len(exact_classes))

    carrier_scalers = {}
    for spec in SPLIT_GROUP_SPECS["carrier"]:
        carrier_scalers[spec["head"]] = joblib.load(carrier_dir / f"{spec['head']}_scaler.joblib")

    mod_scalers = {}
    for spec in SPLIT_GROUP_SPECS["modulators"]:
        mod_scalers[spec["head"]] = joblib.load(mod_dir / f"{spec['head']}_scaler.joblib")

    wav_files = sorted(args.audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"Nenhum arquivo .wav encontrado em {args.audio_dir}")

    rows: list[dict] = []
    for start in range(0, len(wav_files), args.batch_size):
        batch_files = wav_files[start : start + args.batch_size]
        x_batch = np.concatenate([load_audio_file(path, audio_len) for path in batch_files], axis=0)

        family_pred = np.asarray(family_model.predict(x_batch, batch_size=len(batch_files), verbose=0), dtype=np.float32)
        family_idx = np.argmax(family_pred, axis=1)
        family_onehot = tf.keras.utils.to_categorical(family_idx, num_classes=len(family_classes)).astype(np.float32)
        exact_family_aux, exact_series_pred, exact_parallel_pred = [
            np.asarray(arr, dtype=np.float32)
            for arr in exact_model.predict({"audio_input": x_batch, "family_condition_input": family_onehot}, batch_size=len(batch_files), verbose=0)
        ]
        series_global_indices = np.asarray([exact_classes.index(name) for name in series_exact_classes], dtype=np.int32)
        parallel_global_indices = np.asarray([exact_classes.index(name) for name in parallel_exact_classes], dtype=np.int32)
        exact_idx = np.where(
            family_idx == family_classes.index("series"),
            series_global_indices[np.argmax(exact_series_pred, axis=1)],
            parallel_global_indices[np.argmax(exact_parallel_pred, axis=1)],
        ).astype(np.int32)
        exact_onehot = tf.keras.utils.to_categorical(exact_idx, num_classes=len(exact_classes)).astype(np.float32)
        carrier_pred = carrier_model.predict({"audio_input": x_batch, "algorithm_condition_input": exact_onehot}, batch_size=len(batch_files), verbose=0)
        carrier_pred = [np.asarray(arr, dtype=np.float32) for arr in carrier_pred]
        carrier_condition = np.concatenate(carrier_pred, axis=1)
        mod_pred = mod_model.predict(
            {
                "audio_input": x_batch,
                "algorithm_condition_input": exact_onehot,
                "carrier_condition_input": carrier_condition,
            },
            batch_size=len(batch_files),
            verbose=0,
        )

        family_name = [family_classes[i] for i in family_idx]
        exact_name = [exact_classes[i] for i in exact_idx]

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
                "algorithm_family_idx": int(family_idx[idx]),
                "algorithm_family": family_name[idx],
                "algorithm_family_prob": float(np.max(family_pred[idx])),
                "algorithm_family_aux_prob": float(np.max(exact_family_aux[idx])),
                "algorithm_idx": int(exact_idx[idx]),
                "algorithm": exact_name[idx],
                "algorithm_prob": float(np.where(
                    family_idx[idx] == family_classes.index("series"),
                    np.max(exact_series_pred[idx]),
                    np.max(exact_parallel_pred[idx]),
                )),
            }

            for spec_idx, spec in enumerate(SPLIT_GROUP_SPECS["carrier"]):
                scaler = carrier_scalers[spec["head"]]
                pred_scaled = np.asarray(carrier_pred[spec_idx], dtype=np.float32)
                pred_transformed = scaler.inverse_transform(pred_scaled[idx : idx + 1]).reshape(-1)
                pred_raw = inverse_transform_series(pred_transformed, spec["transform"])[0]
                out_key = "frequencia_base_pred" if spec["column"] == "frequencia_base" else spec["column"]
                clip_min, clip_max = CLIP_RANGES[out_key]
                row[out_key] = float(np.clip(pred_raw, clip_min, clip_max))

            for spec_idx, spec in enumerate(SPLIT_GROUP_SPECS["modulators"]):
                scaler = mod_scalers[spec["head"]]
                pred_scaled = np.asarray(mod_pred[spec_idx], dtype=np.float32)
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
