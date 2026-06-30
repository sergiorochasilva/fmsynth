"""Predict FM parameters for NSynth using `model_training_big13_fmsynth3_0_1`.

Architecture:
- Loads the trained multi-task CNN from the `big13` experiment
- Preprocesses NSynth audio into the expected waveform tensor
- Exports predicted `algorithm`, `ratio_carrier`, `frequencia_base`, FM indices, detune, feedback, LFO, key scaling, and ADSR envelopes

Data flow:
- Input: `nsynth-test/audio` and the saved `model_training_big13_fmsynth3_0_1.keras`
- Output: parameter table JSON/CSV for the NSynth resynthesis stage
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import joblib

from model_training_big13_fmsynth3_0_1 import TARGET_SPECS, build_model, inverse_transform_series, logmel_frontend

MODEL_NAME = "model_training_big13_fmsynth3_0_1"
MODEL_DIR = Path("model_training_big13_fmsynth3_0_1")
DEFAULT_AUDIO_DIR = Path("nsynth-test/audio")
DEFAULT_EXAMPLES_JSON = Path("nsynth-test/examples.json")
DEFAULT_META_JSON = Path("dataset_big13/meta.json")

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
    parser = argparse.ArgumentParser(description="Predict NSynth parameters with the big13 model.")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--examples-json", type=Path, default=DEFAULT_EXAMPLES_JSON)
    parser.add_argument("--meta-json", type=Path, default=DEFAULT_META_JSON)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR / "nsynth-pred-big13")
    return parser.parse_args()


def load_audio_batch(audio_dir: Path, expected_len: int) -> tuple[np.ndarray, list[str]]:
    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"Nenhum arquivo .wav encontrado em {audio_dir}")

    samples: list[np.ndarray] = []
    names: list[str] = []
    for wav_path in wav_files:
        signal, sr = sf.read(str(wav_path))
        if sr != 16000:
            raise ValueError(f"Sample rate inesperado em {wav_path}: {sr}")
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
        samples.append(signal)
        names.append(wav_path.name)

    x = np.asarray(samples, dtype=np.float32).reshape(len(samples), expected_len, 1)
    return x, names


def load_trained_model(model_path: Path, weights_path: Path, audio_len: int, n_algorithm_classes: int):
    try:
        from keras.models import load_model

        return load_model(model_path, compile=False, custom_objects={"logmel_frontend": logmel_frontend})
    except Exception as exc:
        print(f"Falha ao carregar `.keras` com serialização completa ({exc}); usando pesos diretamente.")
        model = build_model(audio_len, n_algorithm_classes)
        model.load_weights(str(weights_path))
        return model


def midi_to_hz(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((int(midi_note) - 69) / 12.0))


def main() -> None:
    args = parse_args()
    model_path = args.model_dir / f"{MODEL_NAME}.keras"
    weights_path = args.model_dir / "checkpoints" / "best.weights.h5"
    results_path = args.model_dir / "results.json"

    if not args.examples_json.exists():
        raise FileNotFoundError(f"examples.json não encontrado: {args.examples_json}")

    with open(args.examples_json, "r", encoding="utf-8") as f:
        examples = json.load(f)

    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        algorithm_classes = results.get("algorithm_classes") or []
    else:
        algorithm_classes = []

    if not algorithm_classes:
        algorithm_classes = sorted(pd.read_csv(args.model_dir / "y_train_big13.csv")["algorithm"].unique().tolist())

    scalers = {}
    for spec in TARGET_SPECS:
        scaler_path = args.model_dir / f"{spec['head']}_scaler.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler artifact: {scaler_path}")
        scalers[spec["head"]] = joblib.load(scaler_path)

    if model_path.exists():
        print(f"Carregando modelo: {model_path}")
        if not args.meta_json.exists():
            raise FileNotFoundError(f"meta.json não encontrado: {args.meta_json}")
        with open(args.meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        audio_len = int(meta["audio_sample_len"])
        model = load_trained_model(model_path, weights_path, audio_len, len(algorithm_classes))
    else:
        if not args.meta_json.exists():
            raise FileNotFoundError(f"meta.json não encontrado: {args.meta_json}")
        with open(args.meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        audio_len = int(meta["audio_sample_len"])
        print(f"Construindo arquitetura e carregando pesos: {weights_path}")
        model = build_model(audio_len, len(algorithm_classes))
        if not weights_path.exists():
            raise FileNotFoundError(f"Pesos não encontrados: {weights_path}")
        model.load_weights(str(weights_path))

    expected_len = int(model.input_shape[1])
    x, names = load_audio_batch(args.audio_dir, expected_len)
    print(f"Total de amostras: {len(names)}")
    print(f"Shape de entrada: {x.shape}")

    preds = model.predict(x, batch_size=args.batch_size, verbose=1)
    algo_pred = np.asarray(preds[0], dtype=np.float32)
    algo_idx = np.argmax(algo_pred, axis=1)
    algo_name = [algorithm_classes[i] for i in algo_idx]

    rows = []
    for idx, name in enumerate(names):
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
        }
        for spec_idx, spec in enumerate(TARGET_SPECS, start=1):
            scaler = scalers[spec["head"]]
            pred_scaled = np.asarray(preds[spec_idx], dtype=np.float32)
            pred_transformed = scaler.inverse_transform(pred_scaled[idx : idx + 1]).reshape(-1)
            pred_raw = inverse_transform_series(pred_transformed, spec["transform"])[0]
            out_key = "frequencia_base_pred" if spec["column"] == "frequencia_base" else spec["column"]
            clip_min, clip_max = CLIP_RANGES[out_key]
            row[out_key] = float(np.clip(pred_raw, clip_min, clip_max))
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
