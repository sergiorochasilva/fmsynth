"""Predict FM parameters for NSynth using `model_training_big12_fmsynth3_0_1`.

Architecture:
- Loads the trained multi-task CNN from the `big12` experiment
- Preprocesses NSynth audio into the expected waveform tensor
- Exports predicted `algorithm`, `ratio_carrier`, and `frequencia_base`

Data flow:
- Input: `nsynth-test/audio` and the saved `model_training_big12_fmsynth3_0_1.keras`
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

from model_training_big12_fmsynth3_0_1 import build_model, logmel_frontend

MODEL_NAME = "model_training_big12_fmsynth3_0_1"
MODEL_DIR = Path("model_training_big12_fmsynth3_0_1")
DEFAULT_AUDIO_DIR = Path("nsynth-test/audio")
DEFAULT_EXAMPLES_JSON = Path("nsynth-test/examples.json")
DEFAULT_META_JSON = Path("dataset_big12/meta.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict NSynth parameters with the big12 model.")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--examples-json", type=Path, default=DEFAULT_EXAMPLES_JSON)
    parser.add_argument("--meta-json", type=Path, default=DEFAULT_META_JSON)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR / "nsynth-pred")
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
    ratio_scaler_path = args.model_dir / "ratio_log2_scaler.joblib"
    freq_scaler_path = args.model_dir / "freq_log2_scaler.joblib"
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
        algorithm_classes = sorted(pd.read_csv(args.model_dir / "y_train_big12.csv")["algorithm"].unique().tolist())

    if not ratio_scaler_path.exists() or not freq_scaler_path.exists():
        raise FileNotFoundError(
            f"Missing scalers: {ratio_scaler_path} / {freq_scaler_path}. "
            "The prediction script requires the saved StandardScaler artifacts."
        )
    ratio_scaler = joblib.load(ratio_scaler_path)
    freq_scaler = joblib.load(freq_scaler_path)

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
    ratio_log2_pred = ratio_scaler.inverse_transform(np.asarray(preds[1], dtype=np.float32)).reshape(-1)
    freq_log2_pred = freq_scaler.inverse_transform(np.asarray(preds[2], dtype=np.float32)).reshape(-1)

    algo_idx = np.argmax(algo_pred, axis=1)
    algo_name = [algorithm_classes[i] for i in algo_idx]
    ratio_pred = np.power(2.0, ratio_log2_pred)
    freq_pred = np.power(2.0, freq_log2_pred)

    rows = []
    for idx, name in enumerate(names):
        note_key = name[:-4]
        if note_key not in examples or "pitch" not in examples[note_key]:
            continue
        rows.append(
            {
                "audio_file": name,
                "note_key": note_key,
                "pitch": int(examples[note_key]["pitch"]),
                "pitch_hz": midi_to_hz(int(examples[note_key]["pitch"])),
                "algorithm_idx": int(algo_idx[idx]),
                "algorithm": algo_name[idx],
                "algorithm_prob": float(np.max(algo_pred[idx])),
                "ratio_carrier": float(ratio_pred[idx]),
                "frequencia_base_pred": float(freq_pred[idx]),
            }
        )

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
