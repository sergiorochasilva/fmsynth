"""Synthetic ablation study for the conditioned `big17` cascade.

Architecture:
- Loads the trained `big17` split models for `algorithm`, `pitch`, `timbre`, and `envelope`
- Replays the cascade on `dataset_big13` with controlled oracle substitutions
- Re-synthesizes selected synthetic samples with `FMSynth3`
- Measures FFT, STFT, and log-mel distances against the original rendered corpus audio

Data flow:
- Input: `dataset_big13/parameters.csv`, rendered audio shards, and trained `big17` checkpoints
- Output: mode-wise metric summaries, per-sample diagnostics, and optional listening pairs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

os.environ.pop("TARGET_GROUP", None)

from fm_synth3 import Envelope  # noqa: E402
from model_training_big13_fmsynth3_0_1 import (  # noqa: E402
    inverse_transform_series,
    normalize_audio_batch,
    stratified_split_indices,
    transform_series,
)
from model_training_big17_fmsynth3_0_1 import (  # noqa: E402
    BASE_PATH,
    GROUP_SPECS,
    build_model,
    group_stratify_key,
    load_audio_store,
)
from resynth_nsynth_big17_fmsynth3_0_1 import build_synth  # noqa: E402

SR_REF = 16000
DEFAULT_SAMPLE_COUNT = 256
DEFAULT_HOLDOUT_FRAC = 0.20
DEFAULT_EXPORT_AUDIO_COUNT = 10
MODEL_PREFIX = "model_training_big17_fmsynth3_0_1"
OUTPUT_DIR = Path("evaluation_big17_synthetic_ablation_0_1")
MODES = [
    "full_pred",
    "oracle_algorithm",
    "oracle_pitch",
    "oracle_algorithm_pitch",
    "oracle_timbre",
    "oracle_envelope",
    "oracle_all",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic ablation for the conditioned big17 cascade.")
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--holdout-frac", type=float, default=DEFAULT_HOLDOUT_FRAC)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict-batch-size", type=int, default=32)
    parser.add_argument("--export-audio-count", type=int, default=DEFAULT_EXPORT_AUDIO_COUNT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--modes", nargs="*", default=MODES, choices=MODES)
    return parser.parse_args()


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_json_dump(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def match_lengths(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = min(len(a), len(b))
    return a[:m], b[:m]


def build_prior_table(dataset_params: Path) -> dict[str, dict[str, float]]:
    frame = pd.read_csv(dataset_params)
    numeric_columns = [
        "ratio_carrier",
        "index_12",
        "index_23",
        "index_3c",
        "index_4c",
        "index_5c",
        "detune_carrier",
        "feedback",
        "lfo_rate",
        "lfo_depth_cents",
        "key_scaling",
        "env_mod_attack",
        "env_mod_decay",
        "env_mod_sustain",
        "env_mod_release",
        "env_car_attack",
        "env_car_decay",
        "env_car_sustain",
        "env_car_release",
    ]
    priors: dict[str, dict[str, float]] = {}
    for algorithm, group in frame.groupby("algorithm"):
        priors[str(algorithm)] = {column: float(group[column].median()) for column in numeric_columns}
    priors["_global"] = {column: float(frame[column].median()) for column in numeric_columns}
    return priors


def normalize_for_eval(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 0:
        arr = arr / peak
    return arr.astype(np.float32, copy=False)


def fft_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = match_lengths(normalize_for_eval(a), normalize_for_eval(b))
    return float(np.linalg.norm(np.fft.rfft(a) - np.fft.rfft(b)))


def stft_distance(a: np.ndarray, b: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> float:
    a, b = match_lengths(normalize_for_eval(a), normalize_for_eval(b))
    stft_a = np.abs(librosa.stft(y=a, n_fft=n_fft, hop_length=hop_length))
    stft_b = np.abs(librosa.stft(y=b, n_fft=n_fft, hop_length=hop_length))
    m = min(stft_a.shape[1], stft_b.shape[1])
    stft_a = stft_a[:, :m]
    stft_b = stft_b[:, :m]
    return float(np.linalg.norm(stft_a - stft_b))


def log_mel_distance(a: np.ndarray, b: np.ndarray, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128) -> tuple[float, float]:
    a, b = match_lengths(normalize_for_eval(a), normalize_for_eval(b))
    mel_a = librosa.feature.melspectrogram(y=a, sr=SR_REF, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_b = librosa.feature.melspectrogram(y=b, sr=SR_REF, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_a = np.log1p(mel_a)
    log_b = np.log1p(mel_b)
    m = min(log_a.shape[1], log_b.shape[1])
    log_a = log_a[:, :m]
    log_b = log_b[:, :m]
    raw = float(np.linalg.norm(log_a - log_b))
    norm = raw / float(log_a.size) if log_a.size else 0.0
    return raw, norm


def select_subset(frame: pd.DataFrame, sample_count: int, holdout_frac: float, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    indices = frame.index.to_numpy(dtype=np.int32)
    strata = group_stratify_key(frame, "pitch")
    _, holdout_idx = stratified_split_indices(indices, strata, test_size=holdout_frac, random_state=seed)
    holdout_frame = frame.loc[holdout_idx].reset_index(drop=True)
    holdout_indices = np.asarray(holdout_idx, dtype=np.int32)
    if sample_count <= 0 or sample_count >= len(holdout_frame):
        return holdout_frame, holdout_indices
    subset_indices = np.arange(len(holdout_frame), dtype=np.int32)
    candidate_strata = [
        group_stratify_key(holdout_frame, "pitch"),
        group_stratify_key(holdout_frame, "algorithm"),
    ]
    for subset_strata in candidate_strata:
        try:
            _, subset_idx = stratified_split_indices(
                subset_indices,
                subset_strata,
                test_size=sample_count / float(len(holdout_frame)),
                random_state=seed,
            )
            subset_idx = np.asarray(subset_idx, dtype=np.int32)
            return holdout_frame.iloc[subset_idx].reset_index(drop=True), holdout_indices[subset_idx]
        except ValueError:
            continue

    rng = np.random.default_rng(seed)
    subset_idx = rng.choice(subset_indices, size=sample_count, replace=False)
    subset_idx = np.asarray(np.sort(subset_idx), dtype=np.int32)
    return holdout_frame.iloc[subset_idx].reset_index(drop=True), holdout_indices[subset_idx]


def load_big17_bundle(group: str, audio_len: int, n_classes: int):
    model_dir = Path(f"{MODEL_PREFIX}_{group}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model directory: {model_dir}")

    model = build_model(audio_len, n_classes, target_group=group, target_specs=GROUP_SPECS[group])
    weights_path = model_dir / "checkpoints" / "best.weights.h5"
    if not weights_path.exists():
        weights_path = model_dir / "checkpoints" / "latest.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights for {group}: {weights_path}")
    model.load_weights(str(weights_path))

    scalers = {}
    for spec in GROUP_SPECS[group]:
        scaler_path = model_dir / f"{spec['head']}_scaler.joblib"
        if scaler_path.exists():
            scalers[spec["head"]] = joblib.load(scaler_path)

    condition_scalers = {}
    if group == "timbre":
        for spec in GROUP_SPECS["pitch"]:
            condition_scalers[spec["head"]] = joblib.load(model_dir / f"{spec['head']}_condition_scaler.joblib")
    elif group == "envelope":
        for spec in GROUP_SPECS["pitch"] + GROUP_SPECS["timbre"]:
            condition_scalers[spec["head"]] = joblib.load(model_dir / f"{spec['head']}_condition_scaler.joblib")

    return model, scalers, condition_scalers


def one_hot(indices: np.ndarray, n_classes: int) -> np.ndarray:
    return tf.keras.utils.to_categorical(np.asarray(indices, dtype=np.int32), num_classes=n_classes).astype(np.float32)


def build_condition_matrix(frame: pd.DataFrame, specs: list[dict], condition_scalers: dict[str, object]) -> np.ndarray:
    parts = []
    for spec in specs:
        transformed = transform_series(frame[spec["column"]], spec["transform"]).reshape(-1, 1)
        scaled = condition_scalers[spec["head"]].transform(transformed).astype(np.float32)
        parts.append(scaled)
    return np.concatenate(parts, axis=1).astype(np.float32)


def decode_regression_outputs(preds: list[np.ndarray], specs: list[dict], scalers: dict[str, object]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw_values: dict[str, np.ndarray] = {}
    transformed_values: dict[str, np.ndarray] = {}
    for pred, spec in zip(preds, specs):
        scaler = scalers[spec["head"]]
        pred_scaled = np.asarray(pred, dtype=np.float32)
        pred_transformed = scaler.inverse_transform(pred_scaled).reshape(-1).astype(np.float32)
        pred_raw = inverse_transform_series(pred_transformed, spec["transform"]).astype(np.float32)
        raw_values[spec["column"]] = pred_raw
        transformed_values[spec["column"]] = pred_transformed
    return raw_values, transformed_values


def build_row_from_mode(
    mode: str,
    idx: int,
    true_row: pd.Series,
    pred_algo_label: str,
    pred_pitch: dict[str, np.ndarray],
    pred_timbre: dict[str, np.ndarray],
    pred_envelope: dict[str, np.ndarray],
    true_pitch_raw: dict[str, np.ndarray],
    true_timbre_raw: dict[str, np.ndarray],
    true_envelope_raw: dict[str, np.ndarray],
) -> dict:
    row = {
        "algorithm": pred_algo_label,
        "ratio_carrier": float(pred_pitch["ratio_carrier"][idx]),
        "frequencia_base": float(pred_pitch["frequencia_base"][idx]),
        "index_12": float(pred_timbre["index_12"][idx]),
        "index_23": float(pred_timbre["index_23"][idx]),
        "index_3c": float(pred_timbre["index_3c"][idx]),
        "index_4c": float(pred_timbre["index_4c"][idx]),
        "index_5c": float(pred_timbre["index_5c"][idx]),
        "detune_carrier": float(pred_timbre["detune_carrier"][idx]),
        "feedback": float(pred_timbre["feedback"][idx]),
        "lfo_rate": float(pred_timbre["lfo_rate"][idx]),
        "lfo_depth_cents": float(pred_timbre["lfo_depth_cents"][idx]),
        "key_scaling": float(pred_timbre["key_scaling"][idx]),
        "env_mod_attack": float(pred_envelope["env_mod_attack"][idx]),
        "env_mod_decay": float(pred_envelope["env_mod_decay"][idx]),
        "env_mod_sustain": float(pred_envelope["env_mod_sustain"][idx]),
        "env_mod_release": float(pred_envelope["env_mod_release"][idx]),
        "env_car_attack": float(pred_envelope["env_car_attack"][idx]),
        "env_car_decay": float(pred_envelope["env_car_decay"][idx]),
        "env_car_sustain": float(pred_envelope["env_car_sustain"][idx]),
        "env_car_release": float(pred_envelope["env_car_release"][idx]),
    }

    if mode == "oracle_algorithm":
        row["algorithm"] = str(true_row["algorithm"])
    elif mode == "oracle_pitch":
        row["ratio_carrier"] = float(true_pitch_raw["ratio_carrier"][idx])
        row["frequencia_base"] = float(true_pitch_raw["frequencia_base"][idx])
    elif mode == "oracle_algorithm_pitch":
        row["algorithm"] = str(true_row["algorithm"])
        row["ratio_carrier"] = float(true_pitch_raw["ratio_carrier"][idx])
        row["frequencia_base"] = float(true_pitch_raw["frequencia_base"][idx])
    elif mode == "oracle_timbre":
        for key in true_timbre_raw:
            row[key] = float(true_timbre_raw[key][idx])
    elif mode == "oracle_envelope":
        for key in true_envelope_raw:
            row[key] = float(true_envelope_raw[key][idx])
    elif mode == "oracle_all":
        row["algorithm"] = str(true_row["algorithm"])
        row["ratio_carrier"] = float(true_pitch_raw["ratio_carrier"][idx])
        row["frequencia_base"] = float(true_pitch_raw["frequencia_base"][idx])
        for key in true_timbre_raw:
            row[key] = float(true_timbre_raw[key][idx])
        for key in true_envelope_raw:
            row[key] = float(true_envelope_raw[key][idx])

    return row


def synthesize_row(row: dict, priors: dict[str, dict[str, float]]) -> np.ndarray:
    synth = build_synth(row, priors)
    audio = synth.synth(audio_seconds=4.0, frequency_carrier=float(row["frequencia_base"]), algorithm=str(row["algorithm"]))
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size:
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = 0.891 * audio / peak
    return audio.astype(np.float32, copy=False)


def metrics_for_pair(original: np.ndarray, resynth: np.ndarray) -> dict[str, float]:
    fft = fft_distance(original, resynth)
    stft = stft_distance(original, resynth)
    mel_raw, mel_norm = log_mel_distance(original, resynth)
    return {
        "fft_distance": fft,
        "stft_distance": stft,
        "log_mel_spectrogram_distance_raw": mel_raw,
        "log_mel_spectrogram_distance_normalized": mel_norm,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = args.output_dir / "audio_pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    params_path = BASE_PATH / "parameters.csv"
    frame = pd.read_csv(params_path)
    if "id" not in frame.columns:
        frame = frame.reset_index().rename(columns={"index": "id"})
    frame["algorithm"] = frame["algorithm"].astype(str)

    subset_frame, subset_indices = select_subset(frame, args.sample_count, args.holdout_frac, args.seed)
    sample_ids = subset_frame["id"].astype(int).tolist()

    audio_store = load_audio_store(BASE_PATH)
    original_audio = np.asarray(audio_store[sample_ids], dtype=np.int16)
    original_audio = normalize_audio_batch(original_audio).astype(np.float32)
    audio_input = original_audio.reshape(len(sample_ids), original_audio.shape[1], 1)

    algorithm_dir = Path(f"{MODEL_PREFIX}_algorithm")
    algorithm_map = load_json(algorithm_dir / "algorithm_map.json")
    algorithm_classes = algorithm_map["classes"]
    algorithm_to_idx = {name: idx for idx, name in enumerate(algorithm_classes)}
    n_algorithm_classes = len(algorithm_classes)

    algorithm_model, _, _ = load_big17_bundle("algorithm", int(original_audio.shape[1]), n_algorithm_classes)
    algo_probs = np.asarray(algorithm_model.predict(audio_input, batch_size=args.predict_batch_size, verbose=0), dtype=np.float32)
    algo_pred_idx = np.argmax(algo_probs, axis=1)
    algo_pred_labels = [algorithm_classes[idx] for idx in algo_pred_idx]
    algo_pred_onehot = one_hot(algo_pred_idx, n_algorithm_classes)
    true_algorithm_idx = np.asarray([algorithm_to_idx[a] for a in subset_frame["algorithm"].tolist()], dtype=np.int32)
    true_algorithm_onehot = one_hot(true_algorithm_idx, n_algorithm_classes)

    pitch_model, pitch_scalers, pitch_condition_scalers = load_big17_bundle("pitch", int(original_audio.shape[1]), n_algorithm_classes)
    timbre_model, timbre_scalers, timbre_condition_scalers = load_big17_bundle("timbre", int(original_audio.shape[1]), n_algorithm_classes)
    env_model, env_scalers, env_condition_scalers = load_big17_bundle("envelope", int(original_audio.shape[1]), n_algorithm_classes)

    priors = build_prior_table(params_path)
    true_pitch_raw = {
        "ratio_carrier": subset_frame["ratio_carrier"].to_numpy(dtype=np.float32),
        "frequencia_base": subset_frame["frequencia_base"].to_numpy(dtype=np.float32),
    }
    true_timbre_raw = {
        key: subset_frame[key].to_numpy(dtype=np.float32)
        for key in ["index_12", "index_23", "index_3c", "index_4c", "index_5c", "detune_carrier", "feedback", "lfo_rate", "lfo_depth_cents", "key_scaling"]
    }
    true_envelope_raw = {
        key: subset_frame[key].to_numpy(dtype=np.float32)
        for key in ["env_mod_attack", "env_mod_decay", "env_mod_sustain", "env_mod_release", "env_car_attack", "env_car_decay", "env_car_sustain", "env_car_release"]
    }

    pitch_specs = GROUP_SPECS["pitch"]
    timbre_specs = GROUP_SPECS["timbre"]
    envelope_specs = GROUP_SPECS["envelope"]

    summary_rows = []
    per_sample_rows = []
    rendered_examples_per_mode: dict[str, int] = {mode: 0 for mode in args.modes}

    for mode in args.modes:
        if mode in {"oracle_algorithm", "oracle_algorithm_pitch", "oracle_all"}:
            pitch_algo_cond = true_algorithm_onehot
            timbre_algo_cond = true_algorithm_onehot
            env_algo_cond = true_algorithm_onehot
        else:
            pitch_algo_cond = algo_pred_onehot
            timbre_algo_cond = algo_pred_onehot
            env_algo_cond = algo_pred_onehot

        pitch_preds = pitch_model.predict(
            {"audio_input": audio_input, "algorithm_condition_input": pitch_algo_cond},
            batch_size=args.predict_batch_size,
            verbose=0,
        )
        pred_pitch_raw, _ = decode_regression_outputs(pitch_preds, pitch_specs, pitch_scalers)

        pitch_for_timbre_frame = pd.DataFrame(
            {
                "ratio_carrier": true_pitch_raw["ratio_carrier"] if mode in {"oracle_pitch", "oracle_algorithm_pitch", "oracle_all"} else pred_pitch_raw["ratio_carrier"],
                "frequencia_base": true_pitch_raw["frequencia_base"] if mode in {"oracle_pitch", "oracle_algorithm_pitch", "oracle_all"} else pred_pitch_raw["frequencia_base"],
            }
        )
        timbre_pitch_cond = build_condition_matrix(pitch_for_timbre_frame, pitch_specs, timbre_condition_scalers)

        timbre_preds = timbre_model.predict(
            {
                "audio_input": audio_input,
                "algorithm_condition_input": timbre_algo_cond,
                "pitch_condition_input": timbre_pitch_cond,
            },
            batch_size=args.predict_batch_size,
            verbose=0,
        )
        pred_timbre_raw, _ = decode_regression_outputs(timbre_preds, timbre_specs, timbre_scalers)

        env_pitch_frame = pd.DataFrame(
            {
                "ratio_carrier": true_pitch_raw["ratio_carrier"] if mode in {"oracle_pitch", "oracle_algorithm_pitch", "oracle_all"} else pred_pitch_raw["ratio_carrier"],
                "frequencia_base": true_pitch_raw["frequencia_base"] if mode in {"oracle_pitch", "oracle_algorithm_pitch", "oracle_all"} else pred_pitch_raw["frequencia_base"],
            }
        )
        env_timbre_frame = pd.DataFrame(
            {
                key: true_timbre_raw[key] if mode in {"oracle_timbre", "oracle_all"} else pred_timbre_raw[key]
                for key in true_timbre_raw
            }
        )
        env_pitch_cond = build_condition_matrix(env_pitch_frame, pitch_specs, env_condition_scalers)
        env_timbre_cond = build_condition_matrix(env_timbre_frame, timbre_specs, env_condition_scalers)

        env_preds = env_model.predict(
            {
                "audio_input": audio_input,
                "algorithm_condition_input": env_algo_cond,
                "pitch_condition_input": env_pitch_cond,
                "timbre_condition_input": env_timbre_cond,
            },
            batch_size=args.predict_batch_size,
            verbose=0,
        )
        pred_env_raw, _ = decode_regression_outputs(env_preds, envelope_specs, env_scalers)

        mode_rows = []
        for idx, sample_id in enumerate(sample_ids):
            true_row = subset_frame.iloc[idx]
            pred_row = build_row_from_mode(
                mode,
                idx,
                true_row,
                algo_pred_labels[idx],
                pred_pitch_raw,
                pred_timbre_raw,
                pred_env_raw,
                true_pitch_raw,
                true_timbre_raw,
                true_envelope_raw,
            )
            synth_audio = synthesize_row(pred_row, priors)
            metrics = metrics_for_pair(original_audio[idx], synth_audio)
            record = {
                "mode": mode,
                "sample_id": int(sample_id),
                "algorithm_true": str(true_row["algorithm"]),
                "algorithm_pred": str(algo_pred_labels[idx]),
                "algorithm_correct": bool(algo_pred_labels[idx] == str(true_row["algorithm"])),
                "ratio_true": float(true_pitch_raw["ratio_carrier"][idx]),
                "ratio_pred": float(pred_pitch_raw["ratio_carrier"][idx]),
                "freq_true": float(true_pitch_raw["frequencia_base"][idx]),
                "freq_pred": float(pred_pitch_raw["frequencia_base"][idx]),
                **metrics,
            }
            per_sample_rows.append(record)
            mode_rows.append(record)

            if rendered_examples_per_mode[mode] < args.export_audio_count:
                mode_dir = pairs_dir / mode
                mode_dir.mkdir(parents=True, exist_ok=True)
                sf.write(str(mode_dir / f"sample_{sample_id:05d}_original.wav"), original_audio[idx].astype(np.float32), SR_REF)
                sf.write(str(mode_dir / f"sample_{sample_id:05d}_resynth.wav"), synth_audio.astype(np.float32), SR_REF)
                rendered_examples_per_mode[mode] += 1

        mode_df = pd.DataFrame(mode_rows)
        summary_rows.append(
            {
                "mode": mode,
                "sample_count": int(len(mode_df)),
                "algorithm_accuracy": float(mode_df["algorithm_correct"].mean()),
                "ratio_mae": float(np.mean(np.abs(mode_df["ratio_true"] - mode_df["ratio_pred"]))),
                "freq_mae": float(np.mean(np.abs(mode_df["freq_true"] - mode_df["freq_pred"]))),
                "fft_distance_mean": float(mode_df["fft_distance"].mean()),
                "stft_distance_mean": float(mode_df["stft_distance"].mean()),
                "log_mel_spectrogram_distance_raw_mean": float(mode_df["log_mel_spectrogram_distance_raw"].mean()),
                "log_mel_spectrogram_distance_normalized_mean": float(mode_df["log_mel_spectrogram_distance_normalized"].mean()),
            }
        )

    per_sample_df = pd.DataFrame(per_sample_rows)
    per_sample_df.to_csv(args.output_dir / "per_sample_metrics.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)

    results = {
        "model_name": MODEL_PREFIX,
        "dataset": str(BASE_PATH),
        "sample_count": int(len(subset_frame)),
        "holdout_frac": float(args.holdout_frac),
        "seed": int(args.seed),
        "modes": args.modes,
        "summary": summary_rows,
        "sample_ids": [int(x) for x in sample_ids],
        "algorithm_classes": algorithm_classes,
    }
    atomic_json_dump(args.output_dir / "results.json", results)

    print(summary_df.to_string(index=False))
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
