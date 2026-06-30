"""Resynthesize NSynth audio from the conditioned split `big15` `0_2` predictions.

Architecture:
- Loads the merged predictions from the three split `big15` `0_2` models
- Reconstructs the waveform with `FMSynth3`
- Uses dataset priors only for any missing controls, so the split models remain the primary source of control

Data flow:
- Input: predicted parameter JSON and NSynth metadata
- Output: resynthesized `.wav` files in `nsynth-pred-big15_0_2/`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from fm_synth3 import Envelope, FMSynth3

MODEL_NAME = "model_training_big15_fmsynth3_0_2"
DEFAULT_PRED_JSON = Path("nsynth-pred-big15_0_2") / f"params_pred_nsynth_{MODEL_NAME}.json"
DEFAULT_NSYNTH_AUDIO_DIR = Path("nsynth-test/audio")
DEFAULT_EXAMPLES_JSON = Path("nsynth-test/examples.json")
DEFAULT_OUTPUT_DIR = Path("nsynth-pred-big15_0_2")
DEFAULT_DATASET_PARAMS = Path("dataset_big13/parameters.csv")

FIXED_RATIO_1 = 1.0
FIXED_RATIO_2 = 1.5
FIXED_RATIO_3 = 2.0
FIXED_RATIO_4 = 3.0
FIXED_RATIO_5 = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resynthesize NSynth using split big15 predictions.")
    parser.add_argument("--pred-json", type=Path, default=DEFAULT_PRED_JSON)
    parser.add_argument("--examples-json", type=Path, default=DEFAULT_EXAMPLES_JSON)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_NSYNTH_AUDIO_DIR)
    parser.add_argument("--dataset-params", type=Path, default=DEFAULT_DATASET_PARAMS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audio-seconds", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-predicted-fc", action="store_true")
    return parser.parse_args()


def midi_to_hz(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((int(midi_note) - 69) / 12.0))


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prior_table(dataset_params: Path) -> dict[str, dict[str, float]]:
    df = pd.read_csv(dataset_params)
    priors: dict[str, dict[str, float]] = {}
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
    for algorithm, group in df.groupby("algorithm"):
        priors[str(algorithm)] = {column: float(group[column].median()) for column in numeric_columns}
    priors["_global"] = {column: float(df[column].median()) for column in numeric_columns}
    return priors


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def build_synth(row: dict, priors: dict[str, dict[str, float]]) -> FMSynth3:
    algorithm = str(row.get("algorithm", "series3_parallel2"))
    prior = priors.get(algorithm, priors["_global"])
    env_mod = Envelope(
        to_float(row.get("env_mod_attack"), prior["env_mod_attack"]),
        to_float(row.get("env_mod_decay"), prior["env_mod_decay"]),
        to_float(row.get("env_mod_sustain"), prior["env_mod_sustain"]),
        to_float(row.get("env_mod_release"), prior["env_mod_release"]),
        "exp",
        "exp",
        "exp",
    )
    env_car = Envelope(
        to_float(row.get("env_car_attack"), prior["env_car_attack"]),
        to_float(row.get("env_car_decay"), prior["env_car_decay"]),
        to_float(row.get("env_car_sustain"), prior["env_car_sustain"]),
        to_float(row.get("env_car_release"), prior["env_car_release"]),
        "exp",
        "exp",
        "exp",
    )
    return FMSynth3(
        ratio1=FIXED_RATIO_1,
        ratio2=FIXED_RATIO_2,
        ratio3=FIXED_RATIO_3,
        ratio4=FIXED_RATIO_4,
        ratio5=FIXED_RATIO_5,
        ratio_carrier=to_float(row.get("ratio_carrier"), 1.0),
        detune1=0.0,
        detune2=0.0,
        detune3=0.0,
        detune4=0.0,
        detune5=0.0,
        detune_carrier=to_float(row.get("detune_carrier"), prior["detune_carrier"]),
        index_12=to_float(row.get("index_12"), prior["index_12"]),
        index_23=to_float(row.get("index_23"), prior["index_23"]),
        index_3c=to_float(row.get("index_3c"), prior["index_3c"]),
        index_4c=to_float(row.get("index_4c"), prior["index_4c"]),
        index_5c=to_float(row.get("index_5c"), prior["index_5c"]),
        env_mod=env_mod,
        env_car=env_car,
        env_scale1=1.0,
        env_scale2=1.0,
        env_scale3=1.0,
        env_scale4=1.0,
        env_scale5=1.0,
        env_scale_carrier=1.0,
        carrier_level=1.0,
        feedback=to_float(row.get("feedback"), prior["feedback"]),
        lfo_rate=to_float(row.get("lfo_rate"), prior["lfo_rate"]),
        lfo_depth_cents=to_float(row.get("lfo_depth_cents"), prior["lfo_depth_cents"]),
        key_scaling=to_float(row.get("key_scaling"), prior["key_scaling"]),
        key_scaling_ref_hz=440.0,
        random_phase=False,
        phase1=0.0,
        phase2=0.0,
        phase3=0.0,
        phase4=0.0,
        phase5=0.0,
        phase_carrier=0.0,
        downsample_16k=True,
    )


def main() -> None:
    args = parse_args()
    pred_rows = load_json(args.pred_json)
    examples = load_json(args.examples_json)
    priors = build_prior_table(args.dataset_params)
    fallback_audio = sorted([p.name for p in args.audio_dir.glob("*.wav")])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for idx, row in enumerate(pred_rows):
        audio_file = str(row.get("audio_file") or (fallback_audio[idx] if idx < len(fallback_audio) else f"sample_{idx}.wav"))
        audio_file = Path(audio_file).name
        out_path = args.output_dir / audio_file
        if out_path.exists() and not args.overwrite:
            continue

        note_key = audio_file.replace(".wav", "")
        if note_key not in examples:
            continue

        synth = build_synth(row, priors)
        if args.keep_predicted_fc:
            fc = to_float(row.get("frequencia_base_pred"), midi_to_hz(int(examples[note_key]["pitch"])))
        else:
            fc = midi_to_hz(int(examples[note_key]["pitch"]))

        audio = synth.synth(audio_seconds=args.audio_seconds, frequency_carrier=fc, algorithm=str(row.get("algorithm", "series3_parallel2")))
        if audio.size:
            peak = float(np.max(np.abs(audio)))
            if peak > 0:
                audio = 0.891 * audio / peak
        sf.write(str(out_path), audio.astype(np.float32), 16000)
        rendered += 1

    print(f"Rendered {rendered} files into {args.output_dir}")


if __name__ == "__main__":
    main()
