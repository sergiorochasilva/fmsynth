"""Resynthesize `dataset_big4` audio from predicted FM parameters.

Architecture:
- Loads predicted parameter tables
- Projects parameters into a feasible FM range
- Uses `FMSynth3` to render the reconstructed waveform

Data flow:
- Input: predicted parameters and `dataset_big4` metadata
- Output: synthesized `.wav` files plus comparison metrics
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from fm_synth3 import Envelope, FMSynth3, SAMPLE_RATE_OUT, SAMPLE_RATE_RENDER

MODEL_NAME = "model_training_big4_fmsynth3_0_4"
MODEL_DIR = Path("model_training_big4_fmsynth3_0_4")

DEFAULT_PRED_JSON = MODEL_DIR / f"params_pred_nsynth_{MODEL_NAME}.json"
DEFAULT_RESULTS_JSON = MODEL_DIR / "results.json"
DEFAULT_NSYNTH_EXAMPLES = Path("nsynth-test/examples.json")
DEFAULT_NSYNTH_AUDIO_DIR = Path("nsynth-test/audio")
DEFAULT_OUTPUT_DIR = MODEL_DIR / "nsynth-pred"

AUDIO_SECONDS = 4.0

ALGORITHMS_FALLBACK = (
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
CURVES_ALLOWED = {"linear", "exp", "log", "s_curve"}
RATIOS_DISCRETOS = (
    1 / 8,
    1 / 6,
    1 / 5,
    1 / 4,
    1 / 3,
    1 / 2,
    2 / 3,
    1.0,
    3 / 2,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Resintetiza o NSynth usando parâmetros previstos pelo modelo "
            "model_training_big4_fmsynth3_0_4 e o sintetizador fm_synth3."
        )
    )
    parser.add_argument(
        "--pred-json",
        type=Path,
        default=DEFAULT_PRED_JSON,
        help="Arquivo JSON de parâmetros preditos gerado pelo script de predição.",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=DEFAULT_RESULTS_JSON,
        help="results.json do treino (usado para mapas categóricos).",
    )
    parser.add_argument(
        "--examples-json",
        type=Path,
        default=DEFAULT_NSYNTH_EXAMPLES,
        help="Arquivo examples.json do NSynth (usado para recuperar pitch).",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_NSYNTH_AUDIO_DIR,
        help="Diretório de áudio do NSynth (fallback para nomes de arquivos).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Diretório de saída dos áudios resintetizados.",
    )
    parser.add_argument(
        "--audio-seconds",
        type=float,
        default=AUDIO_SECONDS,
        help="Duração de cada áudio resintetizado em segundos.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita a quantidade de amostras processadas (útil para teste).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescreve arquivos de saída existentes.",
    )
    parser.add_argument(
        "--keep-predicted-fc",
        action="store_true",
        help="Não sobrescreve frequencia_base com pitch do NSynth.",
    )
    return parser.parse_args()


def clamp(value: float, min_value: float, max_value: float) -> float:
    return float(np.clip(value, min_value, max_value))


def to_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def to_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(round(float(value)))
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y"}:
            return True
        if token in {"0", "false", "no", "n"}:
            return False
    return default


def normalize_curve(value: str, default: str = "exp") -> str:
    token = str(value).strip().lower()
    if token in {"s", "smooth"}:
        token = "s_curve"
    if token in {"logarithmic"}:
        token = "log"
    if token not in CURVES_ALLOWED:
        return default
    return token


def wrap_phase(value: float) -> float:
    return float(value % (2.0 * math.pi))


def quantize_to_set(value: float, choices: tuple[float, ...]) -> float:
    x = max(float(value), 1e-9)
    return min(choices, key=lambda c: abs(math.log(x) - math.log(c)))


def quantize_to_ratios_with_upper(value: float, upper: float) -> float:
    # Mantém o comportamento "musical" sem violar faixa segura de frequência.
    filtered = tuple(r for r in RATIOS_DISCRETOS if r <= upper + 1e-12)
    if not filtered:
        return upper
    return quantize_to_set(value, filtered)


def midi_to_hz(midi_note: int) -> float:
    # Mantém a mesma convenção usada no resynth_nsynth.py original.
    return 440.0 * (2.0 ** ((int(midi_note) - 69) / 12.0))


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_categorical_maps(results_json: Path) -> dict[str, list[str]]:
    if not results_json.exists():
        return {}
    data = load_json(results_json)
    raw = data.get("categorical_maps", {})
    out: dict[str, list[str]] = {}
    for key, values in raw.items():
        if isinstance(values, list) and values:
            out[str(key)] = [str(v) for v in values]
    return out


def decode_category(row: dict, column: str, categories: list[str], fallback: str) -> str:
    decoded_key = f"{column}_decoded"
    direct_decoded = row.get(decoded_key)
    if isinstance(direct_decoded, str) and direct_decoded in categories:
        return direct_decoded

    direct_value = row.get(column)
    if isinstance(direct_value, str) and direct_value in categories:
        return direct_value

    idx = int(round(to_float(direct_value, 0.0)))
    idx = int(np.clip(idx, 0, len(categories) - 1))
    return categories[idx] if categories else fallback


def build_env(row: dict, prefix: str, cat_maps: dict[str, list[str]]) -> Envelope:
    curve_attack_col = f"{prefix}_curve_attack"
    curve_decay_col = f"{prefix}_curve_decay"
    curve_release_col = f"{prefix}_curve_release"

    attack = clamp(to_float(row.get(f"{prefix}_attack"), 0.01), 0.001, 0.5)
    decay = clamp(to_float(row.get(f"{prefix}_decay"), 0.2), 0.01, 0.8)
    sustain = clamp(to_float(row.get(f"{prefix}_sustain"), 0.5), 0.05, 0.95)
    release = clamp(to_float(row.get(f"{prefix}_release"), 0.2), 0.02, 1.0)

    if curve_attack_col in cat_maps:
        curve_attack = decode_category(
            row, curve_attack_col, cat_maps[curve_attack_col], "exp"
        )
    else:
        curve_attack = str(row.get(f"{curve_attack_col}_decoded", row.get(curve_attack_col, "exp")))

    if curve_decay_col in cat_maps:
        curve_decay = decode_category(row, curve_decay_col, cat_maps[curve_decay_col], "exp")
    else:
        curve_decay = str(row.get(f"{curve_decay_col}_decoded", row.get(curve_decay_col, "exp")))

    if curve_release_col in cat_maps:
        curve_release = decode_category(
            row, curve_release_col, cat_maps[curve_release_col], "exp"
        )
    else:
        curve_release = str(row.get(f"{curve_release_col}_decoded", row.get(curve_release_col, "exp")))

    return Envelope(
        attack=attack,
        decay=decay,
        sustain=sustain,
        release=release,
        curve_attack=normalize_curve(curve_attack, "exp"),
        curve_decay=normalize_curve(curve_decay, "exp"),
        curve_release=normalize_curve(curve_release, "exp"),
    )


def build_params(row: dict, cat_maps: dict[str, list[str]]) -> dict:
    if "style" in cat_maps:
        styles = [s for s in cat_maps["style"] if s in {"musical", "random"}]
        if not styles:
            styles = ["musical", "random"]
    else:
        styles = ["musical", "random"]

    style = decode_category(
        row=row,
        column="style",
        categories=styles,
        fallback="musical",
    )

    if "algorithm" in cat_maps:
        algorithms = [a for a in cat_maps["algorithm"] if a in ALGORITHMS_FALLBACK]
        if not algorithms:
            algorithms = list(ALGORITHMS_FALLBACK)
    else:
        algorithms = list(ALGORITHMS_FALLBACK)

    algorithm = decode_category(
        row=row,
        column="algorithm",
        categories=algorithms,
        fallback="series3_parallel2",
    )

    params = {
        "style": style,
        "algorithm": algorithm,
        "frequencia_base": clamp(to_float(row.get("frequencia_base"), 440.0), 20.0, 6000.0),
        "ratio1": clamp(to_float(row.get("ratio1"), 1.0), 1.0 / 8.0, 8.0),
        "ratio2": clamp(to_float(row.get("ratio2"), 1.0), 1.0 / 8.0, 8.0),
        "ratio3": clamp(to_float(row.get("ratio3"), 1.0), 1.0 / 8.0, 8.0),
        "ratio4": clamp(to_float(row.get("ratio4"), 1.0), 1.0 / 8.0, 8.0),
        "ratio5": clamp(to_float(row.get("ratio5"), 1.0), 1.0 / 8.0, 8.0),
        "ratio_carrier": clamp(to_float(row.get("ratio_carrier"), 1.0), 1.0 / 8.0, 8.0),
        "detune1": clamp(to_float(row.get("detune1"), 0.0), -30.0, 30.0),
        "detune2": clamp(to_float(row.get("detune2"), 0.0), -30.0, 30.0),
        "detune3": clamp(to_float(row.get("detune3"), 0.0), -30.0, 30.0),
        "detune4": clamp(to_float(row.get("detune4"), 0.0), -30.0, 30.0),
        "detune5": clamp(to_float(row.get("detune5"), 0.0), -30.0, 30.0),
        "detune_carrier": clamp(to_float(row.get("detune_carrier"), 0.0), -30.0, 30.0),
        "index_12": clamp(to_float(row.get("index_12"), 0.0), 0.0, 8.0),
        "index_23": clamp(to_float(row.get("index_23"), 0.0), 0.0, 8.0),
        "index_3c": clamp(to_float(row.get("index_3c"), 0.0), 0.0, 8.0),
        "index_4c": clamp(to_float(row.get("index_4c"), 0.0), 0.0, 8.0),
        "index_5c": clamp(to_float(row.get("index_5c"), 0.0), 0.0, 8.0),
        "env_scale1": clamp(to_float(row.get("env_scale1"), 1.0), 0.2, 1.2),
        "env_scale2": clamp(to_float(row.get("env_scale2"), 1.0), 0.2, 1.2),
        "env_scale3": clamp(to_float(row.get("env_scale3"), 1.0), 0.2, 1.2),
        "env_scale4": clamp(to_float(row.get("env_scale4"), 1.0), 0.2, 1.2),
        "env_scale5": clamp(to_float(row.get("env_scale5"), 1.0), 0.2, 1.2),
        "env_scale_carrier": clamp(to_float(row.get("env_scale_carrier"), 1.0), 0.2, 1.2),
        "carrier_level": clamp(to_float(row.get("carrier_level"), 0.8), 0.4, 1.0),
        "feedback": clamp(to_float(row.get("feedback"), 0.0), 0.0, 0.8),
        "lfo_rate": clamp(to_float(row.get("lfo_rate"), 0.0), 0.0, 8.0),
        "lfo_depth_cents": clamp(to_float(row.get("lfo_depth_cents"), 0.0), 0.0, 20.0),
        "key_scaling": clamp(to_float(row.get("key_scaling"), 0.0), -0.8, 0.4),
        "key_scaling_ref_hz": clamp(to_float(row.get("key_scaling_ref_hz"), 440.0), 20.0, 20000.0),
        "random_phase": to_bool(row.get("random_phase"), False),
        "phase1": wrap_phase(to_float(row.get("phase1"), 0.0)),
        "phase2": wrap_phase(to_float(row.get("phase2"), 0.0)),
        "phase3": wrap_phase(to_float(row.get("phase3"), 0.0)),
        "phase4": wrap_phase(to_float(row.get("phase4"), 0.0)),
        "phase5": wrap_phase(to_float(row.get("phase5"), 0.0)),
        "phase_carrier": wrap_phase(to_float(row.get("phase_carrier"), 0.0)),
        "downsample_16k": True,
    }

    params["env1"] = build_env(row, "env1", cat_maps)
    params["env2"] = build_env(row, "env2", cat_maps)
    params["env3"] = build_env(row, "env3", cat_maps)
    params["env4"] = build_env(row, "env4", cat_maps)
    params["env5"] = build_env(row, "env5", cat_maps)
    params["env_carrier"] = build_env(row, "env_carrier", cat_maps)

    return params


def project_params_to_generate_dataset4(params: dict) -> dict:
    # Limites base usados em generate_dataset4.py
    min_frequency = 20.0
    max_frequency = 6000.0
    min_ratio = 1.0 / 8.0
    max_ratio = 8.0
    min_index = 0.0
    max_index = 8.0
    feedback_range = (0.0, 0.8)
    lfo_rate_range = (0.5, 8.0)
    lfo_depth_range = (0.0, 20.0)

    style = params.get("style", "musical")
    if style not in {"musical", "random"}:
        style = "musical"

    params["frequencia_base"] = clamp(
        to_float(params.get("frequencia_base"), 440.0), min_frequency, max_frequency
    )

    # Faixas dependentes do estilo (mesmo padrão do gerador).
    if style == "musical":
        detune_min, detune_max = -15.0, 15.0
        env_scale_min, env_scale_max = 0.4, 1.1
        carrier_level_min, carrier_level_max = 0.6, 1.0
        key_scaling_min, key_scaling_max = -0.6, 0.2
    else:
        detune_min, detune_max = -30.0, 30.0
        env_scale_min, env_scale_max = 0.2, 1.2
        carrier_level_min, carrier_level_max = 0.4, 1.0
        key_scaling_min, key_scaling_max = -0.8, 0.4

    for key in ("detune1", "detune2", "detune3", "detune4", "detune5", "detune_carrier"):
        params[key] = clamp(to_float(params.get(key), 0.0), detune_min, detune_max)

    for key in ("index_12", "index_23", "index_3c", "index_4c", "index_5c"):
        params[key] = clamp(to_float(params.get(key), 0.0), min_index, max_index)

    for key in ("env_scale1", "env_scale2", "env_scale3", "env_scale4", "env_scale5", "env_scale_carrier"):
        params[key] = clamp(to_float(params.get(key), 1.0), env_scale_min, env_scale_max)

    params["carrier_level"] = clamp(
        to_float(params.get("carrier_level"), 0.8), carrier_level_min, carrier_level_max
    )
    params["feedback"] = clamp(
        to_float(params.get("feedback"), 0.0), feedback_range[0], feedback_range[1]
    )
    params["lfo_rate"] = clamp(
        to_float(params.get("lfo_rate"), 0.0), 0.0, lfo_rate_range[1]
    )
    params["lfo_depth_cents"] = clamp(
        to_float(params.get("lfo_depth_cents"), 0.0), lfo_depth_range[0], lfo_depth_range[1]
    )
    params["key_scaling"] = clamp(
        to_float(params.get("key_scaling"), 0.0), key_scaling_min, key_scaling_max
    )

    ratio_keys = ("ratio1", "ratio2", "ratio3", "ratio4", "ratio5", "ratio_carrier")
    for key in ratio_keys:
        params[key] = clamp(to_float(params.get(key), 1.0), min_ratio, max_ratio)

    # Padrão musical: aproximar para ratios discretas do dataset_big4.
    if style == "musical":
        for key in ratio_keys:
            params[key] = quantize_to_set(params[key], RATIOS_DISCRETOS)
        # No gerador musical, ratio_carrier é frequentemente 1.0.
        params["ratio_carrier"] = quantize_to_set(params["ratio_carrier"], RATIOS_DISCRETOS)

    # Reaplica a lógica de "valid_frequencies" do generate_dataset4 para o fc final.
    fc = max(params["frequencia_base"], 1e-9)
    safe_out = 0.9 * (SAMPLE_RATE_OUT / 2.0)
    safe_render = 0.45 * (SAMPLE_RATE_RENDER / 2.0)

    max_ratio_carrier = min(max_ratio, safe_out / fc)
    max_ratio_mod = min(max_ratio, safe_render / fc)

    max_ratio_carrier = max(max_ratio_carrier, 0.0)
    max_ratio_mod = max(max_ratio_mod, 0.0)

    if style == "musical":
        if max_ratio_carrier >= min_ratio:
            params["ratio_carrier"] = quantize_to_ratios_with_upper(
                params["ratio_carrier"], max_ratio_carrier
            )
        else:
            params["ratio_carrier"] = max_ratio_carrier

        for key in ("ratio1", "ratio2", "ratio3", "ratio4", "ratio5"):
            if max_ratio_mod >= min_ratio:
                params[key] = quantize_to_ratios_with_upper(params[key], max_ratio_mod)
            else:
                params[key] = max_ratio_mod
    else:
        if max_ratio_carrier >= min_ratio:
            params["ratio_carrier"] = clamp(
                params["ratio_carrier"], min_ratio, max_ratio_carrier
            )
        else:
            params["ratio_carrier"] = max_ratio_carrier

        for key in ("ratio1", "ratio2", "ratio3", "ratio4", "ratio5"):
            if max_ratio_mod >= min_ratio:
                params[key] = clamp(params[key], min_ratio, max_ratio_mod)
            else:
                params[key] = max_ratio_mod

    # Padrão musical do gerador: se não há vibrato, zera rate/depth juntos.
    if params["lfo_depth_cents"] < 1e-6 or params["lfo_rate"] < lfo_rate_range[0]:
        params["lfo_rate"] = 0.0
        params["lfo_depth_cents"] = 0.0

    return params


def main():
    args = parse_args()

    pred_rows = load_json(args.pred_json)
    if not isinstance(pred_rows, list) or not pred_rows:
        raise ValueError(f"Arquivo de predição inválido ou vazio: {args.pred_json}")

    cat_maps = load_categorical_maps(args.results_json)
    nsynth_examples = load_json(args.examples_json)

    fallback_audio_files = sorted([p.name for p in args.audio_dir.glob("*.wav")])

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = len(pred_rows)
    if args.limit is not None:
        total_rows = min(total_rows, max(args.limit, 0))
        pred_rows = pred_rows[:total_rows]

    print(f"Parâmetros de entrada: {len(pred_rows)}")
    print(f"Diretório de saída: {output_dir}")

    rendered = 0
    skipped = 0

    for idx, row in enumerate(pred_rows):
        audio_file = row.get("audio_file")
        if not isinstance(audio_file, str) or not audio_file.strip():
            if idx < len(fallback_audio_files):
                audio_file = fallback_audio_files[idx]
            else:
                audio_file = f"sample_{idx}.wav"

        audio_file = Path(audio_file).name
        out_path = output_dir / audio_file

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        params = build_params(row, cat_maps)

        if not args.keep_predicted_fc:
            note_key = audio_file.replace(".wav", "")
            if note_key in nsynth_examples and "pitch" in nsynth_examples[note_key]:
                params["frequencia_base"] = midi_to_hz(nsynth_examples[note_key]["pitch"])

        params = project_params_to_generate_dataset4(params)

        synth = FMSynth3(
            ratio1=params["ratio1"],
            ratio2=params["ratio2"],
            ratio3=params["ratio3"],
            ratio4=params["ratio4"],
            ratio5=params["ratio5"],
            ratio_carrier=params["ratio_carrier"],
            detune1=params["detune1"],
            detune2=params["detune2"],
            detune3=params["detune3"],
            detune4=params["detune4"],
            detune5=params["detune5"],
            detune_carrier=params["detune_carrier"],
            index_12=params["index_12"],
            index_23=params["index_23"],
            index_3c=params["index_3c"],
            index_4c=params["index_4c"],
            index_5c=params["index_5c"],
            env1=params["env1"],
            env2=params["env2"],
            env3=params["env3"],
            env4=params["env4"],
            env5=params["env5"],
            env_carrier=params["env_carrier"],
            env_scale1=params["env_scale1"],
            env_scale2=params["env_scale2"],
            env_scale3=params["env_scale3"],
            env_scale4=params["env_scale4"],
            env_scale5=params["env_scale5"],
            env_scale_carrier=params["env_scale_carrier"],
            carrier_level=params["carrier_level"],
            feedback=params["feedback"],
            lfo_rate=params["lfo_rate"],
            lfo_depth_cents=params["lfo_depth_cents"],
            key_scaling=params["key_scaling"],
            key_scaling_ref_hz=params["key_scaling_ref_hz"],
            random_phase=params["random_phase"],
            phase1=params["phase1"],
            phase2=params["phase2"],
            phase3=params["phase3"],
            phase4=params["phase4"],
            phase5=params["phase5"],
            phase_carrier=params["phase_carrier"],
            downsample_16k=True,
        )

        signal = synth.synth(
            audio_seconds=args.audio_seconds,
            frequency_carrier=params["frequencia_base"],
            algorithm=params["algorithm"],
        )

        peak = np.max(np.abs(signal))
        if peak > 0:
            signal = 0.891 * signal / peak

        sf.write(str(out_path), signal, SAMPLE_RATE_OUT)
        rendered += 1

        if (idx + 1) % 100 == 0 or idx == len(pred_rows) - 1:
            print(f"Processado {idx + 1}/{len(pred_rows)}")

    print(f"Concluído. Renderizados: {rendered}. Pulados: {skipped}.")


if __name__ == "__main__":
    main()
