"""Generate the `dataset_big5` FM-synthesis corpus.

Architecture:
- Balanced sampling of FM parameters, curves, and synthesis settings
- Renders audio with `fm_synth3`

Data flow:
- Input: configuration constants in the script
- Output: `dataset_big5/parameters.csv`, `sample_*.wav`, and `meta.json`
"""

import csv
import json
import math
import os
import random

import numpy as np
import soundfile as sf

from fm_synth3 import Envelope, FMSynth3, SAMPLE_RATE_OUT, SAMPLE_RATE_RENDER

# -------------------------
# Configuração geral
# -------------------------
duracao_amostras = 4.0
tamanho_dataset = 10000
precisao_decimal = 4
output_dir = "dataset_big5"
seed = 42

# Proporção entre estilos (igual ao espírito do generate_dataset2.py)
MUSICAL_PROB = 0.9

# Frequência base (Hz)
min_frequency = 20.0
max_frequency = 6000.0

# Ratios (musicais)
RATIOS_DISCRETOS = [
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
]

min_ratio = 1 / 8
max_ratio = 8.0

# Índices (I)
min_index = 0.0
max_index = 8.0

# Detune (cents)
detune_musical = (-15.0, 15.0)
detune_random = (-30.0, 30.0)

# Envelope ranges
attack_range = (0.001, 0.5)
decay_range = (0.01, 0.8)
sustain_range = (0.05, 0.95)
release_range = (0.02, 1.0)

# Escalas de envelope
env_scale_range = (0.2, 1.2)

# Feedback e LFO
feedback_range = (0.0, 0.8)
lfo_rate_range = (0.5, 8.0)
lfo_depth_cents_range = (0.0, 20.0)

# Nível da portadora
carrier_level_range = (0.4, 1.0)

# Key scaling
key_scaling_range = (-0.8, 0.4)
key_scaling_ref_hz = 440.0

# Curvas de envelope disponíveis
CURVES = ["linear", "exp", "log", "s_curve"]

# Algoritmos disponíveis.
# Nota: "dual_chain" foi removido daqui porque é alias de "series2x2_parallel1"
# em fm_synth3. Isso evita desbalanceamento quando o treino mergeia rótulos.
ALGORITHMS = [
    "series3_parallel2",
    "series3",
    "parallel5",
    "series2x2_parallel1",
    "series5",
    "series4_parallel1",
    "series2_parallel3",
    "series3_parallel1_plus1",
]

CURVE_COLUMNS = [
    "env1_curve_attack",
    "env1_curve_decay",
    "env1_curve_release",
    "env2_curve_attack",
    "env2_curve_decay",
    "env2_curve_release",
    "env3_curve_attack",
    "env3_curve_decay",
    "env3_curve_release",
    "env4_curve_attack",
    "env4_curve_decay",
    "env4_curve_release",
    "env5_curve_attack",
    "env5_curve_decay",
    "env5_curve_release",
    "env_carrier_curve_attack",
    "env_carrier_curve_decay",
    "env_carrier_curve_release",
]

# Cabeçalho CSV
CSV_HEADER = [
    "id",
    "style",
    "algorithm",
    "frequencia_base",
    "ratio1",
    "ratio2",
    "ratio3",
    "ratio4",
    "ratio5",
    "ratio_carrier",
    "detune1",
    "detune2",
    "detune3",
    "detune4",
    "detune5",
    "detune_carrier",
    "index_12",
    "index_23",
    "index_3c",
    "index_4c",
    "index_5c",
    "env1_attack",
    "env1_decay",
    "env1_sustain",
    "env1_release",
    "env1_curve_attack",
    "env1_curve_decay",
    "env1_curve_release",
    "env2_attack",
    "env2_decay",
    "env2_sustain",
    "env2_release",
    "env2_curve_attack",
    "env2_curve_decay",
    "env2_curve_release",
    "env3_attack",
    "env3_decay",
    "env3_sustain",
    "env3_release",
    "env3_curve_attack",
    "env3_curve_decay",
    "env3_curve_release",
    "env4_attack",
    "env4_decay",
    "env4_sustain",
    "env4_release",
    "env4_curve_attack",
    "env4_curve_decay",
    "env4_curve_release",
    "env5_attack",
    "env5_decay",
    "env5_sustain",
    "env5_release",
    "env5_curve_attack",
    "env5_curve_decay",
    "env5_curve_release",
    "env_carrier_attack",
    "env_carrier_decay",
    "env_carrier_sustain",
    "env_carrier_release",
    "env_carrier_curve_attack",
    "env_carrier_curve_decay",
    "env_carrier_curve_release",
    "env_scale1",
    "env_scale2",
    "env_scale3",
    "env_scale4",
    "env_scale5",
    "env_scale_carrier",
    "carrier_level",
    "feedback",
    "lfo_rate",
    "lfo_depth_cents",
    "key_scaling",
    "key_scaling_ref_hz",
    "random_phase",
    "phase1",
    "phase2",
    "phase3",
    "phase4",
    "phase5",
    "phase_carrier",
    "downsample_16k",
]


# -------------------------
# Utilitários
# -------------------------
def r(x: float) -> float:
    return round(float(x), precisao_decimal)


def uniform_log(min_val: float, max_val: float) -> float:
    return math.exp(random.uniform(math.log(min_val), math.log(max_val)))


def sample_beta(style: str) -> float:
    # Distribuição semelhante ao generate_dataset2
    x = random.random()
    if x < 0.2:
        v = random.uniform(0.0, 0.5)
    elif x < 0.8:
        v = random.uniform(0.5, 3.0)
    else:
        v = random.uniform(3.0, max_index)
    if style == "random":
        v = random.uniform(min_index, max_index)
    return v


def sample_ratio(style: str) -> float:
    if style == "musical":
        if random.random() < 0.9:
            return random.choice(RATIOS_DISCRETOS)
        return uniform_log(min_ratio, max_ratio)
    return uniform_log(min_ratio, max_ratio)


def sample_ratio_carrier(style: str) -> float:
    if style == "musical":
        if random.random() < 0.6:
            return 1.0
        return sample_ratio(style)
    return sample_ratio(style)


def sample_detune(style: str) -> float:
    if style == "musical":
        return random.uniform(*detune_musical)
    return random.uniform(*detune_random)


def sample_env(
    curve_attack: str,
    curve_decay: str,
    curve_release: str,
) -> Envelope:
    a = random.uniform(*attack_range)
    d = random.uniform(*decay_range)
    s = random.uniform(*sustain_range)
    rls = random.uniform(*release_range)
    return Envelope(
        a,
        d,
        s,
        rls,
        curve_attack=curve_attack,
        curve_decay=curve_decay,
        curve_release=curve_release,
    )


def sample_env_scale(style: str) -> float:
    if style == "musical":
        return random.uniform(0.4, 1.1)
    return random.uniform(*env_scale_range)


def sample_feedback(style: str) -> float:
    if style == "musical":
        if random.random() < 0.7:
            return 0.0
    return random.uniform(*feedback_range)


def sample_lfo(style: str) -> tuple[float, float]:
    if style == "musical" and random.random() < 0.7:
        return 0.0, 0.0
    return random.uniform(*lfo_rate_range), random.uniform(*lfo_depth_cents_range)


def sample_key_scaling(style: str) -> float:
    if style == "musical":
        return random.uniform(-0.6, 0.2)
    return random.uniform(*key_scaling_range)


def sample_carrier_level(style: str) -> float:
    if style == "musical":
        return random.uniform(0.6, 1.0)
    return random.uniform(*carrier_level_range)


def sample_phases() -> tuple[float, float, float, float, float, float]:
    return (
        random.uniform(0.0, 2.0 * math.pi),
        random.uniform(0.0, 2.0 * math.pi),
        random.uniform(0.0, 2.0 * math.pi),
        random.uniform(0.0, 2.0 * math.pi),
        random.uniform(0.0, 2.0 * math.pi),
        random.uniform(0.0, 2.0 * math.pi),
    )


def valid_frequencies(fc: float, ratios: list[float], ratio_carrier: float) -> bool:
    nyq_out = SAMPLE_RATE_OUT / 2.0
    safe_out = 0.9 * nyq_out
    nyq_render = SAMPLE_RATE_RENDER / 2.0
    safe_render = 0.45 * nyq_render

    if fc * ratio_carrier >= safe_out:
        return False
    for r in ratios:
        if fc * r >= safe_render:
            return False
    return True


def build_algorithm_schedule(
    n: int, algos: list[str]
) -> tuple[list[str], dict[str, int]]:
    base = n // len(algos)
    rem = n % len(algos)
    schedule: list[str] = []
    counts: dict[str, int] = {}
    for idx, algo in enumerate(algos):
        count = base + (1 if idx < rem else 0)
        counts[algo] = count
        schedule.extend([algo] * count)
    random.shuffle(schedule)
    return schedule, counts


def build_style_queues(counts: dict[str, int]) -> dict[str, list[str]]:
    queues: dict[str, list[str]] = {}
    for algo, n in counts.items():
        n_mus = int(round(MUSICAL_PROB * n))
        n_mus = max(0, min(n, n_mus))
        styles = ["musical"] * n_mus + ["random"] * (n - n_mus)
        random.shuffle(styles)
        queues[algo] = styles
    return queues


def build_balanced_curve_schedule(n: int, curves: list[str]) -> list[str]:
    base = n // len(curves)
    rem = n % len(curves)
    schedule: list[str] = []
    for idx, curve in enumerate(curves):
        count = base + (1 if idx < rem else 0)
        schedule.extend([curve] * count)
    random.shuffle(schedule)
    return schedule


def build_curve_queues(n: int) -> dict[str, list[str]]:
    queues: dict[str, list[str]] = {}
    for col in CURVE_COLUMNS:
        queues[col] = build_balanced_curve_schedule(n, CURVES)
    return queues


# -------------------------
# Geração
# -------------------------


def generate_parameters_csv(csv_path: str) -> None:
    random.seed(seed)
    np.random.seed(seed)

    algorithm_schedule, algo_counts = build_algorithm_schedule(
        tamanho_dataset, ALGORITHMS
    )
    style_queues = build_style_queues(algo_counts)
    curve_queues = build_curve_queues(tamanho_dataset)

    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()

        for i in range(tamanho_dataset):
            print(f"Gerando parâmetros {i + 1} de {tamanho_dataset}...")
            algorithm = algorithm_schedule[i]
            style = style_queues[algorithm].pop()

            # Tenta sortear parâmetros válidos
            attempts = 0
            while True:
                attempts += 1
                if attempts > 200:
                    raise RuntimeError("Não foi possível sortear parâmetros válidos")

                # Frequência base
                fc = uniform_log(min_frequency, max_frequency)

                ratio1 = sample_ratio(style)
                ratio2 = sample_ratio(style)
                ratio3 = sample_ratio(style)
                ratio4 = sample_ratio(style)
                ratio5 = sample_ratio(style)
                ratio_carrier = sample_ratio_carrier(style)

                if not valid_frequencies(
                    fc, [ratio1, ratio2, ratio3, ratio4, ratio5], ratio_carrier
                ):
                    continue

                detune1 = sample_detune(style)
                detune2 = sample_detune(style)
                detune3 = sample_detune(style)
                detune4 = sample_detune(style)
                detune5 = sample_detune(style)
                detune_carrier = sample_detune(style)

                index_12 = sample_beta(style)
                index_23 = sample_beta(style)
                index_3c = sample_beta(style)
                index_4c = sample_beta(style)
                index_5c = sample_beta(style)

                env1_curve_attack = curve_queues["env1_curve_attack"].pop()
                env1_curve_decay = curve_queues["env1_curve_decay"].pop()
                env1_curve_release = curve_queues["env1_curve_release"].pop()
                env2_curve_attack = curve_queues["env2_curve_attack"].pop()
                env2_curve_decay = curve_queues["env2_curve_decay"].pop()
                env2_curve_release = curve_queues["env2_curve_release"].pop()
                env3_curve_attack = curve_queues["env3_curve_attack"].pop()
                env3_curve_decay = curve_queues["env3_curve_decay"].pop()
                env3_curve_release = curve_queues["env3_curve_release"].pop()
                env4_curve_attack = curve_queues["env4_curve_attack"].pop()
                env4_curve_decay = curve_queues["env4_curve_decay"].pop()
                env4_curve_release = curve_queues["env4_curve_release"].pop()
                env5_curve_attack = curve_queues["env5_curve_attack"].pop()
                env5_curve_decay = curve_queues["env5_curve_decay"].pop()
                env5_curve_release = curve_queues["env5_curve_release"].pop()
                env_carrier_curve_attack = curve_queues["env_carrier_curve_attack"].pop()
                env_carrier_curve_decay = curve_queues["env_carrier_curve_decay"].pop()
                env_carrier_curve_release = curve_queues["env_carrier_curve_release"].pop()

                env1 = sample_env(
                    curve_attack=env1_curve_attack,
                    curve_decay=env1_curve_decay,
                    curve_release=env1_curve_release,
                )
                env2 = sample_env(
                    curve_attack=env2_curve_attack,
                    curve_decay=env2_curve_decay,
                    curve_release=env2_curve_release,
                )
                env3 = sample_env(
                    curve_attack=env3_curve_attack,
                    curve_decay=env3_curve_decay,
                    curve_release=env3_curve_release,
                )
                env4 = sample_env(
                    curve_attack=env4_curve_attack,
                    curve_decay=env4_curve_decay,
                    curve_release=env4_curve_release,
                )
                env5 = sample_env(
                    curve_attack=env5_curve_attack,
                    curve_decay=env5_curve_decay,
                    curve_release=env5_curve_release,
                )
                env_carrier = sample_env(
                    curve_attack=env_carrier_curve_attack,
                    curve_decay=env_carrier_curve_decay,
                    curve_release=env_carrier_curve_release,
                )

                env_scale1 = sample_env_scale(style)
                env_scale2 = sample_env_scale(style)
                env_scale3 = sample_env_scale(style)
                env_scale4 = sample_env_scale(style)
                env_scale5 = sample_env_scale(style)
                env_scale_carrier = sample_env_scale(style)

                feedback = sample_feedback(style)
                lfo_rate, lfo_depth_cents = sample_lfo(style)
                key_scaling = sample_key_scaling(style)
                carrier_level = sample_carrier_level(style)

                phase1, phase2, phase3, phase4, phase5, phase_carrier = sample_phases()
                break

            data = {
                "id": i,
                "style": style,
                "algorithm": algorithm,
                "frequencia_base": r(fc),
                "ratio1": r(ratio1),
                "ratio2": r(ratio2),
                "ratio3": r(ratio3),
                "ratio4": r(ratio4),
                "ratio5": r(ratio5),
                "ratio_carrier": r(ratio_carrier),
                "detune1": r(detune1),
                "detune2": r(detune2),
                "detune3": r(detune3),
                "detune4": r(detune4),
                "detune5": r(detune5),
                "detune_carrier": r(detune_carrier),
                "index_12": r(index_12),
                "index_23": r(index_23),
                "index_3c": r(index_3c),
                "index_4c": r(index_4c),
                "index_5c": r(index_5c),
                "env1_attack": r(env1.attack),
                "env1_decay": r(env1.decay),
                "env1_sustain": r(env1.sustain),
                "env1_release": r(env1.release),
                "env1_curve_attack": env1_curve_attack,
                "env1_curve_decay": env1_curve_decay,
                "env1_curve_release": env1_curve_release,
                "env2_attack": r(env2.attack),
                "env2_decay": r(env2.decay),
                "env2_sustain": r(env2.sustain),
                "env2_release": r(env2.release),
                "env2_curve_attack": env2_curve_attack,
                "env2_curve_decay": env2_curve_decay,
                "env2_curve_release": env2_curve_release,
                "env3_attack": r(env3.attack),
                "env3_decay": r(env3.decay),
                "env3_sustain": r(env3.sustain),
                "env3_release": r(env3.release),
                "env3_curve_attack": env3_curve_attack,
                "env3_curve_decay": env3_curve_decay,
                "env3_curve_release": env3_curve_release,
                "env4_attack": r(env4.attack),
                "env4_decay": r(env4.decay),
                "env4_sustain": r(env4.sustain),
                "env4_release": r(env4.release),
                "env4_curve_attack": env4_curve_attack,
                "env4_curve_decay": env4_curve_decay,
                "env4_curve_release": env4_curve_release,
                "env5_attack": r(env5.attack),
                "env5_decay": r(env5.decay),
                "env5_sustain": r(env5.sustain),
                "env5_release": r(env5.release),
                "env5_curve_attack": env5_curve_attack,
                "env5_curve_decay": env5_curve_decay,
                "env5_curve_release": env5_curve_release,
                "env_carrier_attack": r(env_carrier.attack),
                "env_carrier_decay": r(env_carrier.decay),
                "env_carrier_sustain": r(env_carrier.sustain),
                "env_carrier_release": r(env_carrier.release),
                "env_carrier_curve_attack": env_carrier_curve_attack,
                "env_carrier_curve_decay": env_carrier_curve_decay,
                "env_carrier_curve_release": env_carrier_curve_release,
                "env_scale1": r(env_scale1),
                "env_scale2": r(env_scale2),
                "env_scale3": r(env_scale3),
                "env_scale4": r(env_scale4),
                "env_scale5": r(env_scale5),
                "env_scale_carrier": r(env_scale_carrier),
                "carrier_level": r(carrier_level),
                "feedback": r(feedback),
                "lfo_rate": r(lfo_rate),
                "lfo_depth_cents": r(lfo_depth_cents),
                "key_scaling": r(key_scaling),
                "key_scaling_ref_hz": r(key_scaling_ref_hz),
                "random_phase": False,
                "phase1": r(phase1),
                "phase2": r(phase2),
                "phase3": r(phase3),
                "phase4": r(phase4),
                "phase5": r(phase5),
                "phase_carrier": r(phase_carrier),
                "downsample_16k": True,
            }

            writer.writerow(data)

    os.replace(tmp_path, csv_path)


def parse_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def generate_audio_from_csv(csv_path: str) -> None:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = int(row["id"])
            out_path = f"{output_dir}/sample_{sample_id}.wav"
            if os.path.exists(out_path):
                continue

            print(f"Gerando áudio {sample_id + 1} de {tamanho_dataset}...")

            fc = float(row["frequencia_base"])
            ratio1 = float(row["ratio1"])
            ratio2 = float(row["ratio2"])
            ratio3 = float(row["ratio3"])
            ratio4 = float(row["ratio4"])
            ratio5 = float(row["ratio5"])
            ratio_carrier = float(row["ratio_carrier"])
            detune1 = float(row["detune1"])
            detune2 = float(row["detune2"])
            detune3 = float(row["detune3"])
            detune4 = float(row["detune4"])
            detune5 = float(row["detune5"])
            detune_carrier = float(row["detune_carrier"])
            index_12 = float(row["index_12"])
            index_23 = float(row["index_23"])
            index_3c = float(row["index_3c"])
            index_4c = float(row["index_4c"])
            index_5c = float(row["index_5c"])

            env1 = Envelope(
                float(row["env1_attack"]),
                float(row["env1_decay"]),
                float(row["env1_sustain"]),
                float(row["env1_release"]),
                curve_attack=row["env1_curve_attack"],
                curve_decay=row["env1_curve_decay"],
                curve_release=row["env1_curve_release"],
            )
            env2 = Envelope(
                float(row["env2_attack"]),
                float(row["env2_decay"]),
                float(row["env2_sustain"]),
                float(row["env2_release"]),
                curve_attack=row["env2_curve_attack"],
                curve_decay=row["env2_curve_decay"],
                curve_release=row["env2_curve_release"],
            )
            env3 = Envelope(
                float(row["env3_attack"]),
                float(row["env3_decay"]),
                float(row["env3_sustain"]),
                float(row["env3_release"]),
                curve_attack=row["env3_curve_attack"],
                curve_decay=row["env3_curve_decay"],
                curve_release=row["env3_curve_release"],
            )
            env4 = Envelope(
                float(row["env4_attack"]),
                float(row["env4_decay"]),
                float(row["env4_sustain"]),
                float(row["env4_release"]),
                curve_attack=row["env4_curve_attack"],
                curve_decay=row["env4_curve_decay"],
                curve_release=row["env4_curve_release"],
            )
            env5 = Envelope(
                float(row["env5_attack"]),
                float(row["env5_decay"]),
                float(row["env5_sustain"]),
                float(row["env5_release"]),
                curve_attack=row["env5_curve_attack"],
                curve_decay=row["env5_curve_decay"],
                curve_release=row["env5_curve_release"],
            )
            env_carrier = Envelope(
                float(row["env_carrier_attack"]),
                float(row["env_carrier_decay"]),
                float(row["env_carrier_sustain"]),
                float(row["env_carrier_release"]),
                curve_attack=row["env_carrier_curve_attack"],
                curve_decay=row["env_carrier_curve_decay"],
                curve_release=row["env_carrier_curve_release"],
            )

            env_scale1 = float(row["env_scale1"])
            env_scale2 = float(row["env_scale2"])
            env_scale3 = float(row["env_scale3"])
            env_scale4 = float(row["env_scale4"])
            env_scale5 = float(row["env_scale5"])
            env_scale_carrier = float(row["env_scale_carrier"])

            carrier_level = float(row["carrier_level"])
            feedback = float(row["feedback"])
            lfo_rate = float(row["lfo_rate"])
            lfo_depth_cents = float(row["lfo_depth_cents"])
            key_scaling = float(row["key_scaling"])
            key_scaling_ref_hz = float(row["key_scaling_ref_hz"])

            random_phase = parse_bool(row["random_phase"])
            phase1 = float(row["phase1"])
            phase2 = float(row["phase2"])
            phase3 = float(row["phase3"])
            phase4 = float(row["phase4"])
            phase5 = float(row["phase5"])
            phase_carrier = float(row["phase_carrier"])
            downsample_16k = parse_bool(row["downsample_16k"])

            fm_synth = FMSynth3(
                ratio1=ratio1,
                ratio2=ratio2,
                ratio3=ratio3,
                ratio4=ratio4,
                ratio5=ratio5,
                ratio_carrier=ratio_carrier,
                detune1=detune1,
                detune2=detune2,
                detune3=detune3,
                detune4=detune4,
                detune5=detune5,
                detune_carrier=detune_carrier,
                index_12=index_12,
                index_23=index_23,
                index_3c=index_3c,
                index_4c=index_4c,
                index_5c=index_5c,
                env1=env1,
                env2=env2,
                env3=env3,
                env4=env4,
                env5=env5,
                env_carrier=env_carrier,
                env_scale1=env_scale1,
                env_scale2=env_scale2,
                env_scale3=env_scale3,
                env_scale4=env_scale4,
                env_scale5=env_scale5,
                env_scale_carrier=env_scale_carrier,
                carrier_level=carrier_level,
                feedback=feedback,
                lfo_rate=lfo_rate,
                lfo_depth_cents=lfo_depth_cents,
                key_scaling=key_scaling,
                key_scaling_ref_hz=key_scaling_ref_hz,
                random_phase=random_phase,
                phase1=phase1,
                phase2=phase2,
                phase3=phase3,
                phase4=phase4,
                phase5=phase5,
                phase_carrier=phase_carrier,
                downsample_16k=downsample_16k,
            )

            algorithm = row["algorithm"]
            signal = fm_synth.synth(duracao_amostras, fc, algorithm=algorithm)
            sf.write(out_path, signal, SAMPLE_RATE_OUT)


def count_rows(csv_path: str) -> int:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def summarize_csv(csv_path: str) -> dict:
    counts_algo = {a: 0 for a in ALGORITHMS}
    counts_style = {"musical": 0, "random": 0}
    counts_algo_style = {a: {"musical": 0, "random": 0} for a in ALGORITHMS}
    counts_curve_by_col = {col: {curve: 0 for curve in CURVES} for col in CURVE_COLUMNS}
    total = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            algo = row.get("algorithm", "")
            style = row.get("style", "")
            if algo in counts_algo:
                counts_algo[algo] += 1
                if style in counts_algo_style[algo]:
                    counts_algo_style[algo][style] += 1
            if style in counts_style:
                counts_style[style] += 1
            for col in CURVE_COLUMNS:
                curve = row.get(col, "")
                if curve in counts_curve_by_col[col]:
                    counts_curve_by_col[col][curve] += 1

    return {
        "total_rows": total,
        "counts_algorithm": counts_algo,
        "counts_style": counts_style,
        "counts_algorithm_style": counts_algo_style,
        "counts_curve_by_col": counts_curve_by_col,
    }


def count_audio_files() -> int:
    try:
        return sum(
            1
            for name in os.listdir(output_dir)
            if name.startswith("sample_") and name.endswith(".wav")
        )
    except FileNotFoundError:
        return 0


def write_meta(csv_path: str) -> None:
    summary = summarize_csv(csv_path)
    meta = {
        "output_dir": output_dir,
        "parameters_csv": os.path.basename(csv_path),
        "seed": seed,
        "tamanho_dataset": tamanho_dataset,
        "duracao_amostras": duracao_amostras,
        "sample_rate_out": SAMPLE_RATE_OUT,
        "musical_prob": MUSICAL_PROB,
        "algorithms": ALGORITHMS,
        "config": {
            "min_frequency": min_frequency,
            "max_frequency": max_frequency,
            "min_ratio": min_ratio,
            "max_ratio": max_ratio,
            "min_index": min_index,
            "max_index": max_index,
            "detune_musical": detune_musical,
            "detune_random": detune_random,
            "attack_range": attack_range,
            "decay_range": decay_range,
            "sustain_range": sustain_range,
            "release_range": release_range,
            "env_scale_range": env_scale_range,
            "feedback_range": feedback_range,
            "lfo_rate_range": lfo_rate_range,
            "lfo_depth_cents_range": lfo_depth_cents_range,
            "carrier_level_range": carrier_level_range,
            "key_scaling_range": key_scaling_range,
            "key_scaling_ref_hz": key_scaling_ref_hz,
            "curves": CURVES,
        },
        "stats": summary,
        "audio_files_present": count_audio_files(),
    }

    with open(f"{output_dir}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = f"{output_dir}/parameters.csv"

    if not os.path.exists(csv_path):
        generate_parameters_csv(csv_path)
    else:
        n_rows = count_rows(csv_path)
        if n_rows != tamanho_dataset:
            raise RuntimeError(
                f"parameters.csv tem {n_rows} linhas, mas tamanho_dataset={tamanho_dataset}. "
                "Ajuste tamanho_dataset ou recrie o arquivo."
            )

    generate_audio_from_csv(csv_path)
    write_meta(csv_path)


if __name__ == "__main__":
    main()
