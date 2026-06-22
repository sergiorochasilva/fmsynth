"""Generate the `dataset_big7` FM-synthesis corpus.

Architecture:
- Balanced sampling of FM parameters and synthesis settings, with fixed phase and envelope curves
- Renders audio with `fm_synth3`
- Writes incremental WAV files plus sharded memmap-friendly `int16` audio caches

Data flow:
- Input: configuration constants in the script
- Output: `dataset_big7/parameters.csv`, `sample_*.wav`, sharded `int16` audio caches, and `meta.json`
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
tamanho_dataset = 50000
precisao_decimal = 4
output_dir = "dataset_big7"
audio_shard_dir = f"{output_dir}/audio_big7_shards"
audio_manifest_path = f"{output_dir}/audio_big7_manifest.json"
audio_shard_size = int(os.getenv("AUDIO_SHARD_SIZE", "256"))
audio_sample_len = int(round(duracao_amostras * SAMPLE_RATE_OUT))
seed = 42

# Proporção entre estilos internos de amostragem.
MIXED_SAMPLING_PROB = 0.9

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

fixed_carrier_level = 1.0

# Key scaling
key_scaling_range = (-0.8, 0.4)
key_scaling_ref_hz = 440.0
FIXED_CURVE = "exp"
FIXED_PHASE = 0.0

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

# Cabeçalho CSV
CSV_HEADER = [
    "id",
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
    "env2_attack",
    "env2_decay",
    "env2_sustain",
    "env2_release",
    "env3_attack",
    "env3_decay",
    "env3_sustain",
    "env3_release",
    "env4_attack",
    "env4_decay",
    "env4_sustain",
    "env4_release",
    "env5_attack",
    "env5_decay",
    "env5_sustain",
    "env5_release",
    "env_carrier_attack",
    "env_carrier_decay",
    "env_carrier_sustain",
    "env_carrier_release",
    "env_scale1",
    "env_scale2",
    "env_scale3",
    "env_scale4",
    "env_scale5",
    "env_scale_carrier",
    "feedback",
    "lfo_rate",
    "lfo_depth_cents",
    "key_scaling",
]


# -------------------------
# Utilitários
# -------------------------
def r(x: float) -> float:
    return round(float(x), precisao_decimal)


def uniform_log(min_val: float, max_val: float) -> float:
    return math.exp(random.uniform(math.log(min_val), math.log(max_val)))


def sample_beta(is_musical: bool) -> float:
    # Distribuição semelhante ao generate_dataset2
    x = random.random()
    if x < 0.2:
        v = random.uniform(0.0, 0.5)
    elif x < 0.8:
        v = random.uniform(0.5, 3.0)
    else:
        v = random.uniform(3.0, max_index)
    if not is_musical:
        v = random.uniform(min_index, max_index)
    return v


def sample_ratio(is_musical: bool) -> float:
    if is_musical:
        if random.random() < 0.9:
            return random.choice(RATIOS_DISCRETOS)
        return uniform_log(min_ratio, max_ratio)
    return uniform_log(min_ratio, max_ratio)


def sample_ratio_carrier(is_musical: bool) -> float:
    if is_musical:
        if random.random() < 0.6:
            return 1.0
        return sample_ratio(is_musical)
    return sample_ratio(is_musical)


def sample_detune(is_musical: bool) -> float:
    if is_musical:
        return random.uniform(*detune_musical)
    return random.uniform(*detune_random)


def sample_env() -> Envelope:
    a = random.uniform(*attack_range)
    d = random.uniform(*decay_range)
    s = random.uniform(*sustain_range)
    rls = random.uniform(*release_range)
    return Envelope(
        a,
        d,
        s,
        rls,
        curve_attack=FIXED_CURVE,
        curve_decay=FIXED_CURVE,
        curve_release=FIXED_CURVE,
    )


def sample_env_scale(is_musical: bool) -> float:
    if is_musical:
        return random.uniform(0.4, 1.1)
    return random.uniform(*env_scale_range)


def sample_feedback(is_musical: bool) -> float:
    if is_musical:
        if random.random() < 0.7:
            return 0.0
    return random.uniform(*feedback_range)


def sample_lfo(is_musical: bool) -> tuple[float, float]:
    if is_musical and random.random() < 0.7:
        return 0.0, 0.0
    return random.uniform(*lfo_rate_range), random.uniform(*lfo_depth_cents_range)


def sample_key_scaling(is_musical: bool) -> float:
    if is_musical:
        return random.uniform(-0.6, 0.2)
    return random.uniform(*key_scaling_range)


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


# -------------------------
# Geração
# -------------------------


def generate_parameters_csv(csv_path: str) -> None:
    random.seed(seed)
    np.random.seed(seed)

    algorithm_schedule, algo_counts = build_algorithm_schedule(
        tamanho_dataset, ALGORITHMS
    )

    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()

        for i in range(tamanho_dataset):
            print(f"Gerando parâmetros {i + 1} de {tamanho_dataset}...")
            algorithm = algorithm_schedule[i]
            is_musical = random.random() < MIXED_SAMPLING_PROB

            # Tenta sortear parâmetros válidos
            attempts = 0
            while True:
                attempts += 1
                if attempts > 200:
                    raise RuntimeError("Não foi possível sortear parâmetros válidos")

                # Frequência base
                fc = uniform_log(min_frequency, max_frequency)

                ratio1 = sample_ratio(is_musical)
                ratio2 = sample_ratio(is_musical)
                ratio3 = sample_ratio(is_musical)
                ratio4 = sample_ratio(is_musical)
                ratio5 = sample_ratio(is_musical)
                ratio_carrier = sample_ratio_carrier(is_musical)

                if not valid_frequencies(
                    fc, [ratio1, ratio2, ratio3, ratio4, ratio5], ratio_carrier
                ):
                    continue

                detune1 = sample_detune(is_musical)
                detune2 = sample_detune(is_musical)
                detune3 = sample_detune(is_musical)
                detune4 = sample_detune(is_musical)
                detune5 = sample_detune(is_musical)
                detune_carrier = sample_detune(is_musical)

                index_12 = sample_beta(is_musical)
                index_23 = sample_beta(is_musical)
                index_3c = sample_beta(is_musical)
                index_4c = sample_beta(is_musical)
                index_5c = sample_beta(is_musical)

                env1 = sample_env()
                env2 = sample_env()
                env3 = sample_env()
                env4 = sample_env()
                env5 = sample_env()
                env_carrier = sample_env()

                env_scale1 = sample_env_scale(is_musical)
                env_scale2 = sample_env_scale(is_musical)
                env_scale3 = sample_env_scale(is_musical)
                env_scale4 = sample_env_scale(is_musical)
                env_scale5 = sample_env_scale(is_musical)
                env_scale_carrier = sample_env_scale(is_musical)

                feedback = sample_feedback(is_musical)
                lfo_rate, lfo_depth_cents = sample_lfo(is_musical)
                key_scaling = sample_key_scaling(is_musical)
                break

            data = {
                "id": i,
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
                "env2_attack": r(env2.attack),
                "env2_decay": r(env2.decay),
                "env2_sustain": r(env2.sustain),
                "env2_release": r(env2.release),
                "env3_attack": r(env3.attack),
                "env3_decay": r(env3.decay),
                "env3_sustain": r(env3.sustain),
                "env3_release": r(env3.release),
                "env4_attack": r(env4.attack),
                "env4_decay": r(env4.decay),
                "env4_sustain": r(env4.sustain),
                "env4_release": r(env4.release),
                "env5_attack": r(env5.attack),
                "env5_decay": r(env5.decay),
                "env5_sustain": r(env5.sustain),
                "env5_release": r(env5.release),
                "env_carrier_attack": r(env_carrier.attack),
                "env_carrier_decay": r(env_carrier.decay),
                "env_carrier_sustain": r(env_carrier.sustain),
                "env_carrier_release": r(env_carrier.release),
                "env_scale1": r(env_scale1),
                "env_scale2": r(env_scale2),
                "env_scale3": r(env_scale3),
                "env_scale4": r(env_scale4),
                "env_scale5": r(env_scale5),
                "env_scale_carrier": r(env_scale_carrier),
                "feedback": r(feedback),
                "lfo_rate": r(lfo_rate),
                "lfo_depth_cents": r(lfo_depth_cents),
                "key_scaling": r(key_scaling),
            }

            writer.writerow(data)

    os.replace(tmp_path, csv_path)


def parse_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def build_envelope(row: dict, prefix: str) -> Envelope:
    return Envelope(
        float(row[f"{prefix}_attack"]),
        float(row[f"{prefix}_decay"]),
        float(row[f"{prefix}_sustain"]),
        float(row[f"{prefix}_release"]),
        curve_attack=FIXED_CURVE,
        curve_decay=FIXED_CURVE,
        curve_release=FIXED_CURVE,
    )


def build_synth_from_row(row: dict) -> tuple[FMSynth3, float, str]:
    fm_synth = FMSynth3(
        ratio1=float(row["ratio1"]),
        ratio2=float(row["ratio2"]),
        ratio3=float(row["ratio3"]),
        ratio4=float(row["ratio4"]),
        ratio5=float(row["ratio5"]),
        ratio_carrier=float(row["ratio_carrier"]),
        detune1=float(row["detune1"]),
        detune2=float(row["detune2"]),
        detune3=float(row["detune3"]),
        detune4=float(row["detune4"]),
        detune5=float(row["detune5"]),
        detune_carrier=float(row["detune_carrier"]),
        index_12=float(row["index_12"]),
        index_23=float(row["index_23"]),
        index_3c=float(row["index_3c"]),
        index_4c=float(row["index_4c"]),
        index_5c=float(row["index_5c"]),
        env1=build_envelope(row, "env1"),
        env2=build_envelope(row, "env2"),
        env3=build_envelope(row, "env3"),
        env4=build_envelope(row, "env4"),
        env5=build_envelope(row, "env5"),
        env_carrier=build_envelope(row, "env_carrier"),
        env_scale1=float(row["env_scale1"]),
        env_scale2=float(row["env_scale2"]),
        env_scale3=float(row["env_scale3"]),
        env_scale4=float(row["env_scale4"]),
        env_scale5=float(row["env_scale5"]),
        env_scale_carrier=float(row["env_scale_carrier"]),
        carrier_level=fixed_carrier_level,
        feedback=float(row["feedback"]),
        lfo_rate=float(row["lfo_rate"]),
        lfo_depth_cents=float(row["lfo_depth_cents"]),
        key_scaling=float(row["key_scaling"]),
        key_scaling_ref_hz=key_scaling_ref_hz,
        random_phase=False,
        phase1=FIXED_PHASE,
        phase2=FIXED_PHASE,
        phase3=FIXED_PHASE,
        phase4=FIXED_PHASE,
        phase5=FIXED_PHASE,
        phase_carrier=FIXED_PHASE,
        downsample_16k=True,
    )
    return fm_synth, float(row["frequencia_base"]), row["algorithm"]


def ensure_audio_length(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.shape[0] == audio_sample_len:
        return signal
    if signal.shape[0] > audio_sample_len:
        return signal[:audio_sample_len]

    padded = np.zeros(audio_sample_len, dtype=np.float32)
    padded[: signal.shape[0]] = signal
    return padded


def to_pcm16(signal: np.ndarray) -> np.ndarray:
    signal = np.clip(signal, -1.0, 1.0)
    return np.round(signal * 32767.0).astype(np.int16)


def synthesize_row(row: dict) -> tuple[np.ndarray, np.ndarray]:
    fm_synth, fc, algorithm = build_synth_from_row(row)
    signal = fm_synth.synth(duracao_amostras, fc, algorithm=algorithm)
    signal = ensure_audio_length(signal)
    return signal, to_pcm16(signal)


def count_rows(csv_path: str) -> int:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def summarize_csv(csv_path: str) -> dict:
    counts_algo = {a: 0 for a in ALGORITHMS}
    total = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            algo = row.get("algorithm", "")
            if algo in counts_algo:
                counts_algo[algo] += 1

    return {
        "total_rows": total,
        "counts_algorithm": counts_algo,
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


def shard_filename(shard_idx: int) -> str:
    return f"shard_{shard_idx:05d}.npy"


def shard_path(shard_idx: int) -> str:
    return os.path.join(audio_shard_dir, shard_filename(shard_idx))


def load_manifest() -> dict | None:
    if not os.path.exists(audio_manifest_path):
        return None
    try:
        with open(audio_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return None
    return manifest


def validate_shard_file(path: str, expected_shape: tuple[int, int]) -> bool:
    if not os.path.exists(path):
        return False
    try:
        shard = np.load(path, mmap_mode="r")
    except Exception:
        return False
    return shard.ndim == 2 and shard.shape == expected_shape and shard.dtype == np.int16


def cache_is_valid(expected_rows: int) -> bool:
    manifest = load_manifest()
    if not manifest:
        return False

    if int(manifest.get("total_rows", -1)) != int(expected_rows):
        return False
    if int(manifest.get("audio_sample_len", -1)) != int(audio_sample_len):
        return False
    if int(manifest.get("audio_shard_size", -1)) != int(audio_shard_size):
        return False
    if manifest.get("dtype") != "int16":
        return False

    shards = manifest.get("shards", [])
    if not shards:
        return False

    for shard_meta in shards:
        if shard_meta.get("dtype") != "int16":
            return False
        file_name = shard_meta.get("file")
        if not file_name:
            return False
        path = os.path.join(output_dir, file_name)
        shape = shard_meta.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            return False
        if not validate_shard_file(path, (int(shape[0]), int(shape[1]))):
            return False

    return True


def generate_missing_wavs_from_csv(csv_path: str) -> None:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = int(row["id"])
            out_path = f"{output_dir}/sample_{sample_id}.wav"
            if os.path.exists(out_path):
                continue

            print(f"Gerando áudio {sample_id + 1} de {tamanho_dataset}...")
            _signal, pcm16 = synthesize_row(row)
            sf.write(out_path, pcm16, SAMPLE_RATE_OUT, subtype="PCM_16")


def generate_audio_shards(csv_path: str) -> None:
    os.makedirs(audio_shard_dir, exist_ok=True)
    n_shards = int(math.ceil(tamanho_dataset / audio_shard_size))

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for shard_idx in range(n_shards):
            shard_start = shard_idx * audio_shard_size
            shard_end = min(shard_start + audio_shard_size, tamanho_dataset)
            shard_rows = shard_end - shard_start
            shard_file = shard_path(shard_idx)
            tmp_file = shard_file + ".tmp"

            if validate_shard_file(shard_file, (shard_rows, audio_sample_len)):
                for _ in range(shard_rows):
                    next(reader, None)
                continue

            shard_audio = np.empty((shard_rows, audio_sample_len), dtype=np.int16)

            for local_idx in range(shard_rows):
                row = next(reader)
                sample_id = int(row["id"])
                out_path = f"{output_dir}/sample_{sample_id}.wav"

                print(f"Gerando áudio {sample_id + 1} de {tamanho_dataset}...")
                _signal, pcm16 = synthesize_row(row)

                if not os.path.exists(out_path):
                    sf.write(out_path, pcm16, SAMPLE_RATE_OUT, subtype="PCM_16")

                shard_audio[local_idx] = pcm16

            with open(tmp_file, "wb") as shard_f:
                np.save(shard_f, shard_audio)
            os.replace(tmp_file, shard_file)


def write_manifest(csv_path: str) -> None:
    shards = []
    n_shards = int(math.ceil(tamanho_dataset / audio_shard_size))
    for shard_idx in range(n_shards):
        shard_start = shard_idx * audio_shard_size
        shard_end = min(shard_start + audio_shard_size, tamanho_dataset)
        shard_rows = shard_end - shard_start
        file_name = os.path.join("audio_big7_shards", shard_filename(shard_idx))
        shards.append(
            {
                "index": shard_idx,
                "file": file_name,
                "start_row": shard_start,
                "end_row": shard_end,
                "shape": [shard_rows, audio_sample_len],
                "dtype": "int16",
            }
        )

    manifest = {
        "output_dir": output_dir,
        "dtype": "int16",
        "audio_sample_len": audio_sample_len,
        "audio_shard_size": audio_shard_size,
        "total_rows": tamanho_dataset,
        "total_shards": n_shards,
        "shard_dir": os.path.basename(audio_shard_dir),
        "csv": os.path.basename(csv_path),
        "shards": shards,
    }

    with open(audio_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def write_meta(csv_path: str) -> None:
    summary = summarize_csv(csv_path)
    meta = {
        "output_dir": output_dir,
        "parameters_csv": os.path.basename(csv_path),
        "seed": seed,
        "tamanho_dataset": tamanho_dataset,
        "duracao_amostras": duracao_amostras,
        "sample_rate_out": SAMPLE_RATE_OUT,
        "audio_sample_len": audio_sample_len,
        "audio_dtype": "int16",
        "audio_manifest": os.path.basename(audio_manifest_path),
        "audio_shard_dir": os.path.basename(audio_shard_dir),
        "audio_shard_size": audio_shard_size,
        "sampling_mix_prob": MIXED_SAMPLING_PROB,
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
            "fixed_carrier_level": fixed_carrier_level,
            "key_scaling_range": key_scaling_range,
            "key_scaling_ref_hz": key_scaling_ref_hz,
            "fixed_curve": FIXED_CURVE,
            "fixed_phase": FIXED_PHASE,
        },
        "stats": summary,
        "audio_files_present": count_audio_files(),
        "audio_shards_present": cache_is_valid(tamanho_dataset),
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

    if cache_is_valid(tamanho_dataset):
        pass
    else:
        generate_audio_shards(csv_path)

    write_manifest(csv_path)
    generate_missing_wavs_from_csv(csv_path)
    write_meta(csv_path)


if __name__ == "__main__":
    main()
