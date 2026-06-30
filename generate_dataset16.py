"""Generate a lower-entropy FM-synthesis benchmark corpus for hierarchical algorithm learning.

Architecture:
- Balanced sampling over algorithm/ratio combinations
- Structural FM indices remain non-zero so the algorithms stay acoustically distinct
- The nuisance controls vary over a narrower range to reduce within-class ambiguity
- Phase is fixed because it injects too much sample-level variability for stable learning
- 4-second clips and a larger corpus size to test whether scale improves generalization
- Renders audio with `fm_synth3`
- Writes `sample_*.wav`, sharded `int16` audio caches, `parameters.csv`, and `meta.json`

Data flow:
- Input: configuration constants and environment overrides
- Output: `dataset_big16/parameters.csv`, `sample_*.wav`, `audio_big16_shards/shard_*.npy`, `audio_big16_manifest.json`, and `meta.json`
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from fm_synth3 import Envelope, FMSynth3, SAMPLE_RATE_OUT, SAMPLE_RATE_RENDER

OUTPUT_DIR = Path("dataset_big16")
SHARD_DIR = OUTPUT_DIR / "audio_big16_shards"
MANIFEST_PATH = OUTPUT_DIR / "audio_big16_manifest.json"
CSV_PATH = OUTPUT_DIR / "parameters.csv"
META_PATH = OUTPUT_DIR / "meta.json"

DATASET_SIZE = int(os.getenv("DATASET_SIZE", "50000"))
AUDIO_SECONDS = float(os.getenv("AUDIO_SECONDS", "4.0"))
AUDIO_SHARD_SIZE = int(os.getenv("AUDIO_SHARD_SIZE", "256"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", str(max((os.cpu_count() or 1) - 1, 1))))
SEED = int(os.getenv("SEED", "42"))

MIN_FREQUENCY = float(os.getenv("MIN_FREQUENCY", "65.0"))
MAX_FREQUENCY = float(os.getenv("MAX_FREQUENCY", "880.0"))

ALGORITHMS = [
    "series3_parallel2",
    "series3",
    "parallel5",
    "series2x2_parallel1",
]

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
)

FIXED_CURVE = "exp"
FIXED_PHASE = 0.0
FIXED_CARRIER_LEVEL = 1.0
FIXED_RATIO_1 = 1.0
FIXED_RATIO_2 = 1.5
FIXED_RATIO_3 = 2.0
FIXED_RATIO_4 = 3.0
FIXED_RATIO_5 = 4.0
FIXED_DETUNE_1 = 0.0
FIXED_DETUNE_2 = 0.0
FIXED_DETUNE_3 = 0.0
FIXED_DETUNE_4 = 0.0
FIXED_DETUNE_5 = 0.0
FIXED_ENV_MOD = Envelope(0.01, 0.14, 0.68, 0.18, FIXED_CURVE, FIXED_CURVE, FIXED_CURVE)
FIXED_ENV_CAR = Envelope(0.01, 0.16, 0.75, 0.20, FIXED_CURVE, FIXED_CURVE, FIXED_CURVE)

CSV_HEADER = [
    "id",
    "algorithm",
    "algorithm_family",
    "frequencia_base",
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

TARGET_INFO = {
    "dataset": "dataset_big16",
    "description": "lower-entropy benchmark with balanced algorithm/ratio coverage, hierarchical algorithm labels, and 4-second clips",
    "audio_seconds": AUDIO_SECONDS,
    "dataset_size": DATASET_SIZE,
    "audio_sample_len": int(round(AUDIO_SECONDS * SAMPLE_RATE_OUT)),
    "sample_rate_out": SAMPLE_RATE_OUT,
    "sample_rate_render": SAMPLE_RATE_RENDER,
    "target_columns": CSV_HEADER[2:],
    "algorithm_classes": ALGORITHMS,
}

random.seed(SEED)
np.random.seed(SEED)


def r(value: float) -> float:
    return round(float(value), 4)


def uniform_log(min_val: float, max_val: float) -> float:
    return math.exp(random.uniform(math.log(min_val), math.log(max_val)))


def sample_algorithm_ratio_schedule(n: int) -> list[tuple[str, float]]:
    combos = [(algo, ratio) for algo in ALGORITHMS for ratio in RATIOS_DISCRETOS]
    repeats, remainder = divmod(n, len(combos))
    schedule: list[tuple[str, float]] = []
    for _ in range(repeats):
        shuffled = combos[:]
        random.shuffle(shuffled)
        schedule.extend(shuffled)
    if remainder:
        shuffled = combos[:]
        random.shuffle(shuffled)
        schedule.extend(shuffled[:remainder])
    random.shuffle(schedule)
    return schedule


def sample_index_active() -> float:
    x = random.random()
    if x < 0.45:
        return random.uniform(0.15, 1.8)
    if x < 0.88:
        return random.uniform(1.8, 4.2)
    return random.uniform(4.2, 6.0)


def algorithm_family(algorithm: str) -> str:
    if algorithm in {"series3", "series3_parallel2"}:
        return "series"
    if algorithm in {"parallel5", "series2x2_parallel1"}:
        return "parallel"
    raise ValueError(f"Unknown algorithm family for {algorithm}")


def sample_env_mod() -> Envelope:
    return Envelope(
        r(uniform_log(0.003, 0.03)),
        r(uniform_log(0.04, 0.20)),
        r(random.uniform(0.45, 0.85)),
        r(uniform_log(0.03, 0.22)),
        FIXED_CURVE,
        FIXED_CURVE,
        FIXED_CURVE,
    )


def sample_env_car() -> Envelope:
    return Envelope(
        r(uniform_log(0.003, 0.04)),
        r(uniform_log(0.04, 0.28)),
        r(random.uniform(0.55, 0.90)),
        r(uniform_log(0.05, 0.30)),
        FIXED_CURVE,
        FIXED_CURVE,
        FIXED_CURVE,
    )


def valid_configuration(fc: float, ratio_carrier: float) -> bool:
    safe_out = 0.45 * (SAMPLE_RATE_OUT / 2.0)
    safe_render = 0.45 * (SAMPLE_RATE_RENDER / 2.0)
    return fc * ratio_carrier < safe_out and fc * FIXED_RATIO_5 < safe_render


def build_row(sample_id: int, algorithm: str, ratio_carrier: float) -> dict:
    attempts = 0
    while True:
        attempts += 1
        if attempts > 200:
            raise RuntimeError("Could not sample a valid FM configuration.")

        fc = uniform_log(MIN_FREQUENCY, MAX_FREQUENCY)
        if not valid_configuration(fc, ratio_carrier):
            continue

        env_mod = sample_env_mod()
        env_car = sample_env_car()
        return {
            "id": sample_id,
            "algorithm": algorithm,
            "algorithm_family": algorithm_family(algorithm),
            "frequencia_base": r(fc),
            "ratio_carrier": r(ratio_carrier),
            "index_12": r(sample_index_active()),
            "index_23": r(sample_index_active()),
            "index_3c": r(sample_index_active()),
            "index_4c": r(sample_index_active()),
            "index_5c": r(sample_index_active()),
            "detune_carrier": r(random.uniform(-3.0, 3.0)),
            "feedback": r(random.uniform(0.0, 0.18)),
            "lfo_rate": r(random.uniform(0.0, 4.0)),
            "lfo_depth_cents": r(random.uniform(0.0, 8.0)),
            "key_scaling": r(random.uniform(0.0, 0.25)),
            "env_mod_attack": env_mod.attack,
            "env_mod_decay": env_mod.decay,
            "env_mod_sustain": env_mod.sustain,
            "env_mod_release": env_mod.release,
            "env_car_attack": env_car.attack,
            "env_car_decay": env_car.decay,
            "env_car_sustain": env_car.sustain,
            "env_car_release": env_car.release,
        }


def build_envelope_from_fixed() -> tuple[Envelope, Envelope]:
    return FIXED_ENV_MOD, FIXED_ENV_CAR


def build_synth_from_row(row: dict) -> tuple[FMSynth3, float, str]:
    env_mod = Envelope(
        float(row["env_mod_attack"]),
        float(row["env_mod_decay"]),
        float(row["env_mod_sustain"]),
        float(row["env_mod_release"]),
        FIXED_CURVE,
        FIXED_CURVE,
        FIXED_CURVE,
    )
    env_car = Envelope(
        float(row["env_car_attack"]),
        float(row["env_car_decay"]),
        float(row["env_car_sustain"]),
        float(row["env_car_release"]),
        FIXED_CURVE,
        FIXED_CURVE,
        FIXED_CURVE,
    )
    synth = FMSynth3(
        ratio1=FIXED_RATIO_1,
        ratio2=FIXED_RATIO_2,
        ratio3=FIXED_RATIO_3,
        ratio4=FIXED_RATIO_4,
        ratio5=FIXED_RATIO_5,
        ratio_carrier=float(row["ratio_carrier"]),
        detune1=FIXED_DETUNE_1,
        detune2=FIXED_DETUNE_2,
        detune3=FIXED_DETUNE_3,
        detune4=FIXED_DETUNE_4,
        detune5=FIXED_DETUNE_5,
        detune_carrier=float(row["detune_carrier"]),
        index_12=float(row["index_12"]),
        index_23=float(row["index_23"]),
        index_3c=float(row["index_3c"]),
        index_4c=float(row["index_4c"]),
        index_5c=float(row["index_5c"]),
        env_mod=env_mod,
        env_car=env_car,
        env_scale1=1.0,
        env_scale2=1.0,
        env_scale3=1.0,
        env_scale4=1.0,
        env_scale5=1.0,
        env_scale_carrier=1.0,
        carrier_level=FIXED_CARRIER_LEVEL,
        feedback=float(row["feedback"]),
        lfo_rate=float(row["lfo_rate"]),
        lfo_depth_cents=float(row["lfo_depth_cents"]),
        key_scaling=float(row["key_scaling"]),
        key_scaling_ref_hz=440.0,
        random_phase=False,
        phase1=FIXED_PHASE,
        phase2=FIXED_PHASE,
        phase3=FIXED_PHASE,
        phase4=FIXED_PHASE,
        phase5=FIXED_PHASE,
        phase_carrier=FIXED_PHASE,
        downsample_16k=True,
    )
    return synth, float(row["frequencia_base"]), row["algorithm"]


def ensure_audio_length(signal: np.ndarray, audio_sample_len: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.shape[0] == audio_sample_len:
        return signal
    if signal.shape[0] > audio_sample_len:
        return signal[:audio_sample_len]
    padded = np.zeros(audio_sample_len, dtype=np.float32)
    padded[: signal.shape[0]] = signal
    return padded


def to_pcm16(signal: np.ndarray) -> np.ndarray:
    return np.round(np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)


def shard_filename(shard_idx: int) -> str:
    return f"shard_{shard_idx:05d}.npy"


def render_shard(task: tuple[int, list[dict], int]) -> dict:
    shard_idx, shard_rows, audio_sample_len = task
    shard_start = int(shard_rows[0]["id"])
    shard_end = int(shard_rows[-1]["id"]) + 1
    shard_audio = np.empty((len(shard_rows), audio_sample_len), dtype=np.int16)
    shard_file = SHARD_DIR / shard_filename(shard_idx)
    tmp_file = shard_file.with_suffix(".npy.tmp")

    print(
        f"Rendering shard {shard_idx + 1} with {len(shard_rows)} samples "
        f"({shard_start + 1}..{shard_end})",
        flush=True,
    )

    for local_idx, row in enumerate(shard_rows):
        sample_id = int(row["id"])
        wav_path = OUTPUT_DIR / f"sample_{sample_id}.wav"
        if wav_path.exists():
            pcm16, sample_rate = sf.read(str(wav_path), dtype="int16", always_2d=False)
            if sample_rate != SAMPLE_RATE_OUT:
                raise ValueError(f"Unexpected sample rate in cached wav {wav_path}: {sample_rate}")
            pcm16 = np.asarray(pcm16, dtype=np.int16).reshape(-1)
        else:
            synth, fc, algorithm = build_synth_from_row(row)
            signal = synth.synth(AUDIO_SECONDS, fc, algorithm=algorithm)
            signal = ensure_audio_length(signal, audio_sample_len)
            pcm16 = to_pcm16(signal)
            sf.write(str(wav_path), pcm16, SAMPLE_RATE_OUT, subtype="PCM_16")
        shard_audio[local_idx] = pcm16

        if local_idx % 64 == 0 or local_idx == len(shard_rows) - 1:
            print(
                f"  shard {shard_idx + 1}: {local_idx + 1}/{len(shard_rows)}",
                flush=True,
            )

    with open(tmp_file, "wb") as f:
        np.save(f, shard_audio)
    os.replace(tmp_file, shard_file)

    return {
        "index": shard_idx,
        "file": str(Path("audio_big16_shards") / shard_filename(shard_idx)),
        "start_row": shard_start,
        "end_row": shard_end,
        "shape": [len(shard_rows), audio_sample_len],
        "dtype": "int16",
    }


def write_csv(csv_path: Path) -> list[dict]:
    schedule = sample_algorithm_ratio_schedule(DATASET_SIZE)
    rows: list[dict] = []
    tmp_path = csv_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for sample_id in range(DATASET_SIZE):
            print(f"Generating parameters {sample_id + 1}/{DATASET_SIZE}...")
            algorithm, ratio_carrier = schedule[sample_id]
            row = build_row(sample_id, algorithm, ratio_carrier)
            env_mod = sample_env_mod()
            env_car = sample_env_car()
            row["env_mod_attack"] = env_mod.attack
            row["env_mod_decay"] = env_mod.decay
            row["env_mod_sustain"] = env_mod.sustain
            row["env_mod_release"] = env_mod.release
            row["env_car_attack"] = env_car.attack
            row["env_car_decay"] = env_car.decay
            row["env_car_sustain"] = env_car.sustain
            row["env_car_release"] = env_car.release
            writer.writerow(row)
            rows.append(row)
    os.replace(tmp_path, csv_path)
    return rows


def write_audio(rows: list[dict], audio_sample_len: int) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    n_shards = int(math.ceil(DATASET_SIZE / AUDIO_SHARD_SIZE))
    shards_meta = []

    shard_tasks: list[tuple[int, list[dict], int]] = []
    for shard_idx in range(n_shards):
        shard_start = shard_idx * AUDIO_SHARD_SIZE
        shard_end = min(shard_start + AUDIO_SHARD_SIZE, DATASET_SIZE)
        shard_tasks.append((shard_idx, rows[shard_start:shard_end], audio_sample_len))

    if NUM_WORKERS > 1 and len(shard_tasks) > 1:
        print(f"Rendering {len(shard_tasks)} shards with {NUM_WORKERS} workers...")
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for meta in executor.map(render_shard, shard_tasks):
                shards_meta.append(meta)
    else:
        for task in shard_tasks:
            shards_meta.append(render_shard(task))

    shards_meta.sort(key=lambda item: int(item["index"]))

    manifest = {
        "dataset": "dataset_big16",
        "audio_seconds": AUDIO_SECONDS,
        "audio_sample_len": audio_sample_len,
        "audio_shard_size": AUDIO_SHARD_SIZE,
        "num_workers": NUM_WORKERS,
        "dtype": "int16",
        "total_rows": DATASET_SIZE,
        "total_shards": n_shards,
        "shard_dir": SHARD_DIR.name,
        "shards": shards_meta,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def write_meta(manifest: dict, rows: list[dict]) -> None:
    counts = {algo: 0 for algo in ALGORITHMS}
    for row in rows:
        counts[row["algorithm"]] += 1

    ratio_counts = {}
    for ratio in RATIOS_DISCRETOS:
        ratio_counts[str(round(float(ratio), 4))] = 0
    for row in rows:
        ratio_counts[str(float(row["ratio_carrier"]))] = ratio_counts.get(str(float(row["ratio_carrier"])), 0) + 1

    meta = {
        **TARGET_INFO,
        "seed": SEED,
        "audio_shard_size": AUDIO_SHARD_SIZE,
        "num_workers": NUM_WORKERS,
        "audio_manifest": MANIFEST_PATH.name,
        "audio_shard_dir": SHARD_DIR.name,
        "parameters_csv": CSV_PATH.name,
        "algorithm_counts": counts,
        "ratio_counts": ratio_counts,
        "manifest_total_rows": manifest["total_rows"],
        "manifest_total_shards": manifest["total_shards"],
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = write_csv(CSV_PATH)
    manifest = write_audio(rows, TARGET_INFO["audio_sample_len"])
    write_meta(manifest, rows)
    print(f"Dataset written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
