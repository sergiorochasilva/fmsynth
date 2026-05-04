"""Encode `dataset_big4` audio into latent vectors with a trained encoder.

Architecture:
- Loads the saved encoder from the matching autoencoder training run
- Applies the same waveform preprocessing used at training time

Data flow:
- Input: `dataset_big4/sample_*.wav` plus encoder/preprocess artifacts
- Output: `latent_big4_fmsynth3_0_1.npy` and `encoding_metadata_big4_fmsynth3_0_1.json`
"""

import json
import os
import re
from typing import List, Tuple

import joblib
import numpy as np
import soundfile as sf
import tensorflow as tf

BASE_PATH = os.getenv("ENC_BASE_PATH", "dataset_big4")
MODEL_DIR = os.getenv("ENC_MODEL_DIR", "autoencoder_training_big4_fmsynth3_0_1")
OUTPUT_DIR = os.getenv("ENC_OUTPUT_DIR", "dataset_big4_encoded")

ENCODER_FILENAME = os.getenv(
    "ENC_ENCODER_FILENAME",
    "encoder_autoencoder_training_big4_fmsynth3_0_1.keras",
)
PREPROCESS_FILENAME = os.getenv(
    "ENC_PREPROCESS_FILENAME",
    "preprocess_autoencoder_training_big4_fmsynth3_0_1.save",
)

BATCH_SIZE = int(os.getenv("ENC_BATCH_SIZE", "8"))
MAX_SAMPLES = int(os.getenv("ENC_MAX_SAMPLES", "0"))
AUDIO_DTYPE = os.getenv("ENC_AUDIO_DTYPE", "float32").strip().lower()
SAVE_NPY = os.getenv("ENC_SAVE_NPY", "1") == "1"

if BATCH_SIZE < 1:
    raise ValueError("ENC_BATCH_SIZE deve ser >= 1")
if MAX_SAMPLES < 0:
    raise ValueError("ENC_MAX_SAMPLES deve ser >= 0")
if AUDIO_DTYPE not in {"float16", "float32"}:
    raise ValueError("ENC_AUDIO_DTYPE deve ser 'float16' ou 'float32'.")


def load_sample_entries(base_path: str) -> List[Tuple[int, str]]:
    pattern = re.compile(r"^sample_(\d+)\.wav$")
    entries: List[Tuple[int, str]] = []

    for filename in os.listdir(base_path):
        match = pattern.match(filename)
        if not match:
            continue
        sample_id = int(match.group(1))
        wav_path = os.path.join(base_path, filename)
        entries.append((sample_id, wav_path))

    if not entries:
        raise FileNotFoundError(f"Nenhum arquivo sample_*.wav encontrado em: {base_path}")

    entries.sort(key=lambda x: x[0])
    return entries


def preprocess_audio(signal: np.ndarray, expected_len: int, peak_norm: float) -> np.ndarray:
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    signal = np.asarray(signal, dtype=np.float32)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = (peak_norm * signal) / peak

    if signal.shape[0] > expected_len:
        signal = signal[:expected_len]
    elif signal.shape[0] < expected_len:
        pad_width = expected_len - signal.shape[0]
        signal = np.pad(signal, (0, pad_width), mode="constant")

    if AUDIO_DTYPE == "float16":
        return signal.astype(np.float16)
    return signal.astype(np.float32)


def load_audio_batch(
    batch_paths: List[str],
    expected_len: int,
    peak_norm: float,
) -> np.ndarray:
    batch = []
    for wav_path in batch_paths:
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {wav_path}")

        signal, _sr = sf.read(wav_path)
        signal = preprocess_audio(signal, expected_len=expected_len, peak_norm=peak_norm)
        batch.append(signal)

    x_batch = np.asarray(batch, dtype=np.float16 if AUDIO_DTYPE == "float16" else np.float32)
    return x_batch.reshape((x_batch.shape[0], x_batch.shape[1], 1))


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"Aviso ao configurar memory growth: {exc}")

    encoder_path = os.path.join(MODEL_DIR, ENCODER_FILENAME)
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder não encontrado: {encoder_path}")

    preprocess_path = os.path.join(MODEL_DIR, PREPROCESS_FILENAME)
    preprocess_bundle = {}
    if os.path.exists(preprocess_path):
        preprocess_bundle = joblib.load(preprocess_path)

    sample_entries = load_sample_entries(BASE_PATH)
    if MAX_SAMPLES > 0:
        sample_entries = sample_entries[:MAX_SAMPLES]
    sample_paths = [path for _, path in sample_entries]
    n_samples = len(sample_entries)

    print(f"Total de amostras para codificar: {n_samples}")
    if MAX_SAMPLES > 0:
        print(f"Limite aplicado por ENC_MAX_SAMPLES={MAX_SAMPLES}")
    print(f"Carregando encoder: {encoder_path}")

    encoder = tf.keras.models.load_model(encoder_path, compile=False)

    if len(encoder.input_shape) != 3:
        raise ValueError(f"Input shape inesperado no encoder: {encoder.input_shape}")

    model_audio_len = int(encoder.input_shape[1])
    bundle_audio_len = int(preprocess_bundle.get("audio_len", model_audio_len))
    audio_len = model_audio_len
    if bundle_audio_len != model_audio_len:
        print(
            "Aviso: audio_len do preprocess difere do modelo. "
            f"preprocess={bundle_audio_len}, modelo={model_audio_len}. Usando modelo."
        )

    peak_norm = float(preprocess_bundle.get("peak_norm", 0.891))

    latent_dim = int(encoder.output_shape[-1])
    latents = np.zeros((n_samples, latent_dim), dtype=np.float32)

    for start in range(0, n_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_samples)
        batch_paths = sample_paths[start:end]

        x_batch = load_audio_batch(
            batch_paths=batch_paths,
            expected_len=audio_len,
            peak_norm=peak_norm,
        )
        z_batch = encoder.predict(x_batch, batch_size=len(batch_paths), verbose=0)
        latents[start:end] = z_batch.astype(np.float32)

        if end % 500 == 0 or end == n_samples:
            print(f"Codificados {end}/{n_samples}")

    npy_path = os.path.join(OUTPUT_DIR, "latent_big4_fmsynth3_0_1.npy")
    meta_out = os.path.join(OUTPUT_DIR, "encoding_metadata_big4_fmsynth3_0_1.json")

    if SAVE_NPY:
        np.save(npy_path, latents)
        print(f"Arquivo salvo: {npy_path}")

    metadata = {
        "base_path": BASE_PATH,
        "model_dir": MODEL_DIR,
        "encoder_path": encoder_path,
        "preprocess_path": preprocess_path if os.path.exists(preprocess_path) else None,
        "output_dir": OUTPUT_DIR,
        "n_samples": int(n_samples),
        "audio_len": int(audio_len),
        "latent_dim": int(latent_dim),
        "batch_size": int(BATCH_SIZE),
        "max_samples": int(MAX_SAMPLES),
        "audio_dtype": AUDIO_DTYPE,
        "peak_norm": peak_norm,
        "files": {
            "latent_npy": npy_path if SAVE_NPY else None,
        },
    }

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Metadata salva: {meta_out}")
    print("Concluído.")


if __name__ == "__main__":
    main()
