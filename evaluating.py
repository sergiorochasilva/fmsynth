import os
import numpy as np
import soundfile as sf
import librosa
import json

SR_REF = 16000


# ---------- utilidades ----------
def load_audio_mono(file_path, target_sr=None):
    y, sr = sf.read(file_path, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    assert sr == SR_REF, f"SR inesperado: {sr} (esperado {SR_REF})"
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y.astype(np.float32), sr


def match_lengths(a, b):
    """Corta ambos para o menor tamanho para permitir comparação."""
    m = min(len(a), len(b))
    return a[:m], b[:m]


# ---------- métricas ----------
def fft_distance(file1, file2, sr_ref=SR_REF):
    y1, sr1 = load_audio_mono(file1, target_sr=sr_ref)
    y2, sr2 = load_audio_mono(file2, target_sr=sr_ref)
    y1, y2 = match_lengths(y1, y2)
    fft1 = np.fft.rfft(y1)
    fft2 = np.fft.rfft(y2)
    return float(np.linalg.norm(fft1 - fft2))


def stft_distance(file1, file2, sr_ref=SR_REF, n_fft=2048, hop_length=512):
    y1, sr1 = load_audio_mono(file1, target_sr=sr_ref)
    y2, sr2 = load_audio_mono(file2, target_sr=sr_ref)
    y1, y2 = match_lengths(y1, y2)
    stft1 = np.abs(librosa.stft(y=y1, n_fft=n_fft, hop_length=hop_length))
    stft2 = np.abs(librosa.stft(y=y2, n_fft=n_fft, hop_length=hop_length))
    m = min(stft1.shape[1], stft2.shape[1])
    stft1, stft2 = stft1[:, :m], stft2[:, :m]
    return float(np.linalg.norm(stft1 - stft2))


def log_mel_spectrogram_distance(
    file1,
    file2,
    sr_ref=SR_REF,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    fmin=0.0,
    fmax=None,
):
    y1, sr1 = load_audio_mono(file1, target_sr=sr_ref)
    y2, sr2 = load_audio_mono(file2, target_sr=sr_ref)
    y1, y2 = match_lengths(y1, y2)

    S1 = librosa.feature.melspectrogram(
        y=y1,
        sr=sr_ref,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    S2 = librosa.feature.melspectrogram(
        y=y2,
        sr=sr_ref,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )

    logS1 = np.log1p(S1)
    logS2 = np.log1p(S2)

    m = min(logS1.shape[1], logS2.shape[1])
    logS1, logS2 = logS1[:, :m], logS2[:, :m]

    raw_dist = np.linalg.norm(logS1 - logS2)
    num_elements = logS1.size  # n_mels * n_frames
    norm_dist = raw_dist / num_elements

    return float(raw_dist), float(norm_dist), int(num_elements)


# ---------- varredura de arquivos ----------
base_path_pred = "nsynth-pred"
base_path_test = "nsynth-test/audio"

file_list = [f for f in os.listdir(base_path_pred) if f.endswith(".wav")]

# Comparação do original do NSynth versus o sintetizado comp apoio do modelo
# results = []

# for file_name in file_list:
#     print(file_name)

#     f_pred = os.path.join(base_path_pred, file_name)
#     f_test = os.path.join(base_path_test, file_name)

#     fft_dist = fft_distance(f_pred, f_test)
#     stft_dist = stft_distance(f_pred, f_test)
#     logmel_raw, logmel_norm, num_elements = log_mel_spectrogram_distance(f_pred, f_test)

#     results.append(
#         {
#             "file_name": file_name,
#             "fft_distance": fft_dist,
#             "stft_distance": stft_dist,
#             "log_mel_spectrogram_distance_raw": logmel_raw,
#             "log_mel_spectrogram_distance_normalized": logmel_norm,
#             "log_mel_num_elements": num_elements,
#         }
#     )

# Baseline de comparação entre duas amostras quaisquer
# with open("distances.json", "w") as f:
#     json.dump(results, f, indent=4)

# f_pred = os.path.join(base_path_test, "keyboard_electronic_098-090-100.wav")
# f_test = os.path.join(base_path_test, "brass_acoustic_046-092-075.wav")

# fft_dist = fft_distance(f_pred, f_test)
# stft_dist = stft_distance(f_pred, f_test)
# logmel_raw, logmel_norm, num_elements = log_mel_spectrogram_distance(f_pred, f_test)

# print("FFT Distance:", fft_dist)
# print("STFT Distance:", stft_dist)
# print("Log Mel Distance (raw):", logmel_raw)
# print("Log Mel Distance (normalized):", logmel_norm)
# print("Log Mel Number of Elements:", num_elements)

# Comparação da Nsynth com um baseline senoide 440 Hz
results = []

for file_name in file_list:
    print(file_name)

    f_pred = os.path.join(base_path_test, file_name)
    f_test = "output2.wav"

    fft_dist = fft_distance(f_pred, f_test)
    stft_dist = stft_distance(f_pred, f_test)
    logmel_raw, logmel_norm, num_elements = log_mel_spectrogram_distance(f_pred, f_test)

    results.append(
        {
            "file_name": file_name,
            "fft_distance": fft_dist,
            "stft_distance": stft_dist,
            "log_mel_spectrogram_distance_raw": logmel_raw,
            "log_mel_spectrogram_distance_normalized": logmel_norm,
            "log_mel_num_elements": num_elements,
        }
    )

with open("distances_baseline.json", "w") as f:
    json.dump(results, f, indent=4)
