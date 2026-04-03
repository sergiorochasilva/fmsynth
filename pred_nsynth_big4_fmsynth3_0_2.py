import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from keras.models import load_model

MODEL_NAME = "model_training_big4_fmsynth3_0_2"
MODEL_DIR = Path("model_training_big4_fmsynth3_0_2")
DEFAULT_NSYNTH_AUDIO_DIR = Path("nsynth-test/audio")
DEFAULT_PARAMS_CSV = Path("dataset_big4/parameters.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Roda predição de parâmetros FM no dataset NSynth usando o modelo "
            "model_training_big4_fmsynth3_0_2."
        )
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_NSYNTH_AUDIO_DIR,
        help="Diretório com os .wav do NSynth.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Diretório com modelo/scaler e destino dos arquivos de saída.",
    )
    parser.add_argument(
        "--params-csv",
        type=Path,
        default=DEFAULT_PARAMS_CSV,
        help="CSV de parâmetros do dataset de treino (usado para recuperar nomes das colunas).",
    )
    return parser.parse_args()


def load_target_columns(params_csv: Path) -> list[str]:
    if not params_csv.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {params_csv}")
    columns = pd.read_csv(params_csv, nrows=0).columns.tolist()
    return [col for col in columns if col != "id"]


def load_categorical_maps(results_json: Path) -> dict[str, list[str]]:
    if not results_json.exists():
        return {}

    with open(results_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_maps = data.get("categorical_maps", {})
    maps: dict[str, list[str]] = {}
    for key, values in raw_maps.items():
        if isinstance(values, list) and values:
            maps[str(key)] = [str(v) for v in values]
    return maps


def preprocess_audio(signal: np.ndarray, expected_len: int) -> np.ndarray:
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    signal = np.asarray(signal, dtype=np.float32)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = 0.891 * signal / peak

    if signal.shape[0] > expected_len:
        signal = signal[:expected_len]
    elif signal.shape[0] < expected_len:
        pad_width = expected_len - signal.shape[0]
        signal = np.pad(signal, (0, pad_width), mode="constant")

    return signal


def load_audio_batch(audio_dir: Path, expected_len: int) -> tuple[np.ndarray, list[str]]:
    if not audio_dir.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {audio_dir}")

    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"Nenhum arquivo .wav encontrado em: {audio_dir}")

    samples = []
    file_names = []

    for wav_path in wav_files:
        signal, _ = sf.read(str(wav_path))
        samples.append(preprocess_audio(signal, expected_len))
        file_names.append(wav_path.name)

    x = np.asarray(samples, dtype=np.float32)
    x = x.reshape((x.shape[0], x.shape[1], 1))
    return x, file_names


def decode_categorical_columns(
    y_pred_df: pd.DataFrame, categorical_maps: dict[str, list[str]]
) -> pd.DataFrame:
    decoded = {}
    for column, categories in categorical_maps.items():
        if column not in y_pred_df.columns:
            continue
        values = np.rint(y_pred_df[column].to_numpy()).astype(int)
        values = np.clip(values, 0, len(categories) - 1)
        decoded[f"{column}_decoded"] = [categories[i] for i in values]

    if not decoded:
        return pd.DataFrame(index=y_pred_df.index)

    return pd.DataFrame(decoded, index=y_pred_df.index)


def main():
    args = parse_args()

    model_dir = args.model_dir
    model_path = model_dir / f"{MODEL_NAME}.keras"
    scaler_path = model_dir / f"scaler_y_{MODEL_NAME}.save"
    results_path = model_dir / "results.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler não encontrado: {scaler_path}")

    print(f"Carregando modelo: {model_path}")
    model = load_model(model_path)

    print(f"Carregando scaler: {scaler_path}")
    scaler_y = joblib.load(scaler_path)

    expected_len = int(model.input_shape[1])
    if int(model.input_shape[2]) != 1:
        raise ValueError(f"Entrada inesperada do modelo: {model.input_shape}")

    print(f"Lendo áudios de: {args.audio_dir}")
    x, file_names = load_audio_batch(args.audio_dir, expected_len)
    print(f"Total de amostras carregadas: {x.shape[0]}")
    print(f"Shape de entrada para predição: {x.shape}")

    print("Rodando predição...")
    y_pred_norm = model.predict(x, verbose=1)
    y_pred = scaler_y.inverse_transform(y_pred_norm)

    target_columns = load_target_columns(args.params_csv)
    if y_pred.shape[1] != len(target_columns):
        raise ValueError(
            "Quantidade de saídas do modelo difere do número de colunas esperadas "
            f"({y_pred.shape[1]} vs {len(target_columns)})."
        )

    y_pred_df = pd.DataFrame(y_pred, columns=target_columns)
    y_pred_df.insert(0, "audio_file", file_names)

    categorical_maps = load_categorical_maps(results_path)
    decoded_df = decode_categorical_columns(y_pred_df, categorical_maps)
    if not decoded_df.empty:
        y_pred_df = pd.concat([y_pred_df, decoded_df], axis=1)

    model_dir.mkdir(parents=True, exist_ok=True)
    output_json = model_dir / f"params_pred_nsynth_{MODEL_NAME}.json"
    output_csv = model_dir / f"params_pred_nsynth_{MODEL_NAME}.csv"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(y_pred_df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    y_pred_df.to_csv(output_csv, index=False)

    print(f"Predições salvas em: {output_json}")
    print(f"Predições salvas em: {output_csv}")


if __name__ == "__main__":
    main()
