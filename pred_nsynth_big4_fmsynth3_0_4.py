import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from keras.models import load_model

MODEL_NAME = "model_training_big4_fmsynth3_0_4"
MODEL_DIR = Path("model_training_big4_fmsynth3_0_4")
DEFAULT_NSYNTH_AUDIO_DIR = Path("nsynth-test/audio")
DEFAULT_PARAMS_CSV = Path("dataset_big4/parameters.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Roda predição de parâmetros FM no dataset NSynth usando o modelo "
            "model_training_big4_fmsynth3_0_4."
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
        help="Diretório com modelo/preprocess e destino dos arquivos de saída.",
    )
    parser.add_argument(
        "--params-csv",
        type=Path,
        default=DEFAULT_PARAMS_CSV,
        help="CSV de parâmetros do dataset de treino (usado para recuperar nomes das colunas).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size para inferência.",
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


def to_pred_map(raw_pred, output_names: list[str]) -> dict[str, np.ndarray]:
    if isinstance(raw_pred, dict):
        return {str(k): np.asarray(v) for k, v in raw_pred.items()}
    if isinstance(raw_pred, (list, tuple)):
        if len(raw_pred) != len(output_names):
            raise ValueError(
                "Quantidade de tensores de saída não confere com output_names: "
                f"{len(raw_pred)} vs {len(output_names)}"
            )
        return {
            str(name): np.asarray(pred)
            for name, pred in zip(output_names, raw_pred, strict=False)
        }
    return {str(output_names[0]): np.asarray(raw_pred)}


def build_prediction_dataframe(
    pred_map: dict[str, np.ndarray],
    preprocess_bundle: dict,
    target_columns: list[str],
) -> pd.DataFrame:
    numeric_cols = list(preprocess_bundle.get("numeric_cols", []))
    freq_col = preprocess_bundle.get("freq_col")
    categorical_cols = list(preprocess_bundle.get("categorical_cols", []))
    constant_targets = dict(preprocess_bundle.get("constant_targets", {}))
    scaler_num = preprocess_bundle.get("scaler_num")
    scaler_freq = preprocess_bundle.get("scaler_freq")

    n_samples = None
    for v in pred_map.values():
        arr = np.asarray(v)
        n_samples = arr.shape[0]
        break

    if n_samples is None:
        raise ValueError("Nenhuma saída foi produzida pelo modelo.")

    pred_df = pd.DataFrame(index=np.arange(n_samples))

    if numeric_cols:
        if "num_head" not in pred_map:
            raise KeyError("Saída 'num_head' não encontrada no modelo.")
        if scaler_num is None:
            raise ValueError("scaler_num ausente no bundle de preprocessamento.")
        num_pred_norm = np.asarray(pred_map["num_head"], dtype=np.float32)
        num_pred = scaler_num.inverse_transform(num_pred_norm)
        pred_df[numeric_cols] = num_pred

    if freq_col is not None:
        if "freq_head" not in pred_map:
            raise KeyError("Saída 'freq_head' não encontrada no modelo.")
        if scaler_freq is None:
            raise ValueError("scaler_freq ausente no bundle de preprocessamento.")

        freq_pred_norm = np.asarray(pred_map["freq_head"], dtype=np.float32).reshape(-1, 1)
        freq_pred_log2 = scaler_freq.inverse_transform(freq_pred_norm)[:, 0]
        pred_df[freq_col] = np.power(2.0, freq_pred_log2)

    for col in categorical_cols:
        head_name = f"cat__{col}"
        if head_name not in pred_map:
            raise KeyError(f"Saída categórica ausente: {head_name}")
        logits = np.asarray(pred_map[head_name], dtype=np.float32)
        pred_cls = np.argmax(logits, axis=1).astype(np.int32)
        pred_df[col] = pred_cls

    for col, value in constant_targets.items():
        pred_df[col] = value

    missing_cols = [c for c in target_columns if c not in pred_df.columns]
    if missing_cols:
        raise ValueError(f"Colunas ausentes após pós-processamento: {missing_cols}")

    return pred_df[target_columns]


def main():
    args = parse_args()

    model_dir = args.model_dir
    model_path = model_dir / f"{MODEL_NAME}.keras"
    preprocess_path = model_dir / f"target_preprocess_{MODEL_NAME}.save"
    results_path = model_dir / "results.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not preprocess_path.exists():
        raise FileNotFoundError(f"Preprocess bundle não encontrado: {preprocess_path}")

    print(f"Carregando modelo: {model_path}")
    model = load_model(model_path, compile=False)

    print(f"Carregando preprocess bundle: {preprocess_path}")
    preprocess_bundle = joblib.load(preprocess_path)

    expected_len = int(model.input_shape[1])
    if int(model.input_shape[2]) != 1:
        raise ValueError(f"Entrada inesperada do modelo: {model.input_shape}")

    print(f"Lendo áudios de: {args.audio_dir}")
    x, file_names = load_audio_batch(args.audio_dir, expected_len)
    print(f"Total de amostras carregadas: {x.shape[0]}")
    print(f"Shape de entrada para predição: {x.shape}")

    print("Rodando predição...")
    raw_pred = model.predict(x, batch_size=args.batch_size, verbose=1)
    pred_map = to_pred_map(raw_pred, list(model.output_names))

    target_columns = load_target_columns(args.params_csv)
    y_pred_df = build_prediction_dataframe(pred_map, preprocess_bundle, target_columns)
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
