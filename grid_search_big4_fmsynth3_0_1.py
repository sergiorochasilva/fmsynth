"""Grid-search runner for `dataset_big4` regression architectures.

Architecture:
- Builds candidate dense/conv model families and evaluates them under different hyperparameters

Data flow:
- Input: `dataset_big4`-derived training arrays and experiment configuration
- Output: combination list, progress file, search logs, and model selection artifacts
"""

import json
import math
import os
import random
import time
import traceback
from datetime import datetime
from itertools import product
from multiprocessing import Process

import joblib
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
)
from keras.models import Model

SCRIPT_NAME = "grid_search_big4_fmsynth3_0_1"
OUTPUT_DIR = SCRIPT_NAME
COMBINATIONS_FILE = os.path.join(OUTPUT_DIR, "combinacoes.json")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")
SEED = 42


def config_gpu():
    # Configurando para nao alocar diretamente toda a memoria da GPU (alocar conforme necessario)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            exit(1)


# Funcoes para construcao do modelo

def regressor(
    input_dims,
    output_dims,
    activation,
    bias_regressor,
    dropout_regressor: float,
    kernel_regularizer_regressor: str,
):
    input_layer = Input(shape=[input_dims])

    x_0 = Dense(
        int(input_dims / 2),
        activation=activation,
        use_bias=bias_regressor,
        kernel_regularizer=kernel_regularizer_regressor,
    )(input_layer)
    x_0 = Dropout(dropout_regressor)(x_0)
    x_2 = Dense(int(input_dims / 4), activation=activation, use_bias=bias_regressor)(
        x_0
    )
    saidas = Dense(
        output_dims, activation=None, name="regressor_saidas", use_bias=bias_regressor
    )(x_2)

    return Model(input_layer, saidas, name="regressor")


def build_model(
    input_len,
    input_dims,
    output_dims,
    activation,
    bias_cnn,
    kernel_regularizer_cnn,
    bias_regressor,
    dropout_regressor,
    kernel_regularizer_regressor,
    optimizer,
):

    # Camadas de entrada
    input_layer = Input(shape=(input_len, input_dims))

    x_n = BatchNormalization()(input_layer)

    # Features 1
    extrator1 = Conv1D(
        filters=8,
        kernel_size=4,
        strides=1,
        activation=activation,
        input_shape=(input_len, 1),
        kernel_regularizer=kernel_regularizer_cnn,
        use_bias=bias_cnn,
    )
    pooling1 = MaxPooling1D(pool_size=16)

    features1 = extrator1(x_n)
    features1 = pooling1(features1)

    extrator1_2 = Conv1D(
        filters=16,
        kernel_size=8,
        strides=1,
        activation=activation,
        input_shape=(input_len, 1),
        kernel_regularizer=kernel_regularizer_cnn,
        use_bias=bias_cnn,
    )
    pooling1_2 = MaxPooling1D(pool_size=16)

    features1_2 = extrator1_2(features1)
    features1_2 = pooling1_2(features1_2)

    extrator1_3 = Conv1D(
        filters=32,
        kernel_size=16,
        strides=1,
        activation=activation,
        input_shape=(input_len, 1),
        kernel_regularizer=kernel_regularizer_cnn,
        use_bias=bias_cnn,
    )
    pooling1_3 = MaxPooling1D(pool_size=16)

    features1_3 = extrator1_3(features1_2)
    features1_3 = pooling1_3(features1_3)

    features1_flatten1 = Flatten()(features1_3)

    features1_flatten1_normalized = BatchNormalization()(features1_flatten1)

    # Regressao
    regressao = regressor(
        features1_flatten1_normalized.shape[1],
        output_dims,
        activation,
        bias_regressor,
        dropout_regressor,
        kernel_regularizer_regressor,
    )

    saida = regressao(features1_flatten1)

    model = Model(input_layer, saida, name="complete")
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mse"])

    return model


def train_model(
    x,
    y,
    input_len: int,
    input_dims: int,  # x.shape[2]
    output_dims: int,  # y.shape[1]
    activation,
    bias_cnn,
    kernel_regularizer_cnn,
    bias_regressor,
    dropout_regressor,
    kernel_regularizer_regressor,
    epochs: int,
    patience: int,
    validation_split: float,
    optimizer: str,
):
    # Construindo o modelo
    model = build_model(
        input_len,
        input_dims,
        output_dims,
        activation,
        bias_cnn,
        kernel_regularizer_cnn,
        bias_regressor,
        dropout_regressor,
        kernel_regularizer_regressor,
        optimizer,
    )

    # Callback para recuperar o melhor peso, e parar quando ficar tres epocas sem melhora
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )

    # Treinando o modelo
    return model, model.fit(
        x, y, epochs=epochs, validation_split=validation_split, callbacks=[callback]
    )


def run_experiment(name: str, run_name: str, x, y, x_t, y_t, scaler_y, params: dict):
    config_gpu()

    # Iniciando o MLFlow
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment(name)

    try:
        mlflow.start_run(run_name=run_name)

        # Gravando os parametros
        mlflow.log_params(params)

        # Treinando a rede
        inicio = time.time()
        model, history = train_model(
            x=x,
            y=y,
            **params,
        )
        tempo_decorrido = time.time() - inicio

        # Gravando as metricas do treino
        mlflow.log_metric("training_time", tempo_decorrido)

        mse = min(history.history["mse"])
        val_mse = min(history.history["val_mse"])
        val_rmse = math.sqrt(val_mse)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("val_mse", val_mse)
        mlflow.log_metric("val_rmse", val_rmse)

        mae = min(history.history["mae"])
        val_mae = min(history.history["val_mae"])

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("val_mae", val_mae)

        # Inferindo o teste
        y_pred_norm = model.predict(x_t)
        y_pred = scaler_y.inverse_transform(y_pred_norm)

        test_mae = tf.keras.losses.MAE(y_t, y_pred).numpy().mean()
        test_mse = tf.keras.losses.MSE(y_t, y_pred).numpy().mean()
        test_rmse = math.sqrt(test_mse)

        # Gravando as metricas de teste
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)

        mlflow.set_tag(
            "Training Info",
            "CNN FMSynth3 (big4).",
        )
    except Exception as e:
        print(traceback.format_exc())
        print(f"Erro no treino: {e}")
    finally:
        mlflow.end_run()


def build_param_grid():
    # Mantido amplo, mas com opcoes razoaveis
    return {
        "activation": [
            "relu",
            "gelu",
            "swish",
        ],
        "bias_cnn": [True, False],
        "kernel_regularizer_cnn": [None, "l2"],
        "bias_regressor": [True, False],
        "dropout_regressor": [0.0, 0.2, 0.4],
        "kernel_regularizer_regressor": [None, "l2"],
        "optimizer": [
            "Adam",
            "Nadam",
            "AdamW",
        ],
    }


def build_combinations(grid: dict, seed: int):
    keys = list(grid.keys())
    combos = [dict(zip(keys, values)) for values in product(*[grid[k] for k in keys])]
    rng = random.Random(seed)
    rng.shuffle(combos)
    for idx, combo in enumerate(combos):
        combo["id"] = idx
    return combos


def save_json(path: str, payload: dict):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_or_create_combinations():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(COMBINATIONS_FILE):
        with open(COMBINATIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    grid = build_param_grid()
    combos = build_combinations(grid, SEED)
    payload = {
        "seed": SEED,
        "grid": grid,
        "total": len(combos),
        "combinations": combos,
    }
    save_json(COMBINATIONS_FILE, payload)
    return payload


def load_progress(total: int):
    if not os.path.exists(PROGRESS_FILE):
        payload = {
            "next_index": 0,
            "total": total,
            "updated_at": None,
            "last_completed_id": None,
        }
        save_json(PROGRESS_FILE, payload)
        return payload

    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if payload.get("total") != total:
        payload["total"] = total
    return payload


def update_progress(next_index: int, last_completed_id: int, total: int):
    payload = {
        "next_index": next_index,
        "total": total,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "last_completed_id": last_completed_id,
    }
    save_json(PROGRESS_FILE, payload)


def load_array(path: str):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        if arr.shape == ():
            obj = arr.item()
            if hasattr(obj, "to_numpy"):
                return obj.to_numpy()
    return arr


def main():
    # Lendo dados dos arquivos .npy
    base_path = os.path.abspath(os.path.dirname(__file__))
    training_dir = os.path.join(base_path, "model_training_big4_fmsynth3_0_1")

    x = load_array(os.path.join(training_dir, "x_train_big4.npy"))
    y = load_array(os.path.join(training_dir, "y_train_big4.npy"))
    x_t = load_array(os.path.join(training_dir, "x_test_big4.npy"))
    y_t = load_array(os.path.join(training_dir, "y_test_big4.npy"))

    # Lendo o scaler do arquivo joblib
    scaler_y = joblib.load(
        os.path.join(training_dir, "scaler_y_model_training_big4_fmsynth3_0_1.save")
    )

    # Garantindo formato do input
    if x.ndim == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))
    if x_t.ndim == 2:
        x_t = x_t.reshape((x_t.shape[0], x_t.shape[1], 1))

    # Carregando combinacoes e progresso
    combos_payload = load_or_create_combinations()
    combos = combos_payload["combinations"]
    total = combos_payload["total"]

    progress = load_progress(total)
    start_idx = int(progress.get("next_index", 0))

    max_runs_env = os.environ.get("MAX_EXPERIMENTS")
    max_runs = int(max_runs_env) if max_runs_env else None

    if start_idx >= total:
        print("Todas as combinacoes ja foram executadas.")
        return

    # Parametros fixos
    epochs = 30
    patience = 5
    validation_split = 0.2

    print(f"Total de experimentos: {total}")
    print(f"Retomando a partir do indice: {start_idx}")

    end_idx = total if max_runs is None else min(total, start_idx + max_runs)

    for idx in range(start_idx, end_idx):
        combo = combos[idx]
        print(f"Combinacao {idx + 1}/{total} (id={combo['id']})")

        params = {
            "input_len": x.shape[1],
            "input_dims": x.shape[2],
            "output_dims": y.shape[1],
            "activation": combo["activation"],
            "bias_cnn": combo["bias_cnn"],
            "kernel_regularizer_cnn": combo["kernel_regularizer_cnn"],
            "bias_regressor": combo["bias_regressor"],
            "dropout_regressor": combo["dropout_regressor"],
            "kernel_regularizer_regressor": combo["kernel_regularizer_regressor"],
            "epochs": epochs,
            "patience": patience,
            "validation_split": validation_split,
            "optimizer": combo["optimizer"],
        }

        try:
            # Criando novo processo para rodar o treino
            processo = Process(
                target=run_experiment,
                args=(
                    "cnn_fmsynth3_big4_0_1",
                    f"combo_{combo['id']}",
                    x,
                    y,
                    x_t,
                    y_t,
                    scaler_y,
                    params,
                ),
            )
            processo.start()
            processo.join()
        except Exception as e:
            print(traceback.format_exc())
            print(f"Erro no novo processo de treino: {e}")

        # Atualiza progresso para permitir retomada
        update_progress(next_index=idx + 1, last_completed_id=combo["id"], total=total)


if __name__ == "__main__":
    main()
