import joblib
import math
import mlflow
import numpy as np
import random
import tensorflow as tf
import time
import traceback

from itertools import product
from multiprocessing import Process

import numpy as np
import random


from keras.layers import (
    Dense,
    Input,
    BatchNormalization,
    Conv1D,
    Flatten,
    Dropout,
    MaxPooling1D,
)
from keras.models import Model
import tensorflow as tf


def config_gpu():
    # Configurando para não alocar diretamente toda a memória da GPU (alocar conforme necessário)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(
                    gpu, True
                )  # Aloca memória conforme necessário
        except RuntimeError as e:
            print(e)
            exit(1)


# Funções para construção do modelo
def regressor(
    input_dims,
    output_dims,
    activation_regressor,
    bias_regressor,
    dropout_regressor: float,
    kernel_regularizer_regressor: str,
):
    input_layer = Input(shape=[input_dims])

    x_0 = Dense(
        int(input_dims / 2),
        activation=activation_regressor,
        use_bias=bias_regressor,
        kernel_regularizer=kernel_regularizer_regressor,
    )(input_layer)
    x_0 = Dropout(dropout_regressor)(x_0)
    x_2 = Dense(
        int(input_dims / 4), activation=activation_regressor, use_bias=bias_regressor
    )(x_0)
    saidas = Dense(
        output_dims, activation=None, name="regressor_saidas", use_bias=bias_regressor
    )(x_2)

    return Model(input_layer, saidas, name="regressor")


def build_model(
    input_len,
    input_dims,
    output_dims,
    activation_cnn,
    bias_cnn,
    kernel_regularizer_cnn,
    activation_regressor,
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
        activation=activation_cnn,
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
        activation=activation_cnn,
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
        activation=activation_cnn,
        input_shape=(input_len, 1),
        kernel_regularizer=kernel_regularizer_cnn,
        use_bias=bias_cnn,
    )
    pooling1_2 = MaxPooling1D(pool_size=16)

    features1_3 = extrator1_3(features1_2)
    features1_2 = pooling1_2(features1_2)

    features1_flatten1 = Flatten()(features1_3)

    features1_flatten1_normalized = BatchNormalization()(features1_flatten1)

    # Regressão
    regressao = regressor(
        features1_flatten1_normalized.shape[1],
        output_dims,
        activation_regressor,
        bias_regressor,
        dropout_regressor,
        kernel_regularizer_regressor,
    )

    saida = regressao(features1_flatten1)

    model = Model(input_layer, saida, name="complete")
    # Model(input_layer, features1_flatten1, name="projecao"),
    # Model(features1_flatten1, saida, name="regressao"),
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mse"])

    return model


def train_model(
    x,
    y,
    input_len: int,
    input_dims: int,  # x.shape[2]
    output_dims: int,  # y.shape[1]
    activation_cnn,
    bias_cnn,
    kernel_regularizer_cnn,
    activation_regressor,
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
        activation_cnn,
        bias_cnn,
        kernel_regularizer_cnn,
        activation_regressor,
        bias_regressor,
        dropout_regressor,
        kernel_regularizer_regressor,
        optimizer,
    )

    # Callback para recuperar o melhor peso, e parar quando ficar três épocas sem melhora
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )

    # Treinando o modelo
    return model, model.fit(
        x, y, epochs=epochs, validation_split=validation_split, callbacks=[callback]
    )


def run_experiment(name: str, x, y, x_t, y_t, scaler_y, params: dict[str, any]):
    config_gpu()

    # Iniciando o MLFlow
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment(name)

    try:
        mlflow.start_run()

        # Gravando os parâmetros
        mlflow.log_params(params)

        # Treinando a rede
        inicio = time.time()
        model, history = train_model(
            x=x,
            y=y,
            **params,
        )
        tempo_decorrido = time.time() - inicio

        # Gravando as métricas do treino
        mlflow.log_metric("training_time", tempo_decorrido)

        mse = min(history.history["mse"])
        val_mse = min(history.history["val_mse"])
        val_rmse = math.sqrt(val_mse)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("val_mse", val_mse)
        mlflow.log_metric("val_rmse", val_rmse)

        mae = min(history.history["mae"])
        val_mae = min(history.history["val_mse"])

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("val_mae", val_mae)

        # Inferindo o teste
        y_pred_norm = model.predict(x_t)
        y_pred = scaler_y.inverse_transform(y_pred_norm)

        test_mae = tf.keras.losses.MAE(y_t, y_pred).numpy().mean()
        test_mse = tf.keras.losses.MSE(y_t, y_pred).numpy().mean()
        test_rmse = math.sqrt(test_mse)

        # Gravando as métricas de teste
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)

        mlflow.set_tag(
            "Training Info",
            "CNN FMSynth.",
        )
    except Exception as e:
        print(traceback.format_exc())
        print(f"Erro no treino: {e}")
    finally:
        mlflow.end_run()

        # # Limpeza de memória após o treino anterior
        # del model  # Remove o modelo da memória
        # tf.keras.backend.clear_session()  # Limpa o backend do Keras
        # gc.collect()  # Opcional: chama o coletor de lixo


def main():
    # Lendo dados dos arquivos .npy
    base_path = "/home/sergio/@pessoal/fmsynth"
    x = np.load(f"{base_path}/dataset_big2/x_train_big2.npy")
    y = np.load(f"{base_path}/dataset_big2/y_train_big2.npy")
    x_t = np.load(f"{base_path}/dataset_big2/x_test_big2.npy")
    y_t = np.load(f"{base_path}/dataset_big2/y_test_big2.npy")

    # Lendo o scaler do arquivo joblib
    scaler_y = joblib.load(f"{base_path}/scaler_y_conv_1_sec_v1_0.save")

    # Reshape x
    x = x.reshape((x.shape[0], x.shape[1], 1))

    # Definindo as possíves combinações de parâmetros para o GridSearch
    activation_cnn = [
        "gelu",
        "swish",
        "elu",
        "relu",
        "tanh",
        "selu",
        "silu",
        "exponential",
    ]
    bias_cnn = [True, False]
    kernel_regularizer_cnn = [None, "l1", "l2"]

    activation_regressor = [
        "gelu",
        "swish",
        "elu",
        "relu",
        "tanh",
        "selu",
        "silu",
        "exponential",
    ]
    bias_regressor = [True, False]
    dropout_regressor = [0, 0.2, 0.3, 0.5]
    kernel_regularizer_regressor = [None, "l1", "l2"]

    optimizer = [
        "AdamW",
        "RMSprop",
        "Adam",
        "Nadam",
        "Adagrad",
        "Adamax",
    ]

    pars_combinacoes = list(
        product(
            activation_cnn,
            bias_cnn,
            kernel_regularizer_cnn,
            activation_regressor,
            bias_regressor,
            dropout_regressor,
            kernel_regularizer_regressor,
            optimizer,
        )
    )

    random.shuffle(pars_combinacoes)
    print(f"Total de experimentos a executar: {len(pars_combinacoes)}")

    # Parâmetros fixos
    epochs = 50
    patience = 5
    validation_split = 0.2

    # Iterando as cobinações de parâmetro
    i = 0
    for pars in pars_combinacoes:
        i += 1
        print(f"Cominação {i}")

        params = {
            "input_len": x.shape[1],
            "input_dims": x.shape[2],
            "output_dims": y.shape[1],
            "activation_cnn": pars[0],
            "bias_cnn": pars[1],
            "kernel_regularizer_cnn": pars[2],
            "activation_regressor": pars[3],
            "bias_regressor": pars[4],
            "dropout_regressor": pars[5],
            "kernel_regularizer_regressor": pars[6],
            "epochs": epochs,
            "patience": patience,
            "validation_split": validation_split,
            "optimizer": pars[7],
        }

        try:
            # Criando novo processo para rodar o treino
            processo = Process(
                target=run_experiment,
                args=(
                    "cnn_fmsynth3",
                    x,
                    y,
                    x_t,
                    y_t,
                    scaler_y,
                    params,
                ),
            )
            processo.start()

            # Esperando o processo terminar
            processo.join()
        except Exception as e:
            print(traceback.format_exc())
            print(f"Erro no novo processo de treino: {e}")


if __name__ == "__main__":
    main()
