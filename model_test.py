import gc
import math
import mlflow
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import time
import traceback

from itertools import product
from multiprocessing import Process


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
def extrator_features(input_dims, activation, bias, dropout, kernel_regularizer):
    input_layer = tf.keras.layers.Input(shape=[input_dims])

    x_0 = tf.keras.layers.Dense(
        int(input_dims),
        activation=activation,
        use_bias=bias,
        kernel_regularizer=kernel_regularizer,
    )(input_layer)

    if dropout:
        x_1 = tf.keras.layers.Dropout(0.2)(x_0)
    else:
        x_1 = x_0

    x_2 = tf.keras.layers.Dense(
        int(input_dims / 4),
        activation=activation,
        use_bias=bias,
        kernel_regularizer=kernel_regularizer,
    )(x_1)
    x_3 = tf.keras.layers.Dense(
        int(input_dims / 8),
        activation=activation,
        use_bias=bias,
        kernel_regularizer=kernel_regularizer,
    )(x_2)
    x_bottleneck = tf.keras.layers.Dense(
        int(input_dims / 16),
        activation=activation,
        name="encoder",
        use_bias=bias,
        kernel_regularizer=kernel_regularizer,
    )(x_3)

    return tf.keras.Model(input_layer, x_bottleneck, name="features")


def regressor(input_dims, output_dims, activation, bias, kernel_regularizer):
    input_layer = tf.keras.layers.Input(shape=[input_dims])

    x_0 = tf.keras.layers.Dense(
        int(input_dims / 2),
        activation=activation,
        use_bias=bias,
        kernel_regularizer=kernel_regularizer,
    )(input_layer)
    x_2 = tf.keras.layers.Dense(
        int(input_dims / 4),
        activation=activation,
        use_bias=bias,
        kernel_regularizer=kernel_regularizer,
    )(x_0)
    saidas = tf.keras.layers.Dense(
        output_dims,
        activation=None,
        name="regressor_saidas",
        use_bias=bias,
        kernel_regularizer=kernel_regularizer,
    )(x_2)

    return tf.keras.Model(input_layer, saidas, name="regressor")


def my_mean(x):
    return tf.reduce_mean(x, axis=1)


def my_max(x):
    return tf.reduce_max(x, axis=1)


def my_sum(x):
    return tf.reduce_sum(x, axis=1)


def my_min(x):
    return tf.reduce_min(x, axis=1)


def my_min(x):
    return tf.reduce_prod(x, axis=1)


def my_logsumexp(x):
    return tf.reduce_logsumexp(x, axis=1)


def my_prod(x):
    return tf.reduce_prod(x, axis=1)


def pooling_function(pooling_strategy):
    if pooling_strategy == "mean":
        return my_mean
    elif pooling_strategy == "max":
        return my_max
    elif pooling_strategy == "sum":
        return my_sum
    elif pooling_strategy == "min":
        return my_min
    elif pooling_strategy == "logsumexp":
        return my_logsumexp
    elif pooling_strategy == "prod":
        return my_prod
    else:
        raise Exception(f"Pooling not implemented: {pooling_strategy}")


def build_model(
    input_len: int,
    input_dims: int,
    output_dims: int,
    activation: str,
    bias: bool,
    kernel_regularizer: str,
    dropout: bool,
    pooling_strategy: str,
    normalization: bool,
    optimizer: str,
):
    # Camadas de entrada
    input_layer = tf.keras.layers.Input(shape=(input_len, input_dims))

    if normalization:
        x_n = tf.keras.layers.BatchNormalization()(input_layer)
    else:
        x_n = input_layer

    # Kernels
    extrator = extrator_features(
        input_dims, activation, bias, dropout, kernel_regularizer
    )

    # Features
    features = tf.keras.layers.TimeDistributed(extrator, name="features_padrao")(x_n)

    # Juntando as features
    media_features = tf.keras.layers.Lambda(pooling_function(pooling_strategy))(
        features
    )

    # Regressão
    regressao = regressor(
        media_features.shape[1], output_dims, activation, bias, kernel_regularizer
    )

    saida = regressao(media_features)

    model = tf.keras.models.Model(input_layer, saida, name="regressao")

    # return (
    #     Model(input_layer, saida, name="regressao"),
    #     Model(input_layer, media_features, name="projecao")
    # )

    # model, _ = build_models(input_len, input_dims, output_dims, activation, bias, kernel_regularizer, dropout, pooling_strategy, normalization)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae", "mse"],
    )
    # plot_model(model, show_shapes=True, expand_nested=True)
    return model


def train_model(
    x,
    y,
    input_len: int,  # x.shape[1]
    input_dims: int,  # x.shape[2]
    output_dims: int,  # y.shape[1]
    activation: str,  # elu
    bias: bool,  # False
    kernel_regularizer: str,  # None
    dropout: bool,  # True
    pooling_strategy: str,  # mean
    normalization: bool,  # True
    optimizer: str,  # adamax
    epochs: int,
    patience: int,
    validation_split: float,
):
    # Construindo o modelo
    model = build_model(
        input_len,
        input_dims,
        output_dims,
        activation,
        bias,
        kernel_regularizer,
        dropout,
        pooling_strategy,
        normalization,
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


def run_experiment(name: str, x, y, x_t, y_t, params: dict[str, any]):
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
            validation_split=0.2,
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
        y_pred = model.predict(x_t)
        test_mae = tf.keras.losses.MAE(y_t, y_pred).numpy().mean()
        test_mse = tf.keras.losses.MSE(y_t, y_pred).numpy().mean()
        test_rmse = math.sqrt(test_mse)

        # Gravando as métricas de teste
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)

        mlflow.set_tag(
            "Training Info",
            "Rede neural normal, com entrada de um áudio repartido em frames, para inferir os parâmetros do FMSynth (em resintetizar um áudio com paÂemtros aleatórios).",
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
    # Lendo os dados
    x = np.load("x.npy")
    x_t = np.load("x_t.npy")

    y = pd.read_csv("y.csv")
    y_t = pd.read_csv("y_t.csv")

    # Definindo as possíves combinações de parâmetros para o GridSearch
    activation = ["elu", "relu", "sigmoid", "tanh", "selu", "silu", "exponential"]
    bias = [True, False]
    kernel_regularizer = [None, "l1", "l2"]
    dropout = [True, False]
    pooling_strategy = ["mean", "max", "sum", "min", "logsumexp", "prod"]
    normalization = [True, False]
    optimizer = [
        "SGD",
        "RMSprop",
        "Adam",
        "Adadelta",
        "Adagrad",
        "Adamax",
        "Nadam",
        "Ftrl",
    ]

    pars_combinacoes = list(
        product(
            activation,
            bias,
            kernel_regularizer,
            dropout,
            pooling_strategy,
            normalization,
            optimizer,
        )
    )

    random.shuffle(pars_combinacoes)
    print(f"Total de experimentos a executar: {len(pars_combinacoes)}")

    # Parâmetros fixos
    epochs = 15
    patience = 5

    # Iterando as cobinações de parâmetro
    for pars in pars_combinacoes:
        params = {
            "input_len": x.shape[1],
            "input_dims": x.shape[2],
            "output_dims": y.shape[1],
            "activation": pars[0],
            "bias": pars[1],
            "kernel_regularizer": pars[2],
            "dropout": pars[3],
            "pooling_strategy": pars[4],
            "normalization": pars[5],
            "optimizer": pars[6],
            "epochs": epochs,
            "patience": patience,
        }

        try:
            # Criando novo processo para rodar o treino
            processo = Process(
                target=run_experiment,
                args=(
                    "fmsynth_model_test5",
                    x,
                    y,
                    x_t,
                    y_t,
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
