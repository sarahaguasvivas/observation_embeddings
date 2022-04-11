"""vae.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hBrAapL3hO_CBBNwoE8dgzedAK_nYvsN
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow import keras
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
import tensorflow as tf
import copy
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple
from numpy import genfromtxt
from keras import mixed_precision
from typing import List, Tuple
# mixed_precision.set_global_policy('mixed_float16')
from sklearn.cluster import KMeans
from nptyping import NDArray, Float

NUM_N_SAMPLES = 0
NUM_P_SAMPLES = 949207  # 900000
data = genfromtxt("data/data_Apr_01_20221.csv", delimiter=',',
                  invalid_raise=False)

class Autoencoder(keras.Model):
    def __init__(self, embed_dim, encoder, decoder, task):
        super(Autoencoder, self).__init__()
        self.embed_dim = embed_dim
        self.encoder = encoder
        self.decoder = decoder
        self.task = task

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        task_done = self.task(encoded)
        return [decoded, task_done]


def get_simple_encoder(latent_dim=3, seq_window=3):
    encoder_inputs = keras.Input(shape=(11 * seq_window,))
    x = keras.layers.Dense(55, activation='relu')(encoder_inputs)
    x = keras.layers.Dense(22, activation='relu')(x)
    x = keras.layers.Dense(latent_dim)(x)
    encoder = keras.Model(encoder_inputs, x, name='encoder')
    encoder.summary()
    return encoder

def get_simple_decoder(latent_dim=3, seq_window=3):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = keras.layers.Dense(22, activation='sigmoid')(latent_inputs)
    x = keras.layers.Dense(55, activation='sigmoid')(x)
    x = keras.layers.Dense(seq_window * 11)(x)
    decoder = keras.Model(latent_inputs, x, name='decoder')
    decoder.summary()
    return decoder

def get_task(latent_dim=3, label_size=3):
    latent = keras.Input(shape=(latent_dim,))
    x = keras.layers.Reshape((-1, 1))(latent)
    x = keras.layers.GRU(20)(x)
    x = keras.layers.Dense(label_size)(x)
    task = keras.Model(latent, x, name='task')
    task.summary()
    return task

def min_max_normalization(x, new_min, new_max):
    current_min = np.min(x, axis=0)
    current_max = np.max(x, axis=0)
    normalized = (x - np.min(x, axis=0)) / (current_max - \
                                            current_min) * (new_max - new_min) + new_min
    normalized[np.isnan(normalized)] = 0
    return normalized

def generate_positive_sample_motion_sequences(
        input_data,
        output_data,
        signal_data,
        sequence_window=5):
    signal_data = np.clip(signal_data, 0, 143)
    n_samples = signal_data.shape[0]
    output_data = output_data - \
                  np.array([0.00468998309224844,
                            -0.0009492466342635453,
                            0.12456292659044266])

    input_data = min_max_normalization(input_data, -0.5, 0.5)
    n_inputs = input_data.shape[1]
    n_outputs = output_data.shape[1]
    n_signal_channels = signal_data.shape[1]

    input_sequence = np.empty((n_samples - sequence_window, 0))
    output_sequence = np.empty((n_samples - sequence_window, 0))
    signal_sequence = np.empty((n_samples - sequence_window, 0))

    for i in range(sequence_window):
        input_sequence = np.concatenate(
            (input_sequence, input_data[i:-sequence_window + i, :]),
            axis=1)
        signal_sequence = np.concatenate(
            (signal_sequence, signal_data[i:-sequence_window + i, :]),
            axis=1)
        output_sequence = np.concatenate(
            (output_sequence, output_data[i:-sequence_window + i, :]),
            axis=1)

    prediction_labels = output_data[sequence_window:, :]
    embedding_prediction_labels = np.ones((prediction_labels.shape[0], 1))
    # input_sequence = (input_sequence - input_sequence.min()) / \
    #                        (input_sequence.max() - input_sequence.min()) * (600)
    return (input_sequence[:NUM_P_SAMPLES, :],
            signal_sequence[:NUM_P_SAMPLES, :],
            output_sequence[:NUM_P_SAMPLES, :],
            prediction_labels[:NUM_P_SAMPLES, :],
            embedding_prediction_labels[:NUM_P_SAMPLES, :])


def generate_positive_data_and_labels(
        data,
        sequence_window=5):
    (p_input_sequence, p_signal_sequence, p_output_sequence,
     prediction_labels, p_embedding_prediction_labels) = \
        generate_positive_sample_motion_sequences(
            input_data=data[:, 14:],
            output_data=data[:, 11:14],
            signal_data=data[:, :11],
            sequence_window=sequence_window)

    X = p_signal_sequence  # np.concatenate((p_signal_sequence, n_signal_sequence), axis = 0)
    X = min_max_normalization(X, -1, 1)
    y = prediction_labels  # np.concatenate((p_embedding_prediction_labels,
    # n_embedding_prediction_labels), axis=0)
    r = R.from_rotvec([0, 0, np.pi / 4.])
    y = r.apply(y)
    y = min_max_normalization(y, -1, 1)
    return (X, y)

def generate_motion_sequence_embedding_ae(
        data,
        labels,
        embedding_output_dim=3,
        sequence_window=20,
        record=True
):
    data = data.astype(np.float16)
    encoder = get_simple_encoder(latent_dim=embedding_output_dim,
                                 seq_window=sequence_window)
    decoder = get_simple_decoder(latent_dim=embedding_output_dim,
                                 seq_window=sequence_window)
    task = get_task(latent_dim=embedding_output_dim,
                    label_size=labels.shape[1])
    model = Autoencoder(embedding_output_dim, encoder, decoder, task)

    model.compile(optimizer="adam", loss=['kl_divergence', 'mse'])

    weights = None
    if record:
        weights = []
        save = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch,
                                                                  logs: weights.append(model
                                                                                       .layers[0].get_weights()[0]))
        kfold = TimeSeriesSplit(n_splits=10)
        k_fold_results = []
        for train, test in kfold.split(data, labels):
            x_train = data[train]
            y_train = labels[train]
            model.fit(x_train, [x_train, y_train], epochs=10, verbose=1, batch_size=1000,
                      callbacks=[save], validation_data=(X[test], [X[test], y[test]]))
    else:
        kfold = TimeSeriesSplit(n_splits=5)
        k_fold_results = []
        for train, test in kfold.split(data, labels):
            X_train = data[train]
            y_train = labels[train]
            model.fit([X_train, X_train], labels, epochs=10, verbose=1, batch_size=1000)
    return (model, encoder, decoder, weights)

if __name__ == '__main__':
    import plotly.express as px
    import pandas as pd
    sequence_window = 10
    SUB_SAMPLES = 40000
    TRAINING = True

    (X, y) = generate_positive_data_and_labels(data, sequence_window)
    if TRAINING:
        (model, encoder, decoder, weight_logs) = \
            generate_motion_sequence_embedding_ae(X, y, 3, sequence_window)
        encoder.save('encoder_ae.hdf5', 'hdf5')
        decoder.save('decoder_ae.hdf5', 'hdf5')

    encoder = keras.models.load_model('encoder_ae.hdf5', compile=False)

    kmeans = KMeans(n_clusters=3, random_state=0).fit(y[:SUB_SAMPLES, :])

    df = pd.DataFrame(data=y[:SUB_SAMPLES, :],
                      columns=['x', 'y', 'z'])
    df['partitions'] = kmeans.labels_[:SUB_SAMPLES].astype(str).reshape(-1, 1)
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='partitions')
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[-0.1, 0.1], ),
            yaxis=dict(nticks=4, range=[-0.03, 0.03], ),
            zaxis=dict(nticks=4, range=[-0.06, 0.03], ), ),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10))
    fig.show()

    embedding_output = encoder.predict(X[:SUB_SAMPLES, :])

    df_embedding = pd.DataFrame(data=embedding_output[:, :3],
                                columns=['dim_0', 'dim_1', 'dim_2'])

    df_embedding['partitions'] = kmeans.labels_[:SUB_SAMPLES].astype(str).reshape(-1, 1)
    fig = px.scatter_3d(df_embedding, x='dim_0', y='dim_1',
                        z='dim_2', color='partitions')

    fig.show()
