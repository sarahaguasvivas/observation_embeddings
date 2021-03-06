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
from sklearn.manifold import TSNE

NUM_N_SAMPLES = 0
NUM_P_SAMPLES = 949207  # 900000
data = genfromtxt("../data/data_Apr_01_20221.csv", delimiter=',',
                  invalid_raise=False)

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
    X = np.hstack((X, p_input_sequence))
    y = prediction_labels  # np.concatenate((p_embedding_prediction_labels,
    # n_embedding_prediction_labels), axis=0)
    r = R.from_rotvec([0, 0, np.pi / 4.])
    y = r.apply(y)
    y = min_max_normalization(y, -1, 1)
    return (X, y)

def normalize_coordinates(x):
    return x / (np.max(x, axis = 0) - np.min(x, axis = 0))

if __name__ == '__main__':
    import plotly.express as px
    import pandas as pd
    sequence_window = 1
    SUB_SAMPLES = 4000 #16000
    TRAINING = False
    EMBEDDING_SIZE = 3
    EARLY_EXAGGERATION_RATE = 12.
    (X, y) = generate_positive_data_and_labels(data, sequence_window)

    kmeans = KMeans(n_clusters=10, random_state=0).fit(y[:SUB_SAMPLES, :])
    df = pd.DataFrame(data=y[:SUB_SAMPLES, :],
                      columns=['x', 'y', 'z'])

    df['partitions'] = kmeans.labels_[:SUB_SAMPLES].astype(str).reshape(-1, 1)
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='partitions')
    fig.show()

    encoded = TSNE(n_components = EMBEDDING_SIZE,
                   early_exaggeration = EARLY_EXAGGERATION_RATE,
                   n_iter = 10000,
                   perplexity = SUB_SAMPLES / 100,
                   learning_rate = 500, #max(SUB_SAMPLES / EARLY_EXAGGERATION_RATE, 50),
                   init='pca',
                   angle = 0.3,
                   random_state = 2,
                   verbose = 1,
                   n_jobs = 4).fit_transform(X[:SUB_SAMPLES, :].astype(np.float16))

    to_save = np.hstack((X[:SUB_SAMPLES, :], encoded))
    to_save = np.hstack((to_save, y[:SUB_SAMPLES, :]))
    print(to_save.shape)
    np.savetxt("signal_tsne.csv", to_save, delimiter = ',')

    columns = ['dim_' + str(i) for i in range(EMBEDDING_SIZE)]
    df_embedding = pd.DataFrame(data=encoded,
                                columns=columns)

    df_embedding['partitions'] = kmeans.labels_[:SUB_SAMPLES].astype(str).reshape(-1, 1)
    #fig = px.scatter_3d(df_embedding, x='dim_0', y='dim_1', z= 'dim_2', color='partitions')
    #fig.show()

    for i in range(EMBEDDING_SIZE):
        for j in range(i + 1, EMBEDDING_SIZE):
            dim0 = 'dim_' + str(i)
            dim1 = 'dim_' + str(j)
            fig = px.scatter(df_embedding, x = dim0, y = dim1, color = 'partitions')
            fig.show()

    fig = px.scatter_3d(df_embedding, x = 'dim_0', y = 'dim_1', z = 'dim_2', color = 'partitions')
    fig.show()
