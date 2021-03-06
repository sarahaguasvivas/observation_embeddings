# -*- coding: utf-8 -*-
"""motion_embeddings.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17Y1_Lnekn_A8NSPaf4_eB4m0t_4oBmut
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
from tensorflow.keras import mixed_precision
from nptyping import NDArray, Float
#plt.style.use('classic')
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from scipy.spatial.transform import Rotation as R
import kaleido
from scipy.spatial import KDTree
mixed_precision.set_global_policy('mixed_float16')
from sklearn.cluster import KMeans

NUM_P_SAMPLES = 100000 #949207 #3800000
NUM_N_SAMPLES = 0
# 0.00468998309224844, -0.0009492466342635453, 0.12456292659044266

data = genfromtxt("data/data_Apr_01_20221.csv", delimiter=',',
                  invalid_raise = False)

def generate_negative_sample_motion_sequences(
            input_data,
            output_data,
            signal_data,
            sequence_window: int = 5,
):
    #max_signals = np.gcd(np.max(signal_data, axis = 0).tolist())
    #signals = np.divide(signal_data, max_signals)
    signals = signal_data.copy()
    n_samples = signals.shape[0]

    n_inputs = input_data.shape[1]
    n_outputs = output_data.shape[1]
    n_signal_channels = signal_data.shape[1]

    input_sequence = np.empty((n_samples - sequence_window, 0))
    output_sequence = np.empty((n_samples - sequence_window, 0))
    signal_sequence = np.empty((n_samples - sequence_window, 0))

    # NOTE(sarahaguasvivas): Negative samples are motion sequences that do not have any time progression.
    #                        in this case I chose randomly sampling from the original data for each step
    #                        of the sequence
    for i in range(sequence_window):
        input_sequence = np.concatenate((
                            input_sequence,
                            input_data[np.random.randint(-100, 50,
                                                        size = n_samples-sequence_window), :]),
                            axis = 1
        )
        signal_sequence = np.concatenate((
                            signal_sequence,
                            signal_data[np.random.randint(0, 600,
                                                        size = n_samples-sequence_window), :]),
                            axis = 1
        )
        output_sequence = np.concatenate((
                            output_sequence,
                            output_data[np.random.randint(0, output_data.shape[0]-sequence_window,
                                                        size = n_samples-sequence_window), :]),
                            axis= 1
        )
    embedding_prediction_labels = np.zeros((output_sequence.shape[0], 1))
    input_sequence = (input_sequence - input_sequence.min()) / \
                            (input_sequence.max() - input_sequence.min()) * (600)
    print(input_sequence)
    print(np.min(input_sequence), np.max(input_sequence))
    return (input_sequence[:NUM_N_SAMPLES,:], 
            signal_sequence[:NUM_N_SAMPLES,:], 
            output_sequence[:NUM_N_SAMPLES,:], 
            embedding_prediction_labels[:NUM_N_SAMPLES,:])

def generate_positive_sample_motion_sequences(
            input_data,
            output_data,
            signal_data,
            sequence_window: int = 5,
):
    #max_signals = np.max(signal_data, axis = 0)
    #signal_data = np.divide(signal_data, max_signals) - 0.5
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

    input_sequence = np.empty((n_samples - sequence_window,0))
    output_sequence = np.empty((n_samples - sequence_window, 0))
    signal_sequence = np.empty((n_samples - sequence_window, 0))

    for i in range(sequence_window):
        input_sequence = np.concatenate(
                                (input_sequence, input_data[i:-sequence_window + i, :]),
                                axis = 1)
        signal_sequence = np.concatenate(
                                (signal_sequence, signal_data[i:-sequence_window +i, :]),
                                axis = 1)
        output_sequence = np.concatenate(
                                (output_sequence, output_data[i:-sequence_window + i, :]),
                                axis = 1)
  
    prediction_labels = output_data[sequence_window:, :]
    embedding_prediction_labels = np.ones((prediction_labels.shape[0], 1))
    #input_sequence = (input_sequence - input_sequence.min()) / \
    #                        (input_sequence.max() - input_sequence.min()) * (600)
    return (input_sequence[:NUM_P_SAMPLES, :],
            signal_sequence[:NUM_P_SAMPLES, :],
            output_sequence[:NUM_P_SAMPLES, :],
            prediction_labels[:NUM_P_SAMPLES, :],
            embedding_prediction_labels[:NUM_P_SAMPLES, :])

def generate_data_and_labels(
        data,
        sequence_window: int = 5,
):
    (p_input_sequence, p_signal_sequence, p_output_sequence,
     prediction_labels, p_embedding_prediction_labels) = \
                                      generate_positive_sample_motion_sequences(
        input_data=data[:, 14:],
        output_data=data[:, 11:14],
        signal_data=data[:, :11],
        sequence_window=sequence_window)
    (n_input_sequence, n_signal_sequence, n_output_sequence,
     n_embedding_prediction_labels) = generate_negative_sample_motion_sequences(
        input_data=data[:, 14:],
        output_data=data[:, 11:14],
        signal_data=data[:, :11],
        sequence_window=sequence_window
    )
    assert p_input_sequence.shape[1] == n_input_sequence.shape[1], \
                        "positive and negative input sequence dimension mismatch"
    assert p_signal_sequence.shape[1] == n_signal_sequence.shape[1], \
                        "positive and negative signal sequence dimension mismatch"
    #X = np.concatenate((p_input_sequence, p_signal_sequence), axis=1)
    #X = np.concatenate((X, np.concatenate((n_input_sequence, n_signal_sequence), 
    #                                      axis=1)), axis=0)
    X = np.concatenate((p_signal_sequence, n_signal_sequence), axis = 0)
    y = np.concatenate((p_embedding_prediction_labels, 
                        n_embedding_prediction_labels), axis=0)
    return (X, y)


def generate_positive_data_and_labels(
        data,
        sequence_window: int = 5,
):
    (p_input_sequence, p_signal_sequence, p_output_sequence,
     prediction_labels, p_embedding_prediction_labels) = \
                                      generate_positive_sample_motion_sequences(
        input_data=data[:, 14:],
        output_data=data[:, 11:14],
        signal_data=data[:, :11],
        sequence_window=sequence_window)
    
    X = p_signal_sequence #np.concatenate((p_signal_sequence, n_signal_sequence), axis = 0)
  
    y = prediction_labels #np.concatenate((p_embedding_prediction_labels, 
                                      #n_embedding_prediction_labels), axis=0)
    r = R.from_rotvec([0, 0, np.pi/4.])
    y = r.apply(y)
    y = min_max_normalization(y, -0.5, 0.5)
    return (X, y)

def distance_metric(x):
    n = keras.backend.permute_dimensions(x, pattern=(1, 0, 2))
    a, b = n[0], n[1]
    return keras.backend.mean(x, axis = 1)

def min_max_normalization(x, new_min, new_max):
  return (x - np.min(x, axis = 0))/ (np.max(x, axis = 0) - \
                          np.min(x, axis = 0)) * (new_max - new_min) + new_min

def get_model(
              embedding_output_dim,
              vocabulary_size,
):
    input_layer = keras.layers.Input(shape=(1), dtype = "float16")
    x = keras.layers.Embedding(input_dim=vocabulary_size,
                                output_dim=embedding_output_dim,
                                embeddings_initializer='random_normal'
                                )(input_layer)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(3, activation='tanh')(x)
    model = keras.Model(inputs=input_layer, outputs=x)
    model.summary()
    return model

def generate_motion_sequence_embedding(
    data,
    labels,
    embedding_output_dim = 3,
    record = False,
):
    data =  data.astype(np.float16)
    kmeans = KMeans(n_clusters = 300, random_state=0).fit(data)
    model = get_model(embedding_output_dim = embedding_output_dim,
                      vocabulary_size = 300)
    model.compile(loss='mse', optimizer="adam", metrics=["mse"])
    weights = None
    if record:
        weights = []
        save = keras.callbacks.LambdaCallback(on_epoch_end = lambda batch,
                                             logs: weights.append(model
                                             .layers[0].get_weights()[0]))
        kfold = TimeSeriesSplit(n_splits = 5)
        k_fold_results = []
        for train,test in kfold.split(data, labels):
          X_train = data[train]
          y_train = labels[train]
          train_x = kmeans.labels_[train].reshape(-1, 1)
          model.fit(train_x, y_train, epochs = 10, verbose = 1, batch_size = 1000,
                      callbacks= [save])
    else:
      kfold = TimeSeriesSplit(n_splits = 5)
      k_fold_results = []
      for train,test in kfold.split(data, labels):
        X_train = data[train]
        y_train = labels[train]
        train_x = kmeans.labels_[train].reshape(-1, 1)
        model.fit(train_x, y_train, epochs = 5, verbose = 1, batch_size = 1000)
    return (model, kmeans, weights)

if __name__=='__main__':
    sequence_window = 20
    SUB_SAMPLES = 50000
    EMBEDDING_LAYER = 1
    TRAINING = False
    (X, y) = generate_positive_data_and_labels(data, sequence_window)
    if TRAINING:
        (embedding_model, clusters, weight_logs) = generate_motion_sequence_embedding(X, y, 3)
        embedding_model.save('models/embedding_model.hdf5', 'hdf5')

    nn_model = keras.models.load_model('models/embedding_model.hdf5',
                                                  compile = False)

    kmeans = KMeans(n_clusters = 3, random_state=0).fit(y)
    df = pd.DataFrame(data = y[:SUB_SAMPLES, :],
                      columns = ['x', 'y', 'z'])
    df['partitions'] = kmeans.labels_[:SUB_SAMPLES].astype(str).reshape(-1, 1)
    fig = px.scatter_3d(df, x = 'x', y = 'y', z = 'z', color = 'partitions')
    fig.show()
    fig.write_image('task_space.png', engine = 'kaleido')

    embedding = nn_model.predict(clusters.predict(X[:SUB_SAMPLES, :]))
    embedding_layer_output = keras.Model(nn_model.input,
                                         nn_model.layers[EMBEDDING_LAYER].output)
    embedding_predict = embedding_layer_output.predict(clusters.predict(X[:SUB_SAMPLES, :]))
    df_embedding = pd.DataFrame(data = embedding_predict[:, 0, :],
                      columns = ['dim_0', 'dim_1', 'dim_2'])

    df_embedding['partitions'] = kmeans.labels_[:SUB_SAMPLES].astype(str).reshape(-1, 1)
    fig = px.scatter_3d(df_embedding, x = 'dim_0', y = 'dim_1',
                        z = 'dim_2', color = 'partitions')
    fig.show()