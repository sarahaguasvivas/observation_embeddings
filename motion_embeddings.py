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
mixed_precision.set_global_policy('mixed_float16')

!pip install nptyping
from nptyping import NDArray, Float
from google.colab import drive
drive.mount('/content/gdrive')

NUM_P_SAMPLES = 949207 #3800000 
NUM_N_SAMPLES = 0
# 0.00468998309224844, -0.0009492466342635453, 0.12456292659044266
#np.random.seed(2449890217)
print(np.random.get_state()[1][0])

def generate_negative_sample_motion_sequences(
            input_data: NDArray[float],
            output_data: NDArray[float],
            signal_data: NDArray[float],
            sequence_window: int = 5,
) -> (NDArray[float], NDArray[float], NDArray[float], NDArray[float]):
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
            input_data: NDArray[float],
            output_data: NDArray[float],
            signal_data: NDArray[float],
            sequence_window: int = 5,
) -> (NDArray[float], NDArray[float], NDArray[float], 
      NDArray[float], NDArray[float]):
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
        data: NDArray[float],
        sequence_window: int = 5,
)-> (
    NDArray[float], NDArray[float]):
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
        data: NDArray[float],
        sequence_window: int = 5,
)-> (
    NDArray[float], NDArray[float]):
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
    return (X, y)

def distance_metric(x):
    n = keras.backend.permute_dimensions(x, pattern=(1, 0, 2))
    a, b = n[0], n[1]
    return keras.backend.mean(x, axis = 1)

def min_max_normalization(x, new_min, new_max):
  return (x - np.min(x, axis = 0))/ (np.max(x, axis = 0) - \
                          np.min(x, axis = 0)) * (new_max - new_min) + new_min

def generate_motion_sequence_embedding(
    data: NDArray[float],
    labels: NDArray[float],
    embedding_output_dim: int = 3,
    record: bool = True,
):
    data = data.astype(np.float16)
    # actual sequences that are labeled yes belong to the vocabulary
    vocabulary_size : int = 200 #data.shape[0]
  
    model = keras.Sequential()
    model.add(keras.layers.Embedding(input_dim = vocabulary_size,
                                     output_dim = embedding_output_dim,
                                     input_length = data.shape[1],
                                     embeddings_initializer = 'glorot_uniform'
                                     ))
    model.add(keras.layers.Lambda(distance_metric, output_shape = (3,), 
                                  name ="Dot_Product"))
    model.add(keras.layers.Dense(3, activation=None))
    model.compile(loss='mse', optimizer="adam", metrics=["mse"])
    weights = None
    if record:
        weights = []
        save = keras.callbacks.LambdaCallback(on_epoch_end = lambda batch,
                                             logs: weights.append(model
                                             .layers[0].get_weights()[0]))
        kfold = TimeSeriesSplit(n_splits = 10)
        k_fold_results = []
        for train,test in kfold.split(data, labels):
          X_train = data[train]
          y_train = labels[train]
          model.fit(X_train, y_train, epochs = 10, verbose = 1, batch_size = 20, 
                      callbacks= [save])
    else:
      kfold = TimeSeriesSplit(n_splits = 10)
      k_fold_results = []
      for train,test in kfold.split(data, labels):
        X_train = data[train]
        y_train = labels[train]
        model.fit(data, labels, epochs = 10, verbose = 1, batch_size = 20)
    return (model, weights)

def generate_motion_sequence_embedding1(
    data: NDArray[float],
    labels: NDArray[float],
    embedding_output_dim: int = 3,
    record: bool = True,
):

    data = data.astype(np.float16)
    vocabulary_size : int = 600
    model = keras.Sequential()
    model.add(keras.layers.Embedding(input_dim = vocabulary_size,
                                     output_dim = embedding_output_dim,
                                     input_length = data.shape[1],
                                     embeddings_initializer = 'random_normal'
                                     ))
    model.add(keras.layers.Lambda(distance_metric, output_shape = (3,), 
                                  name ="Dot_Product"))
    model.add(keras.layers.Dense(3, activation=None))
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
          model.fit(X_train, y_train, epochs = 10, verbose = 1, batch_size = 1000, 
                      callbacks= [save])
    else:
      kfold = TimeSeriesSplit(n_splits = 5)
      k_fold_results = []
      for train,test in kfold.split(data, labels):
        X_train = data[train]
        y_train = labels[train]
        model.fit(data, labels, epochs = 10, verbose = 1, batch_size = 1000)
    return (model, weights)

# https://keras.io/examples/generative/vae/

data = genfromtxt("gdrive/My Drive/Collabs/shepherd/iser_2020/figures/" +
                      "figure_3_sys_id/data_models_other/" +
                        "data_Apr_01_20221.csv", delimiter=',', 
                        invalid_raise = False)

data.shape

sequence_window = 20
#(X, y) = generate_data_and_labels(data, sequence_window)
(X, y)  = generate_positive_data_and_labels(data, sequence_window)

plt.figure()
plt.plot(X)
plt.show()

plt.figure()
plt.plot(y)
plt.show()

#(embedding_model, weight_logs) = generate_motion_sequence_embedding(X, y, 3)
(embedding_model, weight_logs) = generate_motion_sequence_embedding1(X, y, 3)
embedding_model.save('embedding_model.hdf5', 'hdf5')

nn_model = keras.models.load_model('embedding_model.hdf5', 
                                              compile = False)

# 3d embeddings 3d position 
(_, _, p_output_sequence,
     prediction_labels, _) = generate_positive_sample_motion_sequences(
        input_data=data[:, 14:],
        output_data=data[:, 11:14],
        signal_data=data[:, :11],
        sequence_window=sequence_window)
positive_samples = X[:NUM_P_SAMPLES, :]

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=0).fit(prediction_labels)

#plt.style.use('classic')
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from scipy.spatial.transform import Rotation as R

!pip install -U kaleido
import kaleido

SUB_SAMPLES = 20000

angles = np.linspace(-np.pi/2., np.pi, 1000)
radius = 10

target = np.array([radius * np.sin(angles) * np.cos(angles),
          radius * np.sin(angles) * np.sin(angles),
          radius * np.cos(angles)]).T


y = prediction_labels[:SUB_SAMPLES, :]
df = pd.DataFrame(data = y,
                  columns = ['x', 'y', 'z'])
target_df = pd.DataFrame(data = target, 
                      columns = ['x', 'y', 'z'])
df['partitions'] = kmeans.labels_[:SUB_SAMPLES].astype(str).reshape(-1, 1)
fig = px.scatter_3d(df, x = 'x', y = 'y', z = 'z', color = 'partitions')
#fig = px.scatter_3d(target_df, x = 'x', y = 'y', z = 'z', color = 'z')
#fig.update_layout(
#    scene = dict(
#        xaxis = dict(nticks=4, range=[-0.1,0.1],),
#                     yaxis = dict(nticks=4, range=[-0.03,0.03],),
#                     zaxis = dict(nticks=4, range=[-0.06,0.03],),),
#    width=700,
#    margin=dict(r=20, l=10, b=10, t=10))
fig.show()
fig.write_image('task_space.png', engine = 'kaleido')

radius = np.linspace(0, 15./1000., 2000)
t = np.radians(np.linspace(0, 2000, 2000))

target = np.array([
           radius*np.cos(t),
           radius*np.sin(t),
           np.linspace(0, 4/1000, len(t))
         ]).T

r = R.from_rotvec([0, np.pi/2., 1])
target = r.apply(target)
#target = target + np.array([-6/1000, -1/1000, -135./1000.])


med = np.median(y, axis = 0)
print(med)
y = prediction_labels[:SUB_SAMPLES, :] - np.median(y, axis = 0)


df = pd.DataFrame(data = y,
                  columns = ['x', 'y', 'z'])
target_df = pd.DataFrame(data = target, 
                      columns = ['x', 'y', 'z'])
df['partitions'] = kmeans.labels_[:SUB_SAMPLES].astype(str).reshape(-1, 1)
fig = go.Figure()
fig.add_trace(go.Scatter3d(x = y[:, 0], y = y[:, 1], z = y[:, 2], 
                              mode = 'markers', 
                           marker=dict(size=2)))
fig.add_trace(go.Scatter3d(x = target[:, 0], y = target[:, 1],
                           z = target[:, 2], mode='markers', 
                           marker=dict(size=2)))
#fig = px.scatter_3d(target_df, x = 'x', y = 'y', z = 'z', color = 'z')
#fig.update_layout(
#    scene = dict(
#        xaxis = dict(nticks=4, range=[-0.1,0.1],),
#                     yaxis = dict(nticks=4, range=[-0.03,0.03],),
#                     zaxis = dict(nticks=4, range=[-0.06,0.03],),),
#    width=700,
#    margin=dict(r=20, l=10, b=10, t=10))
fig.show()

# Easy way to get intermediate layers:
embedding = nn_model.predict(X[:SUB_SAMPLES, :])
print(embedding.shape)
embedding_layer_output = keras.Model(nn_model.input, nn_model.layers[1].output)
input = X[:SUB_SAMPLES, :]
em_input = input.copy()

#em_input[:, [6, 17, 28, 39]] = 0
#em_input[:, [5, 16, 27, 38]] = 0
print(em_input.max(axis = 0))
plt.figure()
plt.plot(em_input)
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.show()

embedding_output = embedding_layer_output.predict(em_input)

print(embedding_output.shape)
embedding_output = embedding_output.reshape(-1, 3)
embedding_output.shape

df_embedding = pd.DataFrame(data = embedding_output,
                  columns = ['dim_0', 'dim_1', 'dim_2'])

df_embedding['partitions'] = kmeans.labels_[:SUB_SAMPLES].astype(str).reshape(-1, 1)
fig = px.scatter_3d(df_embedding, x = 'dim_0', y = 'dim_1', 
                    z = 'dim_2', color = 'partitions')
#fig.update_layout(
#    scene = dict(
#        xaxis = dict(nticks=4, range=[-0.5,0.3],),
#                     yaxis = dict(nticks=4, range=[-0.2,0.4],),
#                     zaxis = dict(nticks=4, range=[-0.3,0.2],),),
#    width=700,
#    margin=dict(r=20, l=10, b=10, t=10))
fig.show()

#generate_gif_files = False
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')

#ax.scatter(embedding_output[:, 0],
#           embedding_output[:, 1],
#           embedding_output[:, 2], 
#            c = kmeans.labels_[:SUB_SAMPLES], s = 100)

#ax.set_xlabel('dim 0')
#ax.set_ylabel('dim 1')
#ax.set_zlabel('dim 2')
#ax.view_init(80,0)
#if generate_gif_files:
#  ax.view_init(elev = 0, azim = 0)
#  for ii in range(0,360):
#      ax.view_init(elev=10., azim=ii)
#      plt.savefig("gif_rotating/movie_embedding%d.png" % ii)
#else:
#  plt.tight_layout()
#  plt.savefig("embedding_output_2_removed.pdf", format = 'pdf')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(weight_logs[-1][:, 0],
           weight_logs[-1][:, 1],
           weight_logs[-1][:, 2], 
          )

ax.set_xlabel('dim 0')
ax.set_ylabel('dim 1')
ax.set_zlabel('dim 2')

plt.title("3D Embedding for Sensor")
ax.view_init(10,0)
plt.savefig("embedding_output.png", dpi=300)

plt.figure()
plt.plot(X[:SUB_SAMPLES, :].reshape(-1, 11)[:, ::-1], alpha = 0.4)
plt.title("Observation Space (Raw)")
plt.legend()
plt.savefig("sensor_readings.png", dpi=300, transparent=True)

from google.colab import drive
drive.mount('/content/drive')

from google.colab import files

!zip -r /content/file.zip /content/gif_rotating
files.download('file.zip')



# Commented out IPython magic to ensure Python compatibility.
# %%capture
# from matplotlib.animation import FuncAnimation
# from IPython.display import HTML
# C_SAMPLES= weight_logs[0].shape[0]
# cs = np.linspace(0, 255, C_SAMPLES).reshape(C_SAMPLES, 1)
# 
# # Create the plots
# with plt.style.context("default"):
#     fig, ax = plt.subplots(figsize=(10, 5))
#     fig.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9)
#     ax.set_xlabel("$x_0$"), ax.set_ylabel("$x_1$")
#     ax.set_title("Training 3D motion Embedding")
#     scat = ax.scatter(weight_logs[0][:, 0], weight_logs[0][:, 1],c=cs, s=10)
# 
#     def init():
#         ax.set_xlim(weight_logs[-1][:,0].min(), weight_logs[-1][:,0].max())
#         ax.set_ylim(weight_logs[-1][:,1].min(), weight_logs[-1][:,1].max())
#         return scat,
#     def update(i):
#         scat.set_offsets(weight_logs[i][:, :2])
#         
#         return scat,
#     nw = len(weight_logs) - 1
#     nf, power = 30 * 10, 2
#     frames = np.unique((np.linspace(1, nw**(1 / power), nf)**power).astype(int))
#     ani = FuncAnimation(fig, update, frames=frames, 
#                         init_func=init, blit=True, interval=33.3)

from IPython.core.display import Video
ani.save('movie01.mp4', fps=30, extra_args=['-vcodec', 'libx264', '-crf', '26'])
Video("movie01.mp4")