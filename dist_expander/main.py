import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow import keras
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
import tensorflow as tf
import copy
from dist_expander import DistExpander, build_first_graph
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple
from numpy import genfromtxt
from graph import Graph, Node
from helpers.bitset import BitSet
from helpers.helpers import min_max_normalization
from keras import mixed_precision
from typing import List, Tuple
# mixed_precision.set_global_policy('mixed_float16')
from sklearn.cluster import KMeans
from nptyping import NDArray, Float
from multiprocessing import Pool

def get_x_y(data: NDArray):
    output_data = data[:, 11:14]
    output_data = output_data - \
                  np.array([0.00468998309224844,
                            -0.0009492466342635453,
                            0.12456292659044266])
    r = R.from_rotvec([0, 0, np.pi / 4.])
    y = r.apply(output_data)
    y = min_max_normalization(y, -1, 1)
    x = np.hstack((data[:, :11], data[:, 14:]))
    return (x, y)

if __name__ == '__main__':
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    import seaborn as sns
    sns.set_theme()
    data = genfromtxt("../data/data_Apr_01_20221.csv", delimiter=',',
                      invalid_raise=False)
    x, y = get_x_y(data)
    autoencoder = keras.models.load_model(
        '../embeddings/models/encoder_ae_2.hdf5',
        compile=False
    )
    task_nn = keras.models.load_model(
        '../embeddings/models/task_ae_2.hdf5',
        compile = False
    )
    graph, indices, lsh, chunks = build_first_graph(
        data=x,
        labels=y,
        percentage=0.00005,
        autoencoder=autoencoder,
        task = task_nn,
        partitions = 4)

    de = DistExpander(
                      graph=graph,
                      mu_1 = 5.,
                      mu_2 = 1e-4,
                      mu_3 = 1.,
                      partitions = 4,
                      task_nn = task_nn,
                      max_iter = 1,
                      mean_point = np.mean(y, axis = 0)
                      )
    de.lsh = lsh
    de.chunks = chunks
    for i in range(10):
        for p in range(de.partitions):
            de.run_iter(p)
        true_labels = y[indices]
        print(np.linalg.norm(true_labels - de.graph.y_hat))

    df_yhat = pd.DataFrame(de.graph.y_hat, columns = ['x', 'y', 'z'])
    df_ytrue = pd.DataFrame(true_labels, columns = ['x', 'y', 'z'])

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=true_labels[:, 0], y=true_labels[:, 1], z= true_labels[:, 2], name = 'true'))
    fig.add_trace(go.Scatter3d(x = de.graph.y_hat[:, 0], y = de.graph.y_hat[:, 1], z = de.graph.y_hat[:, 2], name = 'assigned'))
    fig.show()
    #fig.write_image("dist_expander_learned.svg", format='svg')

