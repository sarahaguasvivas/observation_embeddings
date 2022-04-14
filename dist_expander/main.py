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

def get_x_y(data: NDArray):
    output_data = data[:, 11:14]
    output_data = output_data - \
                  np.array([0.00468998309224844,
                            -0.0009492466342635453,
                            0.12456292659044266])
    r = R.from_rotvec([0, 0, np.pi / 4.])
    y = r.apply(output_data)
    y = min_max_normalization(y, -1, 1)
    return (data[:, :11], y)

if __name__ == '__main__':
    import seaborn as sns
    sns.set_theme()
    data = genfromtxt("../data/data_Apr_01_20221.csv", delimiter=',',
                      invalid_raise=False)
    x, y = get_x_y(data)
    autoencoder = keras.models.load_model(
        '../models/encoder_ae.hdf5',
        compile=False
    )
    graph, unlabeled_indices = build_first_graph(
        data=x,
        labels=y,
        percentage=0.0001,
        autoencoder=autoencoder)
    ax = sns.heatmap(graph.weights)
    plt.savefig('heat_map.png', dpi = 300)

    de = DistExpander(graph=graph,
                      mu_1 = 1,
                      mu_2 = 1,
                      mu_3 = 1,
                      partitions = 10,
                      max_iter = 10
                      )
    de.partition_graph()
    de.run()

