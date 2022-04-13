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
from graph import Graph, Node
from helpers.bitset import BitSet
from helpers.helpers import min_max_normalization
from keras import mixed_precision
from typing import List, Tuple
# mixed_precision.set_global_policy('mixed_float16')
from sklearn.cluster import KMeans
from nptyping import NDArray, Float

NUM_SAMPLES = 949207

class DistExpander:
    def __init__(self, graph : Graph,
                    mu_1 : float = 0.1,
                    mu_2 : float = 0.1,
                    mu_3 : float = 0.1,
                    autoencoder : keras.Model = None):
        self.graph = graph
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.mu_3 = mu_3
        self.autoencoder = autoencoder

        assert graph.v > 0

def build_first_graph(
                        data : NDArray,
                        labels : NDArray,
                        percentage : float = 0.1,
                        autoencoder : keras.Model = None):
    """
        data is the original dataset 
        percentage is the percentage of this dataset that will be 
            sampled from data
    """
    graph = Graph()
    sample = data[np.random.choice(data.shape[0],
                                    int(data.shape[0]*percentage),
                                   replace=False)]
    assert autoencoder is not None
    embeddings = autoencoder.predict(sample)
    for i in range(sample.shape[0]):
        bs1 = BitSet(embeddings[i, 0])
        bs2 = BitSet(embeddings[i, 1])
        node = Node(key = [bs1, bs2], label = labels[i, :])
        graph.add_node(node)
    return graph

if __name__ == '__main__':
    data = genfromtxt("../data/data_Apr_01_20221.csv", delimiter=',',
                      invalid_raise=False)
    output_data = data[:, 11:14]
    output_data = output_data - \
                  np.array([0.00468998309224844,
                            -0.0009492466342635453,
                            0.12456292659044266])
    r = R.from_rotvec([0, 0, np.pi / 4.])
    y = r.apply(output_data)
    y = min_max_normalization(y, -1, 1)
    autoencoder = keras.models.load_model(
                            '../models/encoder_ae.hdf5',
                            compile=False
                  )
    graph = build_first_graph(
                              data = data[:, :11],
                              labels= y,
                              percentage = 0.01,
                              autoencoder = autoencoder)
    de = DistExpander(graph = graph)




