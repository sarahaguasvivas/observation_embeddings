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

# Number of bits in which the neural network trained the
# embedding. In our case, we forced Tensorflow to cast this
# as a 16-bit encoding
BIT_ENCODING = 16

def compute_hamming_weight(x : BitSet):
    """
        Hamming weight of a binary number
        This is a simulated routine to get hamming weight
        The proper algorithm for embedded is a lot more efficient
    :param x:
    :return:
    """
    print(x.bitset_bin)
    n = str(bin(x.bitset_ser))
    print(len(n))
    weight = 0
    for i in range(len(n)):
        if n[i] == 1:
            weight += 1
    return weight


def similarity_score(node0 : Node, node1: Node,
                     latent_dim : int = 2)->float:
    """
        Because we're working on bit-land, we use the edit
        distance between the bitvecs from the two embeddings

        Hamming distance

    :param node0:
    :param node1:
    :return:
    """
    assert len(node0.key) == latent_dim
    assert len(node1.key) == latent_dim
    sim_score = [BitSet(0.0)]*latent_dim
    distance = 0
    for i in range(latent_dim):
        sim_score[i].bitset_ser = node0.key[i].bitset_ser ^ node1.key[i].bitset_ser
        sim_score[i].update()
        distance += compute_hamming_weight(sim_score[i])
    return distance / (BIT_ENCODING * latent_dim)

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
        node = Node(key=[bs1, bs2], label=labels[i, :])
        graph.add_node(node)
        for j in range(graph.v - 1):
            sim_score = similarity_score(node, graph.node_dict[j])
            print(sim_score)
            if sim_score < 0.75:
                graph.add_edge(node, graph.node_dict[j])
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
                              percentage = 0.001,
                              autoencoder = autoencoder)
    de = DistExpander(graph = graph)




