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
from nptyping import NDArray
import random
import multiprocessing

NUM_SAMPLES = 949207

# Number of bits in which the neural network trained the
# embedding. In our case, we forced Tensorflow to cast this
# as a 16-bit encoding
BIT_ENCODING = 16

SIMILARITY_SCORE_THRESHOLD = 1

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
    a = []
    b = []
    for i in range(latent_dim):
        a += [node0.key[i].floating_point]
        b += [node1.key[i].floating_point]
    return np.linalg.norm(np.array(a) - np.array(b), 2)

class DistExpander:
    def __init__(self, graph : Graph,
                    mu_1 : float = 0.1,
                    mu_2 : float = 0.1,
                    mu_3 : float = 0.1,
                    autoencoder : keras.Model = None,
                    partitions : int = 10,
                    max_iter : int = 10):
        self.graph = graph
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.mu_3 = mu_3
        self.autoencoder = autoencoder
        self.partitions = partitions
        self.chunks = []
        self.max_iter = max_iter
        assert graph.v > 0

    def partition_graph(self):
        list_indices = np.arange(self.graph.v)
        random.shuffle(list_indices)
        self.chunks = [list_indices[i::self.partitions] for i in range(self.partitions)]

    def run(self):
        for iter in range(self.max_iter):
            for i in range(self.partitions):
                for node_idx in self.chunks[i].tolist():
                    # broadcast previous label distribution to all neigh
                    node_i = self.graph.node_dict[node_idx]
                    node_i.m_vl= self.mu_1 * self.graph.s[node_idx, node_idx] + self.mu_3
                    for neigh in self.graph.edge_dict[node_i]:
                        # Broadcast:
                        neigh.neighbor_distrib[node_i] = self.graph.y_hat[node_idx]
                        # Receive message:
                        node_i.m_vl += self.mu_2 / self.graph.weights[node_idx, neigh.id] # would be multiplied

                for node_idx in self.chunks[i].tolist():
                    # receive mu from neighbors u with corresponding
                    # message weights, process each message
                    node_i = self.graph.node_dict[node_idx]
                    for l in range(self.graph.m):
                        self.graph.y_hat[node_idx, l] = 1./node_i.m_vl * \
                                         (self.mu_1 * self.graph.s[node_idx, node_idx] * self.graph.y[node_idx, l]) + \
                                         self.mu_3 / self.graph.m
                        for neigh in self.graph.edge_dict[node_i]:
                            self.graph.y_hat[node_idx, l] += self.mu_2 * node_i.neighbor_distrib[neigh][l]

def build_first_graph(
                        data : NDArray,
                        labels : NDArray,
                        percentage : float = 0.1,
                        autoencoder : keras.Model = None
    ):
    """
        data is the original dataset 
        percentage is the percentage of this dataset that will be 
            sampled from data
    """
    graph = Graph()
    indices = np.random.choice(data.shape[0],
                        int(data.shape[0]*(2*percentage)),
                               replace= False)
    unlabeled_indices = indices[indices.shape[0]//2:]
    sample = data[indices]
    sample_labeled = sample[:sample.shape[0]//2, :]
    sample_unlabeled = sample[sample.shape[0]//2:, :]
    sim_score_matrix = np.zeros((len(indices), len(indices)))
    assert autoencoder is not None
    embeddings = autoencoder.predict(sample_labeled)
    for i in range(sample_labeled.shape[0]):
        bs1 = BitSet(embeddings[i, 0])
        bs2 = BitSet(embeddings[i, 1])
        node = Node(key=[bs1, bs2], label=labels[i, :])
        graph.add_node(node)
        for j in range(graph.v - 1):
            sim_score = similarity_score(node, graph.node_dict[j])
            graph.weights[i, j] = sim_score
            graph.weights[j, i] = sim_score
            if sim_score < SIMILARITY_SCORE_THRESHOLD:
                graph.add_edge(node, graph.node_dict[j])

    embeddings_unlabeled = autoencoder.predict(sample_unlabeled)
    for i in range(sample_unlabeled.shape[0]):
        bs1 = BitSet(embeddings_unlabeled[i, 0])
        bs2 = BitSet(embeddings_unlabeled[i, 1])
        node = Node(key = [bs1, bs2], label = None)
        graph.add_node(node)
        for j in range(graph.v - 1):
            sim_score = similarity_score(node, graph.node_dict[j])
            graph.weights[i + sample_labeled.shape[0], j] = sim_score
            graph.weights[j, i + sample_labeled.shape[0]] = sim_score
            if sim_score < SIMILARITY_SCORE_THRESHOLD:
                graph.add_edge(node, graph.node_dict[j])
    return graph, indices

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
    graph, indices = build_first_graph(
                              data = data[:, :11],
                              labels= y,
                              percentage = 0.001,
                              autoencoder = autoencoder)
    true_labels = y[indices]

    dist_e = DistExpander(graph = graph)
    dist_e.partition_graph()

    print(((np.de.graph.y_hat - true_labels)**2).mean(axis = 0))







