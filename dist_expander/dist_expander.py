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
from nptyping import NDArray
from sklearn.cluster import KMeans
import random
import multiprocessing

NUM_SAMPLES = 949207
# Number of bits in which the neural network trained the
# embedding. In our case, we forced Tensorflow to cast this
# as a 16-bit encoding
BIT_ENCODING = 16

SIMILARITY_SCORE_THRESHOLD = 50

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
    a = node0.get_embedding()
    b = node1.get_embedding()
    if np.linalg.norm(a - b) < 1e-30:
        return 1
    else:
        return 1. / np.linalg.norm(a - b)
    #return np.dot(a, b)

class DistExpander:
    def __init__(self, graph : Graph,
                    mu_1 : float = 0.1,
                    mu_2 : float = 0.1,
                    mu_3 : float = 0.1,
                    autoencoder : keras.Model = None,
                    task_nn : keras.Model = None,
                    partitions : int = 10,
                    mean_point = None,
                    max_iter : int = 10):
        self.graph = graph
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.mu_3 = mu_3
        self.autoencoder = autoencoder
        self.task = task_nn
        self.partitions = partitions
        self.mean_point = mean_point
        self.chunks : List[List[int]] = []
        self.max_iter = max_iter
        self.lsh = KMeans(n_clusters= partitions, random_state = 0)
        assert graph.v > 0, "graph cannot be empty"

    def run_iter(self, partition):
        for node_idx in self.chunks[partition]:
            # broadcast previous label distribution to all neigh
            node_i = self.graph.node_dict[node_idx]
            node_i.m_vl = self.mu_1 * self.graph.s[node_idx, node_idx] + self.mu_3
            for neigh in self.graph.edge_dict[node_i]:
                neigh.neighbor_distrib[node_i] = self.graph.y_hat[node_idx, :]
                weight = similarity_score(node_i, neigh)
                node_i.m_vl += self.mu_2 * weight

        for node_idx in self.chunks[partition]:
            # receive mu from neighbors u with corresponding
            # message weights, process each message
            node_i = self.graph.node_dict[node_idx]
            if self.graph.s[node_idx, node_idx] == 0:
                self.graph.y_hat[node_idx, :] = self.mu_1 * self.graph.s[node_idx, node_idx] * \
                                                    self.graph.y[node_idx, :] #+ self.mu_3 * self.graph.task_outputs[node_idx, :]
                for neigh, dist in node_i.neighbor_distrib.items():
                    weight = similarity_score(node_i, neigh)
                    self.graph.y_hat[node_idx, :] += self.mu_2 * np.array(dist) * weight
                self.graph.y_hat[node_idx, :] /= node_i.m_vl

def build_first_graph(
                        data : NDArray,
                        labels : NDArray,
                        percentage : float = 0.1,
                        autoencoder : keras.Model = None,
                        task : keras.Model = None,
                        partitions : int = 10,
                        labeled_to_unlabeled = 0.9
    ):
    """
        data is the original dataset 
        percentage is the percentage of this dataset that will be 
            sampled from data
    """
    graph = Graph()
    lsh = KMeans(n_clusters = partitions, random_state = 0)
    indices = np.random.choice(data.shape[0],
                        int(data.shape[0]*(2*percentage)),
                               replace= False)
    sample = data[indices]
    sampled_labels = labels[indices]
    sample_labeled = sample[:int(len(indices)*labeled_to_unlabeled), :]
    sample_unlabeled = sample[int(len(indices)*labeled_to_unlabeled):, :]
    assert autoencoder is not None, "no autoencoder found"

    embeddings = autoencoder.predict(sample_labeled)
    graph.task_outputs = task.predict(autoencoder.predict(sample))
    for i in range(sample_labeled.shape[0]):
        bs1 = BitSet(embeddings[i, 0])
        bs2 = BitSet(embeddings[i, 1])
        node = Node(key=[bs1, bs2], label=sampled_labels[i, :])
        graph.add_node(node)

    embeddings_unlabeled = autoencoder.predict(sample_unlabeled)
    for i in range(sample_unlabeled.shape[0]):
        bs1 = BitSet(embeddings_unlabeled[i, 0])
        bs2 = BitSet(embeddings_unlabeled[i, 1])
        node = Node(key = [bs1, bs2], label = None)
        graph.add_node(node)

    lsh.fit(np.vstack((embeddings, embeddings_unlabeled)))
    chunks = {}
    for i in range(partitions):
        chunks[i] = []
    for enum, lab in enumerate(lsh.labels_):
        chunks[lab] += [enum]
        for i in range(len(chunks[lab]) - 1):
            if similarity_score(graph.node_dict[enum], graph.node_dict[chunks[lab][i]]) > 0.2:
                graph.add_edge(graph.node_dict[enum], graph.node_dict[chunks[lab][i]])
    graph.embeddings = np.vstack((embeddings, embeddings_unlabeled))
    return graph, indices, lsh, chunks, int(len(indices)*labeled_to_unlabeled)

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
                            '../models/encoder_ae_2.hdf5',
                            compile=False
                  )
    graph, indices, _, _ = build_first_graph(
                              data = np.hstack((data[:, :11], min_max_normalization([:, 14:], -0.5, 0.5))),
                              labels= y,
                              percentage = 0.001,
                              autoencoder = autoencoder)
    true_labels = y[indices]
    dist_e = DistExpander(graph = graph)
    dist_e.partition_graph()



