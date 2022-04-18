import numpy as np
from numpy import genfromtxt
from helpers.bitset import BitSet
from typing import Dict, Set, Tuple, List
from nptyping import NDArray, Float
from sklearn.cluster import KMeans

class Node:
    def __init__(self, key : [BitSet, BitSet],
                    label : NDArray = None,
                   ):
        self.id = None
        self.key : [BitSet, BitSet] = key
        self.value : NDArray = label
        self.lsh_key = None
        self.neighbor_distrib: Dict[Node, List[float]] = {}
        self.neighbors = 0
        self.m_vl = 0.0

# This is an undirected graph
class Graph:
    def __init__(self, embedding_length : int = 32,
                 label_size : int = 3):
        self.v = 0
        self.vl = 0
        self.vu = 0
        self.n = embedding_length
        self.m = label_size
        self.edges : Set[Tuple[Node, Node]] = None
        self.edge_dict : Dict[Node, Set[Node]] = {}
        self.s = np.empty((0, 0)).reshape(0, 0)
        #self.weights = np.empty((0, 0)).reshape(0, 0)
        self.y = np.empty((0, self.m))
        self.y_hat = np.empty((0, self.m))
        self.upd = np.empty((0, self.m))
        self.node_dict : Dict[int, Node] = {}

    def add_node(self, node : Node):
        node.id = self.v
        self.node_dict[self.v] = node
        self.edge_dict[node] = []

        # making node declaration mandatory
        old_seed_matrix = self.s.copy()
        self.s = np.zeros((self.v + 1, self.v + 1))
        self.s[:self.v, :self.v] = old_seed_matrix

        #old_weight_matrix = self.weights.copy()
        #self.weights = np.zeros((self.v + 1, self.v + 1))
        #self.weights[:self.v, :self.v] = old_weight_matrix

        self.y = np.vstack((self.y, np.array([0]*self.m)))
        self.y_hat = np.vstack((self.y_hat, np.array([None]*self.m)))
        self.upd = np.vstack((self.upd, np.array([1/self.m]*self.m)))
        if self.v == 0:
            #self.weights = np.array([0])
            self.upd = np.array([1/self.m]*self.m)
        if node.value is not None:
            self.vl += 1
            self.s[-1, -1] = 1.
            self.y[-1, :] = node.value
            self.y_hat[-1, :] = node.value
        else:
            self.vu += 1
            self.s[-1, -1] = 0.
            self.y[-1, :] = 1. / self.m
            self.y_hat[-1, :] = 1. / self.m
        self.v += 1

    def add_edge(self, node1 : Node, node2 : Node):
        if node2 not in self.edge_dict[node1]:
            self.edge_dict[node1] += [node2]
        if node1 not in self.edge_dict[node2]:
            self.edge_dict[node2] += [node1]
        node2.neighbor_distrib[node1] = [0]*self.m
        node1.neighbor_distrib[node2] = [0]*self.m

if __name__ == '__main__':
    graph = Graph()
    node = Node(key = 3.2)
    graph.add_node(node)



