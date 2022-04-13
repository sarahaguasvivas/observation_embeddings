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
from keras import mixed_precision
from typing import List, Tuple
# mixed_precision.set_global_policy('mixed_float16')
from sklearn.cluster import KMeans
from nptyping import NDArray, Float

NUM_SAMPLES = 949207
data = genfromtxt("../data/data_Apr_01_20221.csv", delimiter=',',
                  invalid_raise=False)

class DistExpander:
    def __init__(self, graph : Graph,
                    mu_1 : float = 0.1,
                    mu_2 : float = 0.1,
                    mu_3 : float = 0.1):
        self.graph = graph
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.mu_3 = mu_3

        assert graph.v > 0

if __name__ == '__main__':
    graph = Graph()
    node = Node(2)
    graph.add_node(node)
    de = DistExpander(graph = graph)




