import numpy as np
from sklearn.manifold import _t_sne
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances

MACHINE_EPSILON = np.finfo(np.double).eps

def min_max_normalization(x, new_min, new_max):
    current_min = np.min(x, axis=0)
    current_max = np.max(x, axis=0)
    normalized = (x - np.min(x, axis=0)) / (current_max - \
                                            current_min) * (new_max - new_min) + new_min
    normalized[np.isnan(normalized)] = 0
    return normalized


def t_sne_score(X_embedded, X, perplexity = 30., degrees_of_freedom = 3):
    print(X.shape)
    distances = pairwise_distances(X = X, metric = 'euclidean', squared=True, n_jobs = 4)
    n_samples = X_embedded.shape[0]
    P = _t_sne._joint_probabilities(distances, perplexity, False)
    # Q is a heavy-tailed distribution: Student's t-distribution
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)
    return 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))

def trustworthiness(X_embedded, X):
    return trustworthiness(X, X_embedded, n_neighbors=5, metric="euclidean")





