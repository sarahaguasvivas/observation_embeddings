import numpy as np

def min_max_normalization(x, new_min, new_max):
    current_min = np.min(x, axis=0)
    current_max = np.max(x, axis=0)
    normalized = (x - np.min(x, axis=0)) / (current_max - \
                                            current_min) * (new_max - new_min) + new_min
    normalized[np.isnan(normalized)] = 0
    return normalized
