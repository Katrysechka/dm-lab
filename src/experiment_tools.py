from src.data_generation import generate_sample
from src.graph_utils import build_knn_graph, build_distance_graph
import numpy as np


def monte_carlo_characteristic(
    dist, graph_type, char_func, graph_param, n_samples, n_iter
):
    values = []
    for _ in range(n_iter):
        sample = generate_sample(dist, n_samples)
        if graph_type == "knn":
            G = build_knn_graph(sample, graph_param)
        elif graph_type == "distance":
            G = build_distance_graph(sample, graph_param)
        else:
            raise ValueError("error")
        values.append(char_func(G))
    return np.array(values)


def build_threshold(values, alpha=0.055):
    #  пороговое значение для критерия на уровне значимости alpha
    sorted_vals = sorted(values)
    idx = int((1 - alpha) * len(sorted_vals))
    return sorted_vals[idx]


#
