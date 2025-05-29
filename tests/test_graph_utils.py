import numpy as np
from src.graph_utils import (
    build_knn_graph,
    build_distance_graph,
    max_degree,
    max_independent_set_size,
)
import sys

sys.path.append("..")


def test_build_knn_graph_structure():
    data = np.array([1, 2, 3, 4, 5])
    G = build_knn_graph(data, k=2)
    assert len(G) == 5
    # Каждая вершина должна иметь не более k соседей
    assert all(len(neigh) >= 2 for neigh in G.values())


def test_build_distance_graph_structure():
    data = np.array([0.1, 0.2, 1.0, 1.1])
    G = build_distance_graph(data, d=0.15)
    assert len(G) == 4
    # Проверяем, что рёбра есть только там, где расстояние ≤ d
    for i in G:
        for j in G[i]:
            assert abs(data[i] - data[j]) <= 0.15


def test_max_degree():
    data = np.array([1, 2, 3, 4, 5])
    G = build_knn_graph(data, k=2)
    deg = max_degree(G)
    assert isinstance(deg, int)
    assert deg >= 2


def test_max_independent_set_size():
    data = np.array([1, 2, 3, 4, 5])
    G = build_knn_graph(data, k=1)
    mis_size = max_independent_set_size(G)
    assert isinstance(mis_size, int)
    assert 1 <= mis_size <= len(G)
