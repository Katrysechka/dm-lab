import numpy as np
from src.experiment_tools import monte_carlo_characteristic, build_threshold
from src.graph_utils import max_degree


def test_monte_carlo_characteristic_length():
    vals = monte_carlo_characteristic("t3", "knn", max_degree, 3, 50, 10)
    assert len(vals) == 10


def test_build_threshold_correctness():
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    thresh = build_threshold(values, alpha=0.1)
    assert thresh >= 8 and thresh <= 10
