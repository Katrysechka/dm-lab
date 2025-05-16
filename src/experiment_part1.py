from dataclasses import dataclass
import numpy as np
from src.graph_utils import max_degree, max_independent_set_size
from src.experiment_tools import monte_carlo_characteristic, build_threshold


@dataclass
class ExperimentResult:
    dist: str
    graph_type: str
    characteristic: str
    threshold: float
    power: float


def run_experiment(n=100, k=4, d=0.5, alpha=0.055, n_iter=500):
    results = []
    characteristics = [max_degree, max_independent_set_size]
    char_names = ["max_degree", "max_independent_set_size"]
    graph_types = [("knn", k), ("distance", d)]

    for graph_type, param in graph_types:
        for char_func, char_name in zip(characteristics, char_names):
            T_H0 = monte_carlo_characteristic(
                "t3", graph_type, char_func, param, n, n_iter
            )
            T_H1 = monte_carlo_characteristic(
                "normal", graph_type, char_func, param, n, n_iter
            )
            threshold = build_threshold(T_H0, alpha)
            power = np.mean(T_H1 > threshold)
            results.append(
                ExperimentResult(
                    "t3 vs normal", graph_type, char_name, threshold, power
                )
            )

    for r in results:
        print(
            f"{r.graph_type.upper()} | {r.characteristic}: "
            f"threshold = {r.threshold:.3f}, power = {r.power:.3f}"
        )

    return results


if __name__ == "__main__":
    run_experiment()


#
