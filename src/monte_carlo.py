import numpy as np
import pandas as pd

from .graph_utils import (
    build_knn_graph,
    build_distance_graph,
    max_degree,
    chromatic_number_interval_graph,
    num_connected_components,
)
from .data_generation import generate_sample


def simulate_statistics(
    n: int, trials: int, graph_type: str, param: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Для trials симуляций генерируем два набора выборок:
    H0 ~ Gamma(1/2, λ0=√0.5) и H1 ~ Exp(λ0=1), строим граф указанного типа,
    вычисляем характеристики (число компонент и хроматическое число/макс степень)
    и возвращаем два DataFrame: по одному для H0 и H1 (trials × 2).
    """
    stats0 = []
    stats1 = []
    for t in range(trials):
        sample0 = generate_sample("gamma", size=n, seed=seed)
        sample1 = generate_sample("exp", size=n, seed=seed)

        if graph_type == "knn":
            graph0 = build_knn_graph(sample0, int(param))
            graph1 = build_knn_graph(sample1, int(param))
            row0 = {
                "num_components": num_connected_components(graph0),
                "max_degree": max_degree(graph0),
            }
            row1 = {
                "num_components": num_connected_components(graph1),
                "max_degree": max_degree(graph1),
            }

        elif graph_type == "dist":
            graph0 = build_distance_graph(sample0, param)
            graph1 = build_distance_graph(sample1, param)
            row0 = {
                "num_components": num_connected_components(graph0),
                "chromatic_number": chromatic_number_interval_graph(sample0, param),
            }
            row1 = {
                "num_components": num_connected_components(graph1),
                "chromatic_number": chromatic_number_interval_graph(sample1, param),
            }

        else:
            raise ValueError(f"Unknown graph_type={graph_type!r}")

        stats0.append(row0)
        stats1.append(row1)

    return pd.DataFrame(stats0), pd.DataFrame(stats1)
