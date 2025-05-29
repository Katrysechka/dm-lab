"""
Модуль simulator: Монте-Карло симуляции графовых характеристик для Gamma и Exp распределений.
"""

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

def run_monte_carlo(
    sample_size: int, num_runs: int, graph_kind: str, graph_param: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Проводит num_runs экспериментов с генерацией выборок из H0 и H1,
    построением графа указанного типа и вычислением статистик.

    Параметры:
    - sample_size: количество точек в каждой выборке
    - num_runs: сколько раз повторять эксперимент
    - graph_kind: "knn" или "dist"
    - graph_param: параметр построения графа (k или d)

    Возвращает:
    - два DataFrame (H0 и H1), где строки — отдельные симуляции
    """
    results_h0 = []
    results_h1 = []

    for i in range(num_runs):
        data_h0 = generate_sample("gamma", size=sample_size, seed=i)
        data_h1 = generate_sample("exp", size=sample_size, seed=i)

        if graph_kind == "knn":
            g0 = build_knn_graph(data_h0, int(graph_param))
            g1 = build_knn_graph(data_h1, int(graph_param))

            res0 = {
                "components": num_connected_components(g0),
                "max_deg": max_degree(g0),
            }
            res1 = {
                "components": num_connected_components(g1),
                "max_deg": max_degree(g1),
            }

        elif graph_kind == "dist":
            g0 = build_distance_graph(data_h0, graph_param)
            g1 = build_distance_graph(data_h1, graph_param)

            res0 = {
                "components": num_connected_components(g0),
                "chrom_num": chromatic_number_interval_graph(data_h0, graph_param),
            }
            res1 = {
                "components": num_connected_components(g1),
                "chrom_num": chromatic_number_interval_graph(data_h1, graph_param),
            }

        else:
            raise RuntimeError(f"Тип графа '{graph_kind}' не поддерживается")

        results_h0.append(res0)
        results_h1.append(res1)

    return pd.DataFrame(results_h0), pd.DataFrame(results_h1)