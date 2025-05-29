import sys

sys.path.append("..")
import numpy as np


def build_knn_graph(data: np.ndarray, k: int) -> dict:
    n = len(data)
    G = {i: set() for i in range(n)}
    dist_matrix = np.abs(data.reshape(-1, 1) - data.reshape(1, -1))

    for i in range(n):
        neighbors = np.argsort(dist_matrix[i])[1 : k + 1]  # k ближайших
        for j in neighbors:  # неориентированный граф
            G[i].add(j)
            G[j].add(i)
    return G


def build_distance_graph(data: np.ndarray, d: float) -> dict:
    n = len(data)
    G = {i: set() for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if abs(data[i] - data[j]) <= d:
                G[i].add(j)
                G[j].add(i)
    return G


def chromatic_number_interval_graph(data: np.ndarray, d: float) -> int:
    intervals = [(x - d, x + d) for x in data]
    endpoints = []
    for idx, (start, end) in enumerate(intervals):
        endpoints.append((start, "start", idx))
        endpoints.append((end, "end", idx))
    endpoints.sort()

    active = 0
    max_active = 0
    for point, kind, _ in endpoints:
        if kind == "start":
            active += 1
            max_active = max(max_active, active)
        else:
            active -= 1
    return max_active


def num_connected_components(G: dict) -> int:
    visited = set()
    count = 0

    def dfs(v):
        stack = [v]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(G[node] - visited)

    for v in G:
        if v not in visited:
            dfs(v)
            count += 1

    return count


def max_degree(G: dict) -> int:  # Максимальная степень вершины графа
    return max(len(neigh) for neigh in G.values())


def greedy_max_independent_set(G: dict) -> set:
    # максимальное независимое множество
    G_copy = {node: set(neigh) for node, neigh in G.items()}
    independent_set = set()

    while G_copy:
        # выбираем вершину с мин степенью
        node = min(G_copy, key=lambda n: len(G_copy[n]))
        independent_set.add(node)  # добавляем в множество
        to_remove = G_copy[node] | {node}

        for rem in to_remove:  # удаляем соседей
            for neigh in G_copy.get(rem, []):
                G_copy[neigh].discard(rem)
            G_copy.pop(rem, None)

    return independent_set


def max_independent_set_size(G: dict) -> int:
    # размер макс независимого множества
    mis = greedy_max_independent_set(G)
    return len(mis)  # возвращаем размер множества


#
