"""
Модуль hypothesis_generators: генераторы выборок для H0 (Gamma) и H1 (Exponential).
"""

import numpy as np


def generate_sample(dist: str, size: int, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    generators = {
        "gamma": lambda: rng.gamma(shape=0.5, scale=1.0 / np.sqrt(0.5), size=size),
        "exp": lambda: rng.exponential(scale=1.0, size=size),
    }

    if dist not in generators:
        raise ValueError(f"Unknown distribution: {dist}")

    return generators[dist]()


def sample_h0(n: int, seed: int = None) -> np.ndarray:
    """
    Генерация выборки из Gamma(1/2, √2) с фиксированным seed.
    """
    return generate_sample("gamma", size=n, seed=seed)


def sample_h1(n: int, seed: int = None) -> np.ndarray:
    """
    Генерация выборки из Exp(1) с фиксированным seed.
    """
    return generate_sample("exp", size=n, seed=seed)
