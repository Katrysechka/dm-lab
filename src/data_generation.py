import numpy as np


def generate_sample(dist: str, size: int, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    generators = {
        "gamma_h0": lambda: rng.gamma(shape=0.5, scale=1.0 / np.sqrt(0.5), size=size),
        "exp_h1": lambda: rng.exponential(scale=1.0, size=size),
    }

    if dist not in generators:
        raise ValueError(f"Unknown distribution: {dist}")

    return generators[dist]()
