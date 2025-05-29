import numpy as np


def generate_sample(dist: str, alpha: float, size: int, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)

    generators = {
        "t3": lambda: rng.standard_t(df=alpha, size=size),
        "normal": lambda: rng.normal(loc=0, scale=alpha, size=size),
        "gamma": lambda: rng.gamma(shape=0.5, scale=alpha, size=size),
        "exp": lambda: rng.exponential(scale=alpha, size=size),
    }

    if dist not in generators:
        raise ValueError(f"Unknown distribution: {dist}")

    return generators[dist]()
