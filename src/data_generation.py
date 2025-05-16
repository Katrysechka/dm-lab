import numpy as np
from scipy.stats import t, norm

def generate_sample(dist: str, size: int, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    generators = {
        "t3": lambda: rng.standard_t(df=3, size=size),
        "normal": lambda: rng.normal(loc=0, scale=1, size=size),
    }

    if dist not in generators:
        raise ValueError(f"unknown dist: {dist}")
    
    return generators[dist]()
