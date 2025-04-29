import torch
import numpy as np
from typing import Sequence



# start with the standard list of indices 
# Then, create an IID noise vector scaled by sigma * n (approaches n if sigma => 1, else closer to 0)
# Then, sorting base + noise will produce an almost random permutation 

def noise_sorted_perm(n: int, sigma: float, rng: np.random.RandomState) -> np.ndarray:
    """
    Returns a permutation of 0…n-1 whose disorder increases with `sigma ∈ [0,1]`.
    """
    assert 0.0 <= sigma <= 1.0
    base   = np.arange(n, dtype=np.float32)
    noise  = rng.randn(n).astype(np.float32) * sigma * n
    keys   = base + noise
    return np.argsort(keys)                 # permutation indices