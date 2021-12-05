# external dependencies
import numpy as np

def sum_(A: np.ndarray, *args) -> np.ndarray:
    assert A.ndim == 3
    return A.sum(axis=0)

def mean_(A: np.ndarray, *args) -> np.ndarray:
    assert A.ndim == 3
    return A.mean(axis=0)

def max_(A: np.ndarray, *args) -> np.ndarray:
    assert A.ndim == 3
    return A.max(axis=0, *args)

def min_(A: np.ndarray, *args) -> np.ndarray:
    assert A.ndim == 3
    return A.min(axis=0)

def median_(A: np.ndarray, *args) -> np.ndarray:
    assert A.ndim == 3
    return A.median(axis=0)

def scalar_power_mean(A: np.ndarray, p: float) -> np.ndarray:
    assert A.ndim == 3
    power_row_sums = (A**p).sum(axis=0)
    means = power_row_sums / A.shape[0]
    return means ** (1/p)