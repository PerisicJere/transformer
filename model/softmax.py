import numpy as np


def softmax(input: np.ndarray) -> np.ndarray:
    max_val = np.max(input, axis=-1, keepdims=True)
    exp_val = np.exp(input - max_val)
    return exp_val / np.sum(exp_val, axis=-1, keepdims=True)
