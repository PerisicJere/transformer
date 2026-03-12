import numpy as np


class ScaledDotProductAttention:
    def __init__(self, d_k: int) -> None:
        self.scale: int = np.sqrt(d_k)

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        return np.matmul(self._softmax(np.matmul(Q, K.T) / self.scale), V)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        max_val = np.max(x, axis=-1, keepdims=True)
        exp_val = np.exp(x - max_val)
        return exp_val / np.sum(exp_val, axis=-1, keepdims=True)
