import numpy as np


class ScaledDotProductAttention:
    def __init__(self, d_k: int, num_heads: int, mask: bool) -> None:
        self.scale: int = np.sqrt(d_k / num_heads)
        self.mask = mask

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        raw_attention_scores: np.ndarray = np.matmul(Q, K.T)
        if self.mask:
            masked: np.ndarray = self.__get_mask(raw_attention_scores.shape)
            return np.matmul(self._softmax((raw_attention_scores + masked) / self.scale), V)
        return np.matmul(self._softmax(raw_attention_scores / self.scale), V)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        max_val = np.max(x, axis=-1, keepdims=True)
        exp_val = np.exp(x - max_val)
        return exp_val / np.sum(exp_val, axis=-1, keepdims=True)

    def __get_mask(self, shape: tuple) -> np.ndarray:
        lower_triangular: np.ndarray = np.tril(np.ones(shape), k=0)
        upper_inf: np.ndarray = np.where(lower_triangular == 0, -np.inf, lower_triangular)
        final_mask: np.ndarray = np.where(upper_inf == 1, 0.0, upper_inf)
        return final_mask
