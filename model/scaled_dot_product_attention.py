import numpy as np


class ScaledDotProductAttention:
    def __init__(self, d_k: int, num_heads: int, mask: bool) -> None:
        self.Q, self.K, self.V, self.attention_weights = None, None, None, None
        self.scale: int = np.sqrt(d_k / num_heads)
        self.mask = mask

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, pad_mask: np.ndarray | None) -> np.ndarray:
        self.Q, self.K, self.V = Q, K, V
        raw_attention_scores: np.ndarray = np.matmul(Q, K.T) / self.scale

        if self.mask:
            masked: np.ndarray = self.__get_mask(raw_attention_scores.shape)
            raw_attention_scores = (raw_attention_scores + masked)

        if pad_mask is not None:
            raw_attention_scores = raw_attention_scores + pad_mask

        self.attention_weights = self._softmax(raw_attention_scores)
        return np.matmul(self.attention_weights, V)

    def backward(self, gradients: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dV = self.attention_weights.T.dot(gradients)
        dS = self._softmax_backward(gradients)

        dQ = np.matmul(dS, self.K) / self.scale
        dK = np.matmul(dS.T, self.Q) / self.scale

        return np.clip(dQ, -5, 5), np.clip(dK, -5, 5), np.clip(dV, -5, 5)

    def _softmax_backward(self, gradients: np.ndarray) -> np.ndarray:
        dP: np.ndarray = gradients.dot(self.V.T)
        O: np.ndarray = np.matmul(self.attention_weights, self.V)
        row_sum: np.ndarray = (gradients * O).sum(axis=1, keepdims=True)

        dS: np.ndarray = (dP * self.attention_weights) - (row_sum * self.attention_weights)

        return dS

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        max_val = np.max(x, axis=-1, keepdims=True)
        exp_val = np.exp(x - max_val)
        return exp_val / np.sum(exp_val, axis=-1, keepdims=True)

    def __get_mask(self, shape: tuple) -> np.ndarray:
        lower_triangular: np.ndarray = np.tril(np.ones(shape), k=0)
        upper_inf: np.ndarray = np.where(lower_triangular == 0, -np.inf, lower_triangular)
        final_mask: np.ndarray = np.where(upper_inf == 1, 0.0, upper_inf)
        return final_mask
