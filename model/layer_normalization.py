import numpy as np


class LayerNormalization:
    def __init__(self, d_model: int, epsilon: np.float16 = 1e-6):
        self.epsilon = epsilon
        self.d_model = d_model
        self.beta = np.full(shape=d_model, fill_value=.5)
        self.gamma = np.full(shape=d_model, fill_value=1.5)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        mean: np.ndarray = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        normalized: np.ndarray = (x - mean) / (np.sqrt(variance + self.epsilon))

        layer_norm: list[float] = []
        for i in range(self.d_model):
            layer_norm.append(self.gamma[i] * normalized[i] + self.beta[i])

        layer_norm: np.ndarray = np.array(layer_norm, dtype=np.float32)
        return layer_norm
