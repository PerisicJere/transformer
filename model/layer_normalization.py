import numpy as np


class LayerNormalization:
    def __init__(self, d_model: int, epsilon: np.float16 = 1e-6):
        self.x, self.normalized, self.mean, self.variance = None, None, None, None
        self.epsilon = epsilon
        self.d_model = d_model
        self.beta = np.full(shape=d_model, fill_value=.5)
        self.gamma = np.full(shape=d_model, fill_value=1.5)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.mean: np.ndarray = np.mean(x, axis=-1, keepdims=True)
        self.variance = np.var(x, axis=-1, keepdims=True)

        self.normalized: np.ndarray = (x - self.mean) / (np.sqrt(self.variance + self.epsilon))

        return self.gamma * self.normalized + self.beta

    def backward(self, gradients: np.ndarray) -> np.ndarray:

        d_beta = gradients.sum(axis=0)
        d_gamma = (gradients * self.normalized).sum(axis=0)

        dx = gradients * self.gamma

        d_variance = (dx * (self.x - self.mean) * (-.5) * (self.variance + self.epsilon)**(-(3/2))).sum(axis=-1, keepdims=True)

        d_mean = -(dx.sum(axis=-1, keepdims=True) / np.sqrt(self.variance + self.epsilon))

        dx = dx / np.sqrt(self.variance + self.epsilon) + d_variance * 2*(self.x - self.mean) / self.d_model + d_mean / self.d_model

        self.beta -= d_beta * 0.001
        self.gamma -= d_gamma * 0.001

        return dx




