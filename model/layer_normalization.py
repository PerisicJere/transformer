import numpy as np


class LayerNormalization:
    def __init__(self, d_model: int, epsilon: np.float32 = 1e-6):
        self.x, self.normalized, self.mean, self.variance = None, None, None, None
        self.epsilon = epsilon
        self.d_model = d_model
        self.beta = np.zeros(d_model)
        self.gamma = np.ones(d_model)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.mean: np.ndarray = np.mean(x, axis=-1, keepdims=True)
        self.variance = np.var(x, axis=-1, keepdims=True)

        self.normalized: np.ndarray = (x - self.mean) / (np.sqrt(self.variance + self.epsilon))

        return self.gamma * self.normalized + self.beta

    def backward(self, gradients: np.ndarray, learning_rate: np.float32) -> np.ndarray:

        d_beta = gradients.sum(axis=0)
        d_gamma = (gradients * self.normalized).sum(axis=0)

        dx = gradients * self.gamma

        d_variance = (dx * (self.x - self.mean) * (-.5) * (self.variance + self.epsilon)**(-(3/2))).sum(axis=-1, keepdims=True)

        d_mean = -(dx.sum(axis=-1, keepdims=True) / np.sqrt(self.variance + self.epsilon))

        dx = dx / np.sqrt(self.variance + self.epsilon) + d_variance * 2*(self.x - self.mean) / self.d_model + d_mean / self.d_model

        self.beta -= np.clip(d_beta, -5, 5) * learning_rate
        self.gamma -= np.clip(d_gamma, -5, 5) * learning_rate

        return np.clip(dx, -5, 5)




