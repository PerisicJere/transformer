import numpy as np

class FeedForwardNN:
    def __init__(self, input_size: int, output_size: int, hidden_layer: int):
        self.x, self.z1, self.a1 = None, None, None
        self.w1 = np.random.randn(input_size, hidden_layer) * np.sqrt(2 / input_size)
        self.b1 = np.random.randn(hidden_layer) * np.sqrt(2 / hidden_layer)
        self.w2 = np.random.randn(hidden_layer, output_size) * np.sqrt(2 / hidden_layer)
        self.b2 = np.random.randn(output_size) * np.sqrt(2 / output_size)

    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z1 = self.linear_transformation(x=x,  weights=self.w1, bias=self.b1)
        self.a1 = self.relu(x=self.z1)
        return self.linear_transformation(x=self.a1, weights=self.w2, bias=self.b2)

    def backward_propagation(self, gradients: np.ndarray) -> np.ndarray:
        da1 = gradients.dot(self.w2.T)
        dz1 = da1 * (self.z1 > 0)
        dw1 = self.x.T.dot(dz1)
        db1 = dz1.sum(axis=0)

        dw2 = self.a1.T.dot(gradients)
        db2 = gradients.sum(axis=0)

        dx = dz1.dot(self.w1.T)

        self.w1 -= np.clip(dw1, -1, 1) * 0.001
        self.b1 -= np.clip(db1, -1, 1) * 0.001
        self.w2 -= np.clip(dw2, -1, 1) * 0.001
        self.b2 -= np.clip(db2, -1, 1) * 0.001

        return np.clip(dx, -1, 1)

    @staticmethod
    def linear_transformation(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        return np.dot(x, weights) + bias

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)
