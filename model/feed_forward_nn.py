import numpy as np

class FeedForwardNN:
    def __init__(self, input_size: int, output_size: int, hidden_layer: int):
        self.x = None
        self.z1 = None
        self.a1 = None
        self.w1 = np.random.randn(input_size, hidden_layer)
        self.b1 = np.random.randn(hidden_layer)
        self.w2 = np.random.randn(hidden_layer, output_size)
        self.b2 = np.random.randn(output_size)

    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z1 = self.linear_transformation(x=x,  weights=self.w1, bias=self.b1)
        self.a1 = self.relu(x=self.z1)
        return self.linear_transformation(x=self.a1, weights=self.w2, bias=self.b2)

    def backward_propagation(self, gradients: np.ndarray) -> None:
        # Once we implement loss we can do this

        da1 = gradients.dot(self.w2.T)
        dz1 = da1 * (self.z1 > 0)
        dw1 = self.x.T.dot(dz1)
        db1 = dz1.sum(axis=0)

        dw2 = self.a1.T.dot(gradients)
        db2 = gradients.sum(axis=0)

        self.w1 -= dw1 * 0.001
        self.b1 -= db1 * 0.001
        self.w2 -= dw2 * 0.001
        self.b2 -= db2 * 0.001

    @staticmethod
    def linear_transformation(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        return np.dot(x, weights) + bias

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)
