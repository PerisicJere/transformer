import numpy as np

class FeedForwardNN:
    def __init__(self, input_size: int, output_size: int, hidden_layer: int):
        self.z1 = None
        self.a1 = None
        self.w1 = np.random.randn(input_size, hidden_layer)
        self.b1 = np.random.randn(hidden_layer)
        self.w2 = np.random.randn(hidden_layer, output_size)
        self.b2 = np.random.randn(output_size)

    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        self.z1 = self.linear_transformation(x=x,  weights=self.w1, bias=self.b1)
        self.a1 = self.relu(x=self.z1)
        return self.linear_transformation(x=self.a1, weights=self.w2, bias=self.b2)

    def backward_propagation(self, x, y):
        # Once we implement loss we can do this
        pass

    @staticmethod
    def linear_transformation(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        return np.dot(x, weights) + bias

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)
