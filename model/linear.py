import numpy as np

class Linear:
    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.x = None
        self.linear: np.ndarray = np.random.rand(in_dim, out_dim) / np.sqrt(2/(in_dim+out_dim))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.matmul(x, self.linear)

    def backward(self, gradients: np.ndarray) -> np.ndarray:

        dw1 = self.x.T.dot(gradients)
        dx = gradients.dot(dw1.T)

        self.linear -= np.clip(dw1, -1, 1) * 0.001

        return np.clip(dx, -1, 1)
