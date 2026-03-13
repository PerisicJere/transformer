import numpy as np

class Linear:
    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.linear: np.ndarray = np.random.rand(in_dim, out_dim)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.linear)
