import cupy as np

class Linear:
    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.x = None
        self.linear: np.ndarray = (np.random.rand(in_dim, out_dim) * np.sqrt(2/(in_dim+out_dim))).astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.matmul(x, self.linear)

    def backward(self, gradients: np.ndarray, learning_rate: np.float32) -> np.ndarray:
        x_2d = self.x.reshape(-1, self.x.shape[-1])
        grad_2d = gradients.reshape(-1, gradients.shape[-1])

        dw1 = x_2d.T @ grad_2d
        dx = gradients @ self.linear.T

        self.linear -= np.clip(dw1, -5, 5) * learning_rate
        return np.clip(dx, -5, 5)
