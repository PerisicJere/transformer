import numpy as np

class PositionalEncoding:
    def __init__(self, vector: np.ndarray, d_model: int) -> None:
        self.vector = vector
        self.d_model = d_model

    def __call__(self) -> np.ndarray:
        encoded_vector: np.ndarray = self.vector.copy()
        for i, val  in enumerate(self.vector):
            position = i + 1
            for j in range(len(val)):
                # Sinusoidal
                if j % 2 == 0:
                    encoded_vector[i][j] = self.vector[i][j] + np.sin(position / np.power(10000, 2*(j//2) / self.d_model))
                # Cosine
                if j % 2 == 1:
                    encoded_vector[i][j] = self.vector[i][j] + np.cos(position / np.power(10000, 2*(j//2) / self.d_model))
        return encoded_vector
