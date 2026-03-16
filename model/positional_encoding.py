import numpy as np

class PositionalEncoding:
    def __init__(self, d_model: int) -> None:
        self.d_model = d_model

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        encoded_vector: np.ndarray = embeddings.copy()
        for i, val  in enumerate(embeddings):
            position = i + 1
            for j in range(len(val)):
                # Sinusoidal
                if j % 2 == 0:
                    encoded_vector[i][j] = embeddings[i][j] + np.sin(position / np.power(10000, 2*(j//2) / self.d_model))
                # Cosine
                if j % 2 == 1:
                    encoded_vector[i][j] = embeddings[i][j] + np.cos(position / np.power(10000, 2*(j//2) / self.d_model))
        return encoded_vector
