import cupy as np

class PositionalEncoding:
    def __init__(self, d_model: int) -> None:
        self.d_model = d_model
        self._positional_encoding_cache = {}

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        batch, seq_len, _ = embeddings.shape
        if seq_len not in self._positional_encoding_cache:
            positions = np.arange(1, seq_len+1).reshape(1, seq_len, 1).astype(np.float32)
            dims = np.arange(self.d_model).reshape(1, 1, self.d_model).astype(np.float32)

            angles = positions / np.power(10000, 2 * (dims // 2) / self.d_model)
            angles[:, :, 0::2] = np.sin(angles[:, :, 0::2])
            angles[:, :, 1::2] = np.cos(angles[:, :, 1::2])

            self._positional_encoding_cache[seq_len] = angles.astype(np.float32)
        return (embeddings + self._positional_encoding_cache[seq_len]).astype(np.float32)
