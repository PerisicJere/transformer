import cupy as np


class CrossEntropyLoss:
    @staticmethod
    def compute(targets: np.ndarray, probabilities: np.ndarray) -> np.float32:
        batch_size, seq_len = targets.shape

        batch_idx = np.arange(batch_size)[:, None]
        seq_idx   = np.arange(seq_len)[None, :]
        target_probs = probabilities[batch_idx, seq_idx, targets]

        loss = -np.log(np.clip(target_probs, 1e-9, 1.0)).mean()
        return loss