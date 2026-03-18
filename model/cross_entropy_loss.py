import numpy as np


class CrossEntropyLoss:
    @staticmethod
    def compute(targets: list[int], probabilities: np.ndarray) -> np.float32:
        loss: np.float32 = np.float32(0.0)
        for i, target_id in enumerate(targets):
            loss += -np.log(probabilities[i][target_id])
        loss /= len(targets)

        return loss
