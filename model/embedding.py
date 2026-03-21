import numpy as np


class Embedding:
    def __init__(self, vocab_size: int, embedding_size: int):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding_weights = np.random.randn(vocab_size, embedding_size) * np.sqrt(2 / (vocab_size+embedding_size)) # Each token gets one row, one row is a dense vector
        self.mappings: dict[str, int] = {} # Each token gets token ID

    def add_mapping(self, tokens: list[str]) -> None:
        for token in tokens:
            if token not in self.mappings:
                self.mappings[token] = len(self.mappings)

    def construct_table(self, tokens: list[str]) -> np.ndarray:
        embedding_rows: np.ndarray = np.array(
            [self.get_embedding_vector(token) for token in tokens]
        )

        return embedding_rows

    def get_targets(self, tokens: list[str]) -> np.ndarray:
        targets: np.ndarray = np.zeros((len(tokens), self.vocab_size))
        for idx, token in enumerate(tokens):
            token_idx = self.mappings[token]
            targets[idx][token_idx] = 1.0
        return targets

    def get_list_of_token_ids(self, tokens: list[str]) -> np.ndarray:
        return np.array([self.__get_token_id(token=token) for token in tokens])

    def __get_token_id(self, token: str) -> int:
        assert token in self.mappings, f"No token found: {token}"
        return self.mappings[token]

    def get_embedding_vector(self, token: str) -> np.ndarray:
        return self.embedding_weights[self.__get_token_id(token)]

    def backward(self, gradients: np.ndarray, target_indices: np.ndarray, learning_rate: np.float32) -> None:
        for i, token_idx in enumerate(target_indices):
            self.embedding_weights[token_idx] -= learning_rate * np.clip(gradients[i], -1, 1)