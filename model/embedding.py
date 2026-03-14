import numpy as np


class Embedding:
    def __init__(self, vocab_size: int, embedding_size: int):
        self.embedding_weights = np.random.randn(vocab_size, embedding_size) # Each token gets one row, one row is a dense vector
        self.mappings: dict[str, int] = {} # Each token gets token ID

    def add_mapping(self, mapping: str) -> None:
        if mapping not in self.mappings:
            self.mappings[mapping] = len(self.mappings)

    def __get_token_id(self, token: str) -> int:
        assert token in self.mappings, "No token found"
        return self.mappings[token]

    def get_embedding_vector(self, token: str) -> np.ndarray:
        assert token in self.mappings, "No token found"

        return self.embedding_weights[self.__get_token_id(token)]

