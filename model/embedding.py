import cupy as np


class Embedding:
    def __init__(self, vocab_size: int, embedding_size: int):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding_weights = (np.random.randn(vocab_size, embedding_size) * np.sqrt(2 / (vocab_size+embedding_size))).astype(np.float32) # Each token gets one row, one row is a dense vector
        self.mappings: dict[str, int] = {} # Each token gets token ID

    def add_mapping(self, tokens: list[str]) -> None:
        for token in tokens:
            if token not in self.mappings:
                self.mappings[token] = len(self.mappings)

    def construct_table(self, batch_sentences: list[list[str]]) -> np.ndarray:
        embedding_rows: np.ndarray = np.array(
            [[self.mappings[token]
             for token in sentence
             ] for sentence in batch_sentences]
        )
        return self.embedding_weights[embedding_rows].astype(np.float32)

    def get_targets(self, sentences: list[list[str]]) -> np.ndarray:
        ids = self.get_list_of_token_ids(sentences)
        batch, seq_len = ids.shape
        targets = np.zeros((batch, seq_len, self.vocab_size))
        b_idx = np.arange(batch)[:, None]
        s_idx = np.arange(seq_len)[None, :]
        targets[b_idx, s_idx, ids] = 1.0
        return targets

    def get_list_of_token_ids(self, sentences: list[list[str]]) -> np.ndarray:
        token_ids = []
        for tokens in sentences:
            token_ids.append([self.__get_token_id(token=token) for token in tokens])
        return np.array(token_ids)

    def __get_token_id(self, token: str) -> int:
        assert token in self.mappings, f"No token found: {token}"
        return self.mappings[token]

    def get_embedding_vector(self, tokens: list[str]) -> np.ndarray:
        return np.array([self.embedding_weights[self.__get_token_id(token)] for token in tokens]).astype(np.float32)

    def backward(self, gradients: np.ndarray, target_indices: np.ndarray, learning_rate: np.float32) -> None:
        indices = target_indices.reshape(-1)
        gradients = gradients.reshape(-1, self.embedding_size)

        np.add.at(self.embedding_weights, indices, -learning_rate * gradients)

    def get_embedding_key(self, token_idx: int) -> str:
        for token_key, idx in self.mappings.items():
            if idx == token_idx:
                return token_key
        assert False, f"No token found: {token_idx}"
