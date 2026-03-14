import numpy as np

from model.embedding import Embedding
from model.encoder import Encoder
from model.positional_encoding import PositionalEncoding


def test_encoder():
    tokens: set[str] = {'Transformer', 'Attention', 'FeedForward', 'Transformer'}
    embedding = Embedding(vocab_size=len(tokens), embedding_size=12)
    # d_model == len(embedding.embedding_weights)
    pse = PositionalEncoding(vector=embedding.embedding_weights, d_model=3)()

    encoders = [Encoder(in_dim=pse.shape[1]) for _ in range(6)]
    x = encoders[0].forward(pse)
    for encoder in encoders[1:]:
        x = encoder.forward(x)

    assert pse.shape == x.shape