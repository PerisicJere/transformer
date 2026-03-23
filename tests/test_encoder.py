import numpy as np

from model.embedding import Embedding
from model.encoder import Encoder
from model.positional_encoding import PositionalEncoding


def test_encoder():
    tokens: set[str] = {'Transformer', 'Attention', 'FeedForward', 'Transformer'}
    embedding = Embedding(vocab_size=len(tokens), embedding_size=12)
    # d_model == len(embedding.embedding_weights)
    pse = PositionalEncoding(d_model=3)(embeddings=embedding.embedding_weights)

    encoders = [Encoder(in_dim=pse.shape[1], num_heads=8, hidden_layer=8) for _ in range(6)]
    x = encoders[0].forward(pse, src_pad_mask=None)
    for encoder in encoders[1:]:
        x = encoder.forward(x, src_pad_mask=None)

    assert pse.shape == x.shape