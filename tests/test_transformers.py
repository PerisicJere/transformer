import numpy as np

from model.embedding import Embedding
from model.encoder_decoder_transformer import EncoderDecoderTransformer
from model.positional_encoding import PositionalEncoding

EMBEDDING_DIM = 4

def test_transformer():
    croatian: list[str] = ['Ja', 'Sam', 'Jere', 'Perisic', '<PAD>']
    english: list[str] = ['I', 'Am', 'Jere', 'Perisic', '<PAD>']
    cro_embedding = Embedding(vocab_size=len(croatian), embedding_size=EMBEDDING_DIM)
    eng_embedding = Embedding(vocab_size=len(english), embedding_size=EMBEDDING_DIM)

    cro_pse = PositionalEncoding(d_model=3)(embeddings=cro_embedding.embedding_weights)
    eng_pse = PositionalEncoding(d_model=3)(embeddings=eng_embedding.embedding_weights)

    transformer = EncoderDecoderTransformer(decoder_layers=6, encoder_layers=6, in_dim=EMBEDDING_DIM, hidden_layer=8, num_heads=8)
    output = transformer.forward(
        encoder_embeddings=cro_pse,
        decoder_embeddings=eng_pse,
        src_pad_mask=np.array([0.0, 0.0, 0.0, 0.0, -np.inf]),
        target_pad_mask=np.array([0.0, 0.0, 0.0, 0.0, -np.inf]),
    )

    assert output.shape == (len(croatian), EMBEDDING_DIM)
