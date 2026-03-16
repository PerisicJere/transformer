from model.embedding import Embedding
from model.encoder_decoder_transformer import EncoderDecoderTransformer
from model.positional_encoding import PositionalEncoding

EMBEDDING_DIM = 4

def test_transformer():
    croatian: set[str] = {'Ja', 'Sam', 'Jere', 'Perisic'}
    english: set[str] = {'I', 'Am', 'Jere', 'Perisic'}
    cro_embedding = Embedding(vocab_size=len(croatian), embedding_size=EMBEDDING_DIM)
    eng_embedding = Embedding(vocab_size=len(english), embedding_size=EMBEDDING_DIM)

    cro_pse = PositionalEncoding(d_model=3)(embeddings=cro_embedding.embedding_weights)
    eng_pse = PositionalEncoding(d_model=3)(embeddings=eng_embedding.embedding_weights)

    transformer = EncoderDecoderTransformer(decoder_layers=6, encoder_layers=6, in_dim=EMBEDDING_DIM)
    output = transformer.forward(encoder_embeddings=cro_pse, decoder_embeddings=eng_pse)

    assert output.shape == (len(croatian), EMBEDDING_DIM)
