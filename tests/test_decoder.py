from model.decoder import Decoder
from model.embedding import Embedding
from model.encoder import Encoder
from model.positional_encoding import PositionalEncoding


def test_decoder():
    tokens: set[str] = {'Transformer', 'Attention', 'FeedForward', 'Transformer'}
    embedding = Embedding(vocab_size=len(tokens), embedding_size=12)
    # d_model == len(embedding.embedding_weights)
    pse = PositionalEncoding(vector=embedding.embedding_weights, d_model=3)()

    encoders = [Encoder(in_dim=pse.shape[1]) for _ in range(6)]
    encoder_output = encoders[0].forward(pse)
    for encoder in encoders[1:]:
        encoder_output = encoder.forward(encoder_output)

    decoders = [Decoder(in_dim=pse.shape[1]) for _ in range(6)]
    decoder_output = decoders[0].forward(x=pse, Q_encoder=encoder_output, K_encoder=encoder_output)
    for decoder in decoders[1:]:
        decoder_output = decoder.forward(decoder_output, Q_encoder=encoder_output, K_encoder=encoder_output)


    assert decoder_output.shape == pse.shape
