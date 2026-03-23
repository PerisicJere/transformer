from model.decoder import Decoder
from model.embedding import Embedding
from model.encoder import Encoder
from model.positional_encoding import PositionalEncoding


def test_decoder():
    tokens: set[str] = {'Transformer', 'Attention', 'FeedForward', 'Transformer'}
    embedding = Embedding(vocab_size=len(tokens), embedding_size=12)
    # d_model == len(embedding.embedding_weights)
    pse = PositionalEncoding(d_model=3)(embeddings=embedding.embedding_weights)

    encoders = [Encoder(in_dim=pse.shape[1], num_heads=8, hidden_layer=8) for _ in range(6)]
    encoder_output = encoders[0].forward(pse, src_pad_mask=None)
    for encoder in encoders[1:]:
        encoder_output = encoder.forward(encoder_output, src_pad_mask=None)

    decoders = [Decoder(in_dim=pse.shape[1], hidden_layer=8, num_heads=8) for _ in range(6)]
    decoder_output = decoders[0].forward(
        x=pse,
        K_encoder=encoder_output,
        V_encoder=encoder_output,
        target_pad_mask=None,
        src_pad_mask=None,
    )
    for decoder in decoders[1:]:
        decoder_output = decoder.forward(
            decoder_output,
            K_encoder=encoder_output,
            V_encoder=encoder_output,
            target_pad_mask=None,
            src_pad_mask=None,
        )


    assert decoder_output.shape == pse.shape
