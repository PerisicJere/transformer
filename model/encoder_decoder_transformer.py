import numpy as np

from model.decoder import Decoder
from model.encoder import Encoder


class EncoderDecoderTransformer:
    def __init__(self, decoder_layers: int, encoder_layers: int, num_heads: int, hidden_layer: int, in_dim: int):
        self.encoders = (
            [Encoder(in_dim=in_dim, num_heads=num_heads, hidden_layer=hidden_layer) for _ in range(encoder_layers)]
        )
        self.decoders = (
            [Decoder(in_dim=in_dim, num_heads=num_heads, hidden_layer=hidden_layer) for _ in range(decoder_layers)]
        )

    def forward(self,
                encoder_embeddings: np.ndarray,
                decoder_embeddings: np.ndarray,
                ) -> np.ndarray:
        # encoder
        encoder_output = self.encoders[0].forward(
            x=encoder_embeddings,
        )
        for encoder in self.encoders[1:]:
            encoder_output = encoder.forward(
                x=encoder_output,
            )

        # decoder
        decoder_output = self.decoders[0].forward(
            x=decoder_embeddings,
            K_encoder=encoder_output,
            V_encoder=encoder_output,
        )
        for decoder in self.decoders[1:]:
            decoder_output = decoder.forward(
                x=decoder_output,
                K_encoder=encoder_output,
                V_encoder=encoder_output,
            )

        return decoder_output

    def backward(
            self,
            d_decoder: np.ndarray,
            learning_rate: np.float32
    ) -> tuple[np.ndarray, np.ndarray]:
        d_encoder = None

        for decoder in reversed(self.decoders):
            d_decoder, dx = decoder.backward(gradients=d_decoder, learning_rate=learning_rate)
            d_encoder = dx if d_encoder is None else d_encoder + dx

        d_encoder = np.array(d_encoder)
        for encoder in reversed(self.encoders):
            d_encoder = encoder.backward(gradients=d_encoder, learning_rate=learning_rate)

        return d_decoder, d_encoder

    def translate(self, encoder_input: np.ndarray, decoder_input: np.ndarray) -> np.ndarray:
        encoder_output = self.encoders[0].forward(
            x=encoder_input,
        )
        for encoder in self.encoders[1:]:
            encoder_output = encoder.forward(
                x=encoder_output,
            )

        # decoder
        decoder_output = self.decoders[0].forward(
            x=decoder_input,
            K_encoder=encoder_output,
            V_encoder=encoder_output,
        )
        for decoder in self.decoders[1:]:
            decoder_output = decoder.forward(
                x=decoder_output,
                K_encoder=encoder_output,
                V_encoder=encoder_output,
            )

        return decoder_output
