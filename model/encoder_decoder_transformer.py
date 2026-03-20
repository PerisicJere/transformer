import numpy as np

from model.decoder import Decoder
from model.encoder import Encoder
from model.linear import Linear


class EncoderDecoderTransformer:
    def __init__(self, decoder_layers: int, encoder_layers: int, in_dim: int):
        self.encoders = [Encoder(in_dim=in_dim) for _ in range(encoder_layers)]
        self.decoders = [Decoder(in_dim=in_dim) for _ in range(decoder_layers)]

    def forward(self, encoder_embeddings: np.ndarray, decoder_embeddings: np.ndarray) -> np.ndarray:
        # decoder
        encoder_output = self.encoders[0].forward(encoder_embeddings)
        for encoder in self.encoders[1:]:
            encoder_output = encoder.forward(encoder_output)

        # encoder
        decoder_output = self.decoders[0].forward(x=decoder_embeddings, Q_encoder=encoder_output, K_encoder=encoder_output)
        for decoder in self.decoders[1:]:
            decoder_output = decoder.forward(x=decoder_output, Q_encoder=encoder_output, K_encoder=encoder_output)

        return decoder_output

    def backward(self, probs: np.ndarray, targets: np.ndarray) -> None:
        gradients: np.ndarray = probs - targets
        d_encoder = np.zeros_like(gradients)
        for decoder in self.decoders:
            gradients, dx = decoder.backward(gradients=gradients)
            d_encoder += dx

        for encoder in self.encoders:
            d_encoder = encoder.backward(gradients=d_encoder)
