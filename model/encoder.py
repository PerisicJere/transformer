import numpy as np

from model.feed_forward_nn import FeedForwardNN
from model.layer_normalization import LayerNormalization
from model.multi_head_attention import MultiHeadAttention


class Encoder:
    def __init__(self, in_dim: int) -> None:
        self.multi_head_attention = MultiHeadAttention(
            in_dim=in_dim,
            out_dim=12,
            num_heads=8,
        )
        self.layer_norm = LayerNormalization(d_model=in_dim)
        self.ffnn = FeedForwardNN(input_size=in_dim,
                      output_size=in_dim,
                      hidden_layer=8)

    def forward(self, x: np.ndarray) -> np.ndarray:
        multi_head_attention = self.multi_head_attention(Q=x, K=x, V=x)
        multi_head_residual = x + multi_head_attention
        layer_norm = self.layer_norm.normalize(multi_head_residual)
        feed_forward = self.ffnn.forward_propagation(layer_norm)
        feed_forward_residual = layer_norm + feed_forward
        second_layer_norm = self.layer_norm.normalize(feed_forward_residual)
        return second_layer_norm
