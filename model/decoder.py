import numpy as np

from model.feed_forward_nn import FeedForwardNN
from model.layer_normalization import LayerNormalization
from model.multi_head_attention import MultiHeadAttention


class Decoder:
    def __init__(self, in_dim: int):
        self.masked_multi_head_attention = MultiHeadAttention(
            in_dim=in_dim,
            out_dim=in_dim,
            num_heads=8,
            mask=True
        )
        self.multi_head_attention = MultiHeadAttention(
            in_dim=in_dim,
            out_dim=in_dim,
            num_heads=8,
            mask=False
        )

        self.layer_norm1 = LayerNormalization(d_model=in_dim)
        self.layer_norm2 = LayerNormalization(d_model=in_dim)
        self.layer_norm3 = LayerNormalization(d_model=in_dim)

        self.ffnn = FeedForwardNN(
            input_size=in_dim,
            output_size=in_dim,
            hidden_layer=8
        )


    def forward(self, x: np.ndarray, Q_encoder: np.ndarray, K_encoder: np.ndarray) -> np.ndarray:
        # First layer
        masked_multi_head_attention = self.masked_multi_head_attention(
            Q=x,
            K=x,
            V=x
        )
        masked_multi_head_residual = x + masked_multi_head_attention
        layer_norm = self.layer_norm1.normalize(masked_multi_head_residual)

        # Second layer
        multi_head_attention = self.multi_head_attention(
            Q=Q_encoder,
            K=K_encoder,
            V=layer_norm
        )
        multi_head_residual = layer_norm + multi_head_attention
        second_layer_norm = self.layer_norm2.normalize(multi_head_residual)

        # Third layer
        feed_forward = self.ffnn.forward_propagation(second_layer_norm)
        feed_forward_residual = second_layer_norm + feed_forward
        third_layer_norm = self.layer_norm3.normalize(feed_forward_residual)

        return third_layer_norm

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        # third
        layer_norm_3__backprop_output = self.layer_norm3.backward(gradients=gradients)
        ffnn__backprop_output = self.ffnn.backward_propagation(gradients=layer_norm_3__backprop_output)
        residual_gradient_1  = layer_norm_3__backprop_output + ffnn__backprop_output

        # second
        layer_norm_2__backprop_output = self.layer_norm2.backward(gradients=ffnn__backprop_output)
        multi_head_attention__backprop_output = self.multi_head_attention.backward(gradients=layer_norm_2__backprop_output)
        residual_gradient_2 = multi_head_attention__backprop_output + layer_norm_2__backprop_output

        # first
        layer_norm_1__backprop_output = self.layer_norm1.backward(gradients=residual_gradient_2)
        masked_multi_head_attention__backprop_output = self.masked_multi_head_attention.backward(gradients=layer_norm_1__backprop_output)
        residual_gradient_3 = masked_multi_head_attention__backprop_output + layer_norm_1__backprop_output

        return residual_gradient_3
