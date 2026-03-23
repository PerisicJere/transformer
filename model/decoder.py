import numpy as np

from model.feed_forward_nn import FeedForwardNN
from model.layer_normalization import LayerNormalization
from model.multi_head_attention import MultiHeadAttention


class Decoder:
    def __init__(self, in_dim: int, num_heads: int, hidden_layer: int):
        self.masked_multi_head_attention = MultiHeadAttention(
            in_dim=in_dim,
            out_dim=in_dim,
            num_heads=num_heads,
            mask=True
        )
        self.multi_head_attention = MultiHeadAttention(
            in_dim=in_dim,
            out_dim=in_dim,
            num_heads=num_heads,
            mask=False
        )

        self.layer_norm1 = LayerNormalization(d_model=in_dim)
        self.layer_norm2 = LayerNormalization(d_model=in_dim)
        self.layer_norm3 = LayerNormalization(d_model=in_dim)

        self.ffnn = FeedForwardNN(
            input_size=in_dim,
            output_size=in_dim,
            hidden_layer=hidden_layer
        )


    def forward(self, x: np.ndarray, K_encoder: np.ndarray, V_encoder: np.ndarray) -> np.ndarray:
        # First layer
        masked_multi_head_attention = self.masked_multi_head_attention(
            Q=x,
            K=x,
            V=x,
        )
        masked_multi_head_residual = x + masked_multi_head_attention
        layer_norm = self.layer_norm1.normalize(masked_multi_head_residual)

        # Second layer
        multi_head_attention = self.multi_head_attention(
            Q=layer_norm,
            K=K_encoder,
            V=V_encoder,
        )
        multi_head_residual = layer_norm + multi_head_attention
        second_layer_norm = self.layer_norm2.normalize(multi_head_residual)

        # Third layer
        feed_forward = self.ffnn.forward_propagation(second_layer_norm)
        feed_forward_residual = second_layer_norm + feed_forward
        third_layer_norm = self.layer_norm3.normalize(feed_forward_residual)

        return third_layer_norm

    def backward(self, gradients: np.ndarray, learning_rate: np.float32) -> tuple[np.ndarray, np.ndarray]:
        # third
        layer_norm_3__backprop_output = self.layer_norm3.backward(gradients=gradients, learning_rate=learning_rate)
        ffnn__backprop_output = self.ffnn.backward_propagation(gradients=layer_norm_3__backprop_output, learning_rate=learning_rate)
        residual_gradient_1  = layer_norm_3__backprop_output + ffnn__backprop_output

        # second
        layer_norm_2__backprop_output = self.layer_norm2.backward(gradients=residual_gradient_1, learning_rate=learning_rate)
        multi_head_attention__backprop_output, d_encoder = self.multi_head_attention.backward(gradients=layer_norm_2__backprop_output, learning_rate=learning_rate, encoder_input=True)
        residual_gradient_2 = multi_head_attention__backprop_output + layer_norm_2__backprop_output

        # first
        layer_norm_1__backprop_output = self.layer_norm1.backward(gradients=residual_gradient_2, learning_rate=learning_rate)
        masked_multi_head_attention__backprop_output = self.masked_multi_head_attention.backward(gradients=layer_norm_1__backprop_output, learning_rate=learning_rate)
        residual_gradient_3 = masked_multi_head_attention__backprop_output + layer_norm_1__backprop_output

        return np.clip(residual_gradient_3, -5, 5), np.clip(d_encoder, -5, 5)
