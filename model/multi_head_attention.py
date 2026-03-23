import numpy as np

from model.linear import Linear
from model.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention:
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mask: bool):
        self.num_heads = num_heads
        self.head_projections = [
            (Linear(in_dim=in_dim, out_dim=out_dim),
             Linear(in_dim=in_dim, out_dim=out_dim),
             Linear(in_dim=in_dim, out_dim=out_dim),
             ScaledDotProductAttention(d_k=out_dim, num_heads=num_heads, mask=mask))
            for _ in range(num_heads)
        ]
        self.final_projection = Linear(in_dim=out_dim * num_heads, out_dim=in_dim)

    def __call__(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        heads: list[np.ndarray] = []
        for Wq, Wk, Wv, attention in self.head_projections:
            Qi = Wq(Q)
            Ki = Wk(K)
            Vi = Wv(V)
            heads.append(attention.forward(Q=Qi, K=Ki, V=Vi))

        concat = np.concatenate(heads, axis=1)
        multi_head_attention = self.final_projection(concat)
        return multi_head_attention

    def backward(self,
                 gradients: np.ndarray,
                 learning_rate: np.float32,
                 encoder_input: bool = False) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        linear_backward = self.final_projection.backward(gradients=gradients, learning_rate=learning_rate)
        heads = np.split(linear_backward, self.num_heads, axis=1)
        dx, dQx = None, None
        if encoder_input:
            for idx, (Wq, Wk, Wv, attention) in enumerate(self.head_projections):
                dQi, dKi, dVi = attention.backward(gradients=heads[idx])
                kv = Wk.backward(dKi, learning_rate=learning_rate) + Wv.backward(dVi, learning_rate=learning_rate)
                q = Wq.backward(dQi, learning_rate=learning_rate)
                dQx = q if dQx is None else dQx + q
                dx = kv if dx is None else dx + kv
        else:
            for idx, (Wq, Wk, Wv, attention) in enumerate(self.head_projections):
                dQi, dKi, dVi = attention.backward(gradients=heads[idx])
                total = (
                        Wk.backward(dKi, learning_rate=learning_rate)
                        + Wv.backward(dVi, learning_rate=learning_rate)
                        + Wq.backward(dQi, learning_rate=learning_rate)
                )
                dx = total if dx is None else dx + total

        return (np.clip(dQx, -5, 5), np.clip(dx, -5, 5)) if encoder_input else np.clip(dx, -5, 5)
