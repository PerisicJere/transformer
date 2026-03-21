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
        self.final_projection = Linear(in_dim=out_dim*num_heads, out_dim=in_dim)

    def __call__(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        heads: list[np.ndarray] = []
        for Wq, Wk, Wv, attention in self.head_projections:
            Qi = Wq(Q)
            Ki = Wk(K)
            Vi = Wv(V)
            heads.append(attention.forward(Q=Qi,K=Ki,V=Vi))

        concat = np.concatenate(heads, axis=1)
        multi_head_attention = self.final_projection(concat)
        return multi_head_attention

    def backward(self, gradients: np.ndarray, encoder_input: bool = False) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        linear_backward = self.final_projection.backward(gradients=gradients)
        heads = np.split(linear_backward, self.num_heads, axis=1)
        dx: np.ndarray = np.zeros_like(heads[0])
        dQx: np.ndarray = np.zeros_like(heads[0])
        for idx, weights in enumerate(self.head_projections):
            Wq, Wk, Wv, attention = weights
            dQi, dKi, dVi = attention.backward(gradients=heads[idx])
            if encoder_input:
                dQx += Wq.backward(dQi)
                dx += (Wk.backward(dKi) + Wv.backward(dVi))
            else:
                dx += (Wq.backward(dQi) + Wk.backward(dKi) + Wv.backward(dVi))

        dx = np.clip(dx, -1, 1)
        dQx = np.clip(dQx, -1, 1)
        return (dx, dQx) if encoder_input else dx
