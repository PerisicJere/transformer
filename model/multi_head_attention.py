import numpy as np

from model.linear import Linear
from model.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention:
    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        self.num_heads = num_heads
        self.attention = ScaledDotProductAttention(d_k=out_dim, num_heads=num_heads)
        self.head_projections = [
            (Linear(in_dim=in_dim, out_dim=out_dim),
            Linear(in_dim=in_dim, out_dim=out_dim),
            Linear(in_dim=in_dim, out_dim=out_dim))
            for _ in range(num_heads)
        ]
        self.final_projection = Linear(in_dim=out_dim*num_heads, out_dim=in_dim)

    def __call__(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        heads: list[np.ndarray] = []
        for Wq, Wk, Wv in self.head_projections:
            Qi = Wq(Q)
            Ki = Wk(K)
            Vi = Wv(V)
            heads.append(self.attention.forward(Q=Qi,K=Ki,V=Vi))

        concat = np.concatenate(heads, axis=1)
        multi_head_attention = self.final_projection(concat)
        return multi_head_attention