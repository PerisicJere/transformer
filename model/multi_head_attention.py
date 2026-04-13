import cupy as np

from model.linear import Linear
from model.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention:
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mask: bool):
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Wq = Linear(in_dim=in_dim, out_dim=out_dim * num_heads)
        self.Wk = Linear(in_dim=in_dim, out_dim=out_dim * num_heads)
        self.Wv = Linear(in_dim=in_dim, out_dim=out_dim * num_heads)
        self.attention = ScaledDotProductAttention(d_k=out_dim, num_heads=num_heads, mask=mask)
        self.final_projection = Linear(in_dim=out_dim * num_heads, out_dim=in_dim)

    def __call__(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        batch, seq_len_q, _ = Q.shape
        batch, seq_len_k, _ = K.shape
        batch, seq_len_v, _ = V.shape

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = Q.reshape(batch, seq_len_q, self.num_heads, self.out_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len_k, self.num_heads, self.out_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len_v, self.num_heads, self.out_dim).transpose(0, 2, 1, 3)

        out = self.attention.forward(Q, K, V).transpose(0, 2, 1, 3).reshape(batch, seq_len_q, self.num_heads*self.out_dim)

        return self.final_projection(out)

    def backward(self,
                 gradients: np.ndarray,
                 learning_rate: np.float32,
                 encoder_input: bool = False) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        dOut = self.final_projection.backward(gradients=gradients, learning_rate=learning_rate)

        batch, seq_len_q, _ = dOut.shape
        dOut = dOut.reshape(batch, seq_len_q, self.num_heads, self.out_dim).transpose(0, 2, 1, 3)

        dQ, dK, dV = self.attention.backward(dOut)

        dQ = dQ.transpose(0, 2, 1, 3).reshape(batch, seq_len_q, self.num_heads*self.out_dim)
        dK = dK.transpose(0, 2, 1, 3).reshape(batch, dK.shape[2], self.num_heads*self.out_dim)
        dV = dV.transpose(0, 2, 1, 3).reshape(batch, dV.shape[2], self.num_heads*self.out_dim)

        if encoder_input:
            dQx = self.Wq.backward(dQ, learning_rate=learning_rate)
            dx = (self.Wk.backward(dK, learning_rate=learning_rate)
                  + self.Wv.backward(dV, learning_rate=learning_rate))
            return np.clip(dQx, -5, 5), np.clip(dx, -5, 5)
        else:
            dx = (self.Wq.backward(dQ, learning_rate=learning_rate)
                  + self.Wk.backward(dK, learning_rate=learning_rate)
                  + self.Wv.backward(dV, learning_rate=learning_rate))
            return np.clip(dx, -5, 5)