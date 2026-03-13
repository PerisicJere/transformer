import numpy as np

from model.linear import Linear
from model.multi_head_attention import MultiHeadAttention
from model.scaled_dot_product_attention import ScaledDotProductAttention


def test_multi_head_attention():
    Q = np.random.randn(10, 64)
    K = np.random.randn(10, 64)
    V = np.random.randn(10, 64)

    mha = MultiHeadAttention(in_dim=64, out_dim=3, num_heads=8)(Q=Q, K=K, V=V)

    assert mha.shape == (10, 64)

