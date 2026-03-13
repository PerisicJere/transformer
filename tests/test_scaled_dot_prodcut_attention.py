from model.scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np

def test_scaled_dot_product_attention__forward():
    Q = np.random.randn(10, 64)
    K = np.random.randn(10, 64)
    V = np.random.randn(10, 64)
    attention = ScaledDotProductAttention(d_k=K.shape[-1])
    result = attention.forward(Q=Q, K=K, V=V)

    assert result.shape == (10, 64)

def test_scaled_dot_product_attention__softmax():
    attention = ScaledDotProductAttention(d_k=64)
    x = np.array([10000, 20000, 30000])
    sftmx = attention._softmax(x)

    assert not np.any(np.isnan(sftmx))




