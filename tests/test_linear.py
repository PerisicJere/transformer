import numpy as np

from model.linear import Linear


def test_linear__callable():
    # matrix 5x7, and matrix 5x3, should produce 5x3
    matrix: np.ndarray = np.random.randn(5, 7)
    linear: Linear = Linear(7, 3)
    new_matrix = linear(x=matrix)

    assert new_matrix.shape == (5, 3)
