import numpy as np
import pytest

from model.positional_encoding import PositionalEncoding


@pytest.mark.parametrize("d_model", [128, 512, 1024])
def test_positional_encoding(d_model):
    value: int = 64
    pse = PositionalEncoding(d_model=d_model)(embeddings=np.random.randn(value, d_model))
    assert pse.shape == (value, d_model)


