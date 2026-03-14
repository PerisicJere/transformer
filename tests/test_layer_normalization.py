import numpy as np

from model.layer_normalization import LayerNormalization


def test_layer_normalization():
    x1: np.ndarray = np.array([[3., 5., 2., 8.], [1., 3., 5., 8.], [3., 2., 7., 9.]])
    lay_norm = LayerNormalization(d_model=4)
    val = lay_norm.normalize(x1)

    assert np.allclose(val[0], np.array([-0.48198041,  0.8273268,  -1.13663402,  2.79128763]))

