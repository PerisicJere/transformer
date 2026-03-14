import numpy as np

from model.feed_forward_nn import FeedForwardNN


def test_feed_forward_neural_network__forward_propagation():
    ffnn = FeedForwardNN(input_size=2, output_size=2, hidden_layer=8)
    inputs = np.random.randn(5, 2)
    assert ffnn.forward_propagation(inputs).shape == (5, 2)

def test_feed_forward_neural_network__relu():
    ffnn = FeedForwardNN(input_size=2, output_size=2, hidden_layer=8)
    inputs = np.array([-0.21, 0.2, 0.25, -0.14])
    relu_output = ffnn.relu(inputs)
    assert np.equal(relu_output, [0., 0.2, 0.25, 0.]).all()