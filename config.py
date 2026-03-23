from typing import Final

import numpy as np

EMBEDDING_DIM: Final = 128
D_MODEL: Final = 128
ENCODER_LAYERS: Final = 6
DECODER_LAYERS: Final = 6
NUM_HEADS: Final = 12
HIDDEN_LAYER: Final = D_MODEL * 4
LEARNING_RATE: Final = np.float32(0.01)
ALPHA: Final = np.float32(0.001)
