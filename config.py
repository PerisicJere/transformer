from typing import Final

import numpy as np

EMBEDDING_DIM: Final = 128
D_MODEL: Final = 128
ENCODER_LAYERS: Final = 4
DECODER_LAYERS: Final = 4
NUM_HEADS: Final = 12
HIDDEN_LAYER: Final = D_MODEL * 4
LEARNING_RATE: Final = np.float32(0.0003)
ALPHA: Final = np.float32(0.00001)
