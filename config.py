from typing import Final

import numpy as np

EMBEDDING_DIM: Final = 128
D_MODEL: int = EMBEDDING_DIM
ENCODER_LAYERS: Final = 2
DECODER_LAYERS: Final = 2
NUM_HEADS: Final = 4
HIDDEN_LAYER: Final = D_MODEL * 4
LEARNING_RATE: Final = np.float32(0.0001)
