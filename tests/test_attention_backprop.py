import numpy as np

from model.cross_entropy_loss import CrossEntropyLoss
from model.embedding import Embedding
from model.encoder_decoder_transformer import EncoderDecoderTransformer
from model.linear import Linear
from model.positional_encoding import PositionalEncoding

EMBEDDING_DIM = 4

def test_attention_backprop():
    cro_list: list[str] = ['Ja', 'Sam', 'Jere', 'Perisic', 'Ja']
    eng_list: list[str] = ['I', 'Am', 'Jere', 'Perisic', 'I']

    cro_embedding = Embedding(vocab_size=len(set(cro_list)), embedding_size=EMBEDDING_DIM)
    eng_embedding = Embedding(vocab_size=len(set(eng_list)), embedding_size=EMBEDDING_DIM)

    cro_embedding.add_mapping(cro_list)
    eng_embedding.add_mapping(eng_list)

    targets: np.ndarray = eng_embedding.get_targets(eng_list)

    cro_pse = PositionalEncoding(d_model=3)(embeddings=cro_embedding.construct_table(tokens=cro_list))
    eng_pse = PositionalEncoding(d_model=3)(embeddings=eng_embedding.construct_table(tokens=eng_list))

    transformer = EncoderDecoderTransformer(decoder_layers=6, encoder_layers=6, in_dim=EMBEDDING_DIM)
    output = transformer.forward(encoder_embeddings=cro_pse, decoder_embeddings=eng_pse)

    lin = Linear(in_dim=4, out_dim=4)(output)
    probs = softmax(x=lin)

    gradients_decoder, gradients_encoder = transformer.backward(
        probs=probs,
        targets=targets,
    )
    cro_embedding.backward(gradients=gradients_encoder, target_indices=cro_embedding.get_list_of_token_ids(cro_list))
    eng_embedding.backward(gradients=gradients_decoder, target_indices=eng_embedding.get_list_of_token_ids(eng_list))

    loss = CrossEntropyLoss()
    loss = loss.compute(targets=cro_embedding.get_list_of_token_ids(cro_list), probabilities=probs)

    assert type(loss) == np.float64

def softmax(x: np.ndarray) -> np.ndarray:
    max_val = np.max(x, axis=-1, keepdims=True)
    exp_val = np.exp(x - max_val)
    return exp_val / np.sum(exp_val, axis=-1, keepdims=True)
