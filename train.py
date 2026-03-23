import re
from typing import Final

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from config import EMBEDDING_DIM, LEARNING_RATE, ALPHA, DECODER_LAYERS, ENCODER_LAYERS, NUM_HEADS, HIDDEN_LAYER
from model.cross_entropy_loss import CrossEntropyLoss
from model.embedding import Embedding
from model.encoder_decoder_transformer import EncoderDecoderTransformer
from model.linear import Linear
from model.positional_encoding import PositionalEncoding
from model.softmax import softmax

_STRING_CLEAN: Final = re.compile(r'[^\w\s]')

def translate(transformer: EncoderDecoderTransformer, hrvatski: Embedding, engleski: Embedding):
    sentence = ["<SOS>", "Ja", "sam", "student", "<EOS>"]
    cro_pse = PositionalEncoding(d_model=EMBEDDING_DIM)(
        embeddings=hrvatski.construct_table(tokens=sentence)
    )
    to_translate = ["<SOS>"]
    while True:
        eng_pse = PositionalEncoding(d_model=EMBEDDING_DIM)(
            embeddings=engleski.construct_table(tokens=to_translate)
        )
        output = transformer.translate(encoder_input=cro_pse, decoder_input=eng_pse)
        lin = linear(output)
        probs = softmax(input=lin)
        token_id: np.int32 = np.argmax(probs[-1])

        translated = eng_embedding.get_embedding_key(token_id.astype(int))
        to_translate.append(translated)

        if translated == "<EOS>" or len(to_translate) > 7:
            break

    print(f"Croatian Sentence: {' '.join(sentence)}")
    print(f"English Translation: {' '.join(to_translate)}")


if __name__ == '__main__':
    df: DataFrame = pd.read_csv('data/EngCro.tsv', sep='\t')
    df['Eng'] = (df['Eng'].apply(lambda row: [re.sub(_STRING_CLEAN, '', token) for token in row.split(' ')])
                 .apply(lambda row: ["<SOS>"] + row + ["<EOS>"]))
    df['Cro'] = (df['Cro'].apply(lambda row: [re.sub(_STRING_CLEAN, '', token) for token in row.split(' ')])
                 .apply(lambda row: ["<SOS>"] + row + ["<EOS>"]))

    cro_set: set[str] = set()
    eng_set: set[str] = set()
    for cro_val, eng_val in zip(df['Cro'].tolist(), df['Eng'].tolist()):
        cro_set.update(cro_val)
        eng_set.update(eng_val)

    cro_embedding: Embedding = Embedding(vocab_size=len(cro_set), embedding_size=EMBEDDING_DIM)
    eng_embedding: Embedding = Embedding(vocab_size=len(eng_set), embedding_size=EMBEDDING_DIM)

    cro_embedding.add_mapping(list(cro_set))
    eng_embedding.add_mapping(list(eng_set))

    transformer_train = EncoderDecoderTransformer(
        decoder_layers=DECODER_LAYERS,
        encoder_layers=ENCODER_LAYERS,
        num_heads=NUM_HEADS,
        hidden_layer=HIDDEN_LAYER,
        in_dim=EMBEDDING_DIM,
    )
    linear = Linear(in_dim=EMBEDDING_DIM, out_dim=len(eng_set))
    loss = CrossEntropyLoss()
    losses: list[float] = []
    epochs = 50
    pbar = tqdm(range(epochs))
    steps = len(df)
    total_steps: int = epochs * steps
    lr_new = LEARNING_RATE
    for epoch in pbar:
        loss_sum = 0.0
        df = df.sample(frac=1).reset_index(drop=True)
        for i, (croatian_input, english_input) in enumerate(zip(df['Cro'], df['Eng'])):
            targets: np.ndarray = eng_embedding.get_targets(english_input)

            cro_pse = PositionalEncoding(d_model=EMBEDDING_DIM)(
                embeddings=cro_embedding.construct_table(tokens=croatian_input)
            )
            eng_pse = PositionalEncoding(d_model=EMBEDDING_DIM)(
                embeddings=eng_embedding.construct_table(tokens=english_input)
            )

            output = transformer_train.forward(
                encoder_embeddings=cro_pse,
                decoder_embeddings=eng_pse,
            )
            lin = linear(output)
            probs = softmax(input=lin)

            loss_value = loss.compute(targets=eng_embedding.get_list_of_token_ids(english_input), probabilities=probs)
            loss_sum += loss_value

            gradients_decoder, gradients_encoder = transformer_train.backward(
                d_decoder=linear.backward(probs-targets, learning_rate=lr_new),
                learning_rate=lr_new
            )

            cro_embedding.backward(
                gradients=gradients_encoder,
                target_indices=cro_embedding.get_list_of_token_ids(croatian_input),
                learning_rate=lr_new
            )

            eng_embedding.backward(
                gradients=gradients_decoder,
                target_indices=eng_embedding.get_list_of_token_ids(english_input),
                learning_rate=lr_new
            )

            lr_new = ALPHA + 0.5*(LEARNING_RATE - ALPHA)*(1 + np.cos((np.pi * ((steps*epoch)+i)) / total_steps))

            pbar.set_description(f"Epoch {epoch} | Curr Loss: {float(loss_sum / len(df)):.4f} | Learning Rate: {float(lr_new):.20f} | Steps: {int((steps*epoch)+i)}/{int(total_steps)}")

        if epoch % 10 == 0:
            translate(transformer=transformer_train, hrvatski=cro_embedding, engleski=eng_embedding)

        losses.append(float(loss_sum / len(df)))

    print(losses)
