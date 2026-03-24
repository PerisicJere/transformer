import re
from typing import Final

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from config import EMBEDDING_DIM, LEARNING_RATE, ALPHA, DECODER_LAYERS, ENCODER_LAYERS, NUM_HEADS, HIDDEN_LAYER, D_MODEL
from model.cross_entropy_loss import CrossEntropyLoss
from model.embedding import Embedding
from model.encoder_decoder_transformer import EncoderDecoderTransformer
from model.linear import Linear
from model.positional_encoding import PositionalEncoding
from model.softmax import softmax

_STRING_CLEAN: Final = re.compile(r'[^\w\s]')

def translate(transformer: EncoderDecoderTransformer, francuski: Embedding, engleski: Embedding):
    sentence = ["<SOS>", "I", "am", "student", "<EOS>"]
    eng_pse = PositionalEncoding(d_model=EMBEDDING_DIM)(
        embeddings=engleski.construct_table(tokens=sentence)
    )
    to_translate = ["<SOS>"]
    while True:
        fra_pse = PositionalEncoding(d_model=EMBEDDING_DIM)(
            embeddings=francuski.construct_table(tokens=to_translate)
        )
        output = transformer.translate(encoder_input=fra_pse, decoder_input=eng_pse)
        lin = linear(output)
        probs = softmax(input=lin)
        token_id: np.int32 = np.argmax(probs[-1])

        translated = francuski.get_embedding_key(token_id.astype(int))
        to_translate.append(translated)

        if translated == "<EOS>" or len(to_translate) > 7:
            break

    print(f"French Sentence: {' '.join(sentence)}")
    print(f"English Translation: {' '.join(to_translate)}")


if __name__ == '__main__':
    df: DataFrame = pd.read_csv('data/engFra.tsv', sep='\t')
    df['Eng'] = (df['Eng'].apply(lambda row: [re.sub(_STRING_CLEAN, '', token) for token in row.split(' ')])
                 .apply(lambda row: ["<SOS>"] + row + ["<EOS>"]))
    df['Fra'] = (df['Fra'].apply(lambda row: [re.sub(_STRING_CLEAN, '', token) for token in row.split(' ')])
                 .apply(lambda row: ["<SOS>"] + row + ["<EOS>"]))

    fra_set: set[str] = set()
    eng_set: set[str] = set()
    for fra_val, eng_val in zip(df['Fra'].tolist(), df['Eng'].tolist()):
        fra_set.update(fra_val)
        eng_set.update(eng_val)

    fra_embedding: Embedding = Embedding(vocab_size=len(fra_set), embedding_size=EMBEDDING_DIM)
    eng_embedding: Embedding = Embedding(vocab_size=len(eng_set), embedding_size=EMBEDDING_DIM)

    fra_embedding.add_mapping(list(fra_set))
    eng_embedding.add_mapping(list(eng_set))

    transformer_train = EncoderDecoderTransformer(
        decoder_layers=DECODER_LAYERS,
        encoder_layers=ENCODER_LAYERS,
        num_heads=NUM_HEADS,
        hidden_layer=HIDDEN_LAYER,
        d_model=D_MODEL,
    )
    linear = Linear(in_dim=EMBEDDING_DIM, out_dim=len(eng_set))
    loss = CrossEntropyLoss()
    losses = []
    epochs = 100
    pbar = tqdm(range(epochs))
    steps = len(df)
    total_steps: int = epochs * steps
    lr_new = LEARNING_RATE
    for epoch in pbar:
        loss_sum = 0.0
        df = df.sample(frac=1).reset_index(drop=True)
        for i, (french_input, english_input) in enumerate(zip(df['Fra'], df['Eng'])):
            targets: np.ndarray = eng_embedding.get_targets(english_input)

            fra_pse = PositionalEncoding(d_model=EMBEDDING_DIM)(
                embeddings=fra_embedding.construct_table(tokens=french_input)
            )
            eng_pse = PositionalEncoding(d_model=EMBEDDING_DIM)(
                embeddings=eng_embedding.construct_table(tokens=english_input)
            )

            output = transformer_train.forward(
                encoder_embeddings=fra_pse,
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

            fra_embedding.backward(
                gradients=gradients_encoder,
                target_indices=fra_embedding.get_list_of_token_ids(french_input),
                learning_rate=lr_new
            )

            eng_embedding.backward(
                gradients=gradients_decoder,
                target_indices=eng_embedding.get_list_of_token_ids(english_input),
                learning_rate=lr_new
            )

            lr_new = ALPHA + 0.5*(LEARNING_RATE - ALPHA)*(1 + np.cos((np.pi * ((steps*epoch)+i)) / total_steps))

            pbar.set_description(f"Epoch {epoch} | Curr Loss: {float(loss_sum / len(df)):.4f} | Learning Rate: {float(lr_new):.20f} | Steps: {int((steps*epoch)+i)}/{int(total_steps)}")

        losses.append(float(loss_sum / len(df)))

        if epoch % 10 == 0:
            translate(transformer=transformer_train, francuski=fra_embedding, engleski=eng_embedding)
            best_loss: int = np.argmin(losses).astype(int)
            first_loss = losses[0]
            curr_loss = losses[-1]
            print(f"Current Best Loss: {losses[best_loss]} | Curr Loss: {curr_loss} | {f"Dropped {((first_loss-curr_loss)/first_loss)*100:.2f}%"}")

    print(losses)
