import re
from typing import Final

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from config import EMBEDDING_DIM, LEARNING_RATE
from model.cross_entropy_loss import CrossEntropyLoss
from model.embedding import Embedding
from model.encoder_decoder_transformer import EncoderDecoderTransformer
from model.linear import Linear
from model.positional_encoding import PositionalEncoding
from model.softmax import softmax

_STRING_CLEAN: Final = re.compile(r'[^\w\s]')


if __name__ == '__main__':
    df: DataFrame = pd.read_csv('data/EngCro.tsv', sep='\t')
    df['Eng'] = (df['Eng'].apply(lambda row: [re.sub(_STRING_CLEAN, '', token) for token in row.split(' ')])
                 .apply(lambda row: ["<SOS>"] + row + ["<EOS>"]))
    df['Cro'] = (df['Cro'].apply(lambda row: [re.sub(_STRING_CLEAN, '', token) for token in row.split(' ')])
                 .apply(lambda row: ["<SOS>"] + row + ["<EOS>"]))

    max_eng_len = max(len(row) for row in df['Eng'])
    max_cro_len = max(len(row) for row in df['Cro'])

    df['Eng'] = df['Eng'].apply(lambda row: row + ["<PAD>"] * (max_eng_len - len(row)))
    df['Cro'] = df['Cro'].apply(lambda row: row + ["<PAD>"] * (max_cro_len - len(row)))

    cro_set: set[str] = set()
    eng_set: set[str] = set()
    for cro_val, eng_val in zip(df['Cro'].tolist(), df['Eng'].tolist()):
        cro_set.update(cro_val)
        eng_set.update(eng_val)

    cro_embedding: Embedding = Embedding(vocab_size=len(cro_set), embedding_size=EMBEDDING_DIM)
    eng_embedding: Embedding = Embedding(vocab_size=len(eng_set), embedding_size=EMBEDDING_DIM)

    cro_embedding.add_mapping(list(cro_set))
    eng_embedding.add_mapping(list(eng_set))

    eng_pad_idx = eng_embedding.get_list_of_token_ids(["<PAD>"])[0]
    cro_pad_idx = cro_embedding.get_list_of_token_ids(["<PAD>"])[0]

    transformer_train = EncoderDecoderTransformer(
        decoder_layers=2,
        encoder_layers=2,
        in_dim=EMBEDDING_DIM,
    )
    linear = Linear(in_dim=EMBEDDING_DIM, out_dim=len(eng_set))
    loss = CrossEntropyLoss()
    losses: list[float] = []
    pbar = tqdm(range(10))
    for epoch in pbar:
        loss_sum = 0.0
        for croatian_input, english_input in zip(df['Cro'], df['Eng']):
            targets: np.ndarray = eng_embedding.get_targets(english_input)
            target_idx: np.ndarray = eng_embedding.get_list_of_token_ids(english_input)
            target_pad_mask = np.where(target_idx == eng_pad_idx, -np.inf, 0.0).reshape(1, -1)

            src_idx: np.ndarray = cro_embedding.get_list_of_token_ids(croatian_input)
            src_pad_mask = np.where(src_idx == cro_pad_idx, -np.inf, 0.0).reshape(1, -1)

            cro_pse = PositionalEncoding(d_model=EMBEDDING_DIM)(
                embeddings=cro_embedding.construct_table(tokens=croatian_input)
            )
            eng_pse = PositionalEncoding(d_model=EMBEDDING_DIM)(
                embeddings=eng_embedding.construct_table(tokens=english_input)
            )

            output = transformer_train.forward(
                encoder_embeddings=cro_pse,
                decoder_embeddings=eng_pse,
                src_pad_mask=src_pad_mask,
                target_pad_mask=target_pad_mask
            )
            lin = linear(output)
            probs = softmax(input=lin)

            loss_value = loss.compute(targets=eng_embedding.get_list_of_token_ids(english_input), probabilities=probs)
            loss_sum += loss_value

            gradients_decoder, gradients_encoder = transformer_train.backward(
                d_decoder=linear.backward(probs-targets, learning_rate=LEARNING_RATE),
                learning_rate=LEARNING_RATE
            )

            cro_embedding.backward(
                gradients=gradients_encoder,
                target_indices=cro_embedding.get_list_of_token_ids(croatian_input),
                learning_rate=LEARNING_RATE
            )

            eng_embedding.backward(
                gradients=gradients_decoder,
                target_indices=eng_embedding.get_list_of_token_ids(english_input),
                learning_rate=LEARNING_RATE
            )
            pbar.set_description(f"Epoch {epoch}")

        losses.append(loss_sum / len(df))

