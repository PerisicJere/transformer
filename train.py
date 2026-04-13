import re
from typing import Final

import cupy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from config import (
    EMBEDDING_DIM,
    LEARNING_RATE,
    DECODER_LAYERS,
    ENCODER_LAYERS,
    NUM_HEADS,
    HIDDEN_LAYER,
    D_MODEL,
    BATCH_SIZE,
)
from model.cross_entropy_loss import CrossEntropyLoss
from model.embedding import Embedding
from model.encoder_decoder_transformer import EncoderDecoderTransformer
from model.linear import Linear
from model.positional_encoding import PositionalEncoding
from model.softmax import softmax

_STRING_CLEAN: Final = re.compile(r"[^\w\s]")


def prepare_batched_input(sentences_to_batch: np.ndarray) -> list[list[str]]:
    max_len = max(len(sentence) for sentence in sentences_to_batch)
    sentences_to_batch: list[list[str]] = [
        sentence + ["<PAD>"] * (max_len - len(sentence))
        for sentence in sentences_to_batch
    ]
    return sentences_to_batch


def translate(
    transformer: EncoderDecoderTransformer, francuski: Embedding, engleski: Embedding
):
    sentence = [
        ["<SOS>", "I", "am", "student", "<EOS>"],
        ["<SOS>", "Are", "you", "going", "to", "school", "<EOS>"],
    ]
    sentence_eng = prepare_batched_input(sentence)
    postitional_encoding = PositionalEncoding(d_model=EMBEDDING_DIM)
    eng_pse = postitional_encoding(
        embeddings=engleski.construct_table(batch_sentences=sentence_eng)
    )
    to_translate = [["<SOS>"], ["<SOS>"]]
    while True:
        sentence_to_translate = prepare_batched_input(to_translate)
        fra_pse = postitional_encoding(
            embeddings=francuski.construct_table(batch_sentences=sentence_to_translate)
        )
        output = transformer.translate(encoder_input=eng_pse, decoder_input=fra_pse)
        lin = linear(output)
        probs = softmax(input=lin)

        done = True
        for idx in range(len(to_translate)):
            token_id = np.argmax(probs[idx, -1])
            translated = francuski.get_embedding_key(token_id.astype(int))
            to_translate[idx].append(translated)
            if translated != "<EOS>" and len(to_translate[idx]) <= 50:
                done = False

        if done:
            break

    for i in range(len(sentence)):
        print(f"English Sentence: {' '.join(sentence[i])}")
        print(f"French Translation: {' '.join(to_translate[i])}")


if __name__ == "__main__":
    df: DataFrame = pd.read_csv("data/engFra.tsv", sep="\t")
    df["Eng"] = (
        df["Eng"]
        .apply(
            lambda row: [re.sub(_STRING_CLEAN, "", token) for token in row.split(" ")]
        )
        .apply(lambda row: ["<SOS>"] + row + ["<EOS>"])
    )
    df["Fra"] = (
        df["Fra"]
        .apply(
            lambda row: [re.sub(_STRING_CLEAN, "", token) for token in row.split(" ")]
        )
        .apply(lambda row: ["<SOS>"] + row + ["<EOS>"])
    )

    df = df[df["Eng"].apply(len) <= 30]
    df = df[df["Fra"].apply(len) <= 30]
    df = df.reset_index(drop=True)
    fra_set: set[str] = set()
    eng_set: set[str] = set()
    for fra_val, eng_val in zip(df["Fra"].tolist(), df["Eng"].tolist()):
        fra_set.update(fra_val)
        eng_set.update(eng_val)
    fra_set.update(["<PAD>"])
    eng_set.update(["<PAD>"])

    fra_embedding: Embedding = Embedding(
        vocab_size=len(fra_set), embedding_size=EMBEDDING_DIM
    )
    eng_embedding: Embedding = Embedding(
        vocab_size=len(eng_set), embedding_size=EMBEDDING_DIM
    )

    fra_embedding.add_mapping(list(fra_set))
    eng_embedding.add_mapping(list(eng_set))

    transformer_train = EncoderDecoderTransformer(
        decoder_layers=DECODER_LAYERS,
        encoder_layers=ENCODER_LAYERS,
        num_heads=NUM_HEADS,
        hidden_layer=HIDDEN_LAYER,
        d_model=D_MODEL,
    )
    linear = Linear(in_dim=EMBEDDING_DIM, out_dim=len(fra_set))
    fra_pos_enc = PositionalEncoding(d_model=EMBEDDING_DIM)
    eng_pos_enc = PositionalEncoding(d_model=EMBEDDING_DIM)

    loss = CrossEntropyLoss()
    losses = []
    epochs, warmup_steps = 78, 4000
    pbar = tqdm(range(epochs))
    steps = len(df)
    total_steps: int = epochs * steps
    lr_new = LEARNING_RATE
    for epoch in pbar:
        loss_sum = 0.0
        df = df.sample(frac=1).reset_index(drop=True)
        fra_df = df["Fra"]
        eng_df = df["Eng"]
        for i in range(0, len(fra_df), BATCH_SIZE):
            step = ((steps * epoch) + i) + 1

            english_input = prepare_batched_input(eng_df[i : i + BATCH_SIZE].tolist())
            french_input = prepare_batched_input(fra_df[i : i + BATCH_SIZE].tolist())
            french_decoder_input = [sentence[:-1] for sentence in french_input]
            french_target = [sentence[1:] for sentence in french_input]
            # not using because it uses a lot of memory
            # targets: np.ndarray = eng_embedding.get_targets(english_input)
            np.get_default_memory_pool().free_all_blocks()
            np.get_default_pinned_memory_pool().free_all_blocks()
            fra_pse = fra_pos_enc(
                embeddings=fra_embedding.construct_table(
                    batch_sentences=french_decoder_input
                )
            )
            eng_pse = eng_pos_enc(
                embeddings=eng_embedding.construct_table(batch_sentences=english_input)
            )

            output = transformer_train.forward(
                encoder_embeddings=eng_pse,
                decoder_embeddings=fra_pse,
            ).astype(np.float32)
            lin = linear(output).astype(np.float32)
            probs = softmax(input=lin)
            del lin
            loss_value = loss.compute(
                targets=fra_embedding.get_list_of_token_ids(french_target),
                probabilities=probs,
            )
            loss_sum += loss_value
            del loss_value
            target_ids = fra_embedding.get_list_of_token_ids(french_target)
            grad = probs.copy()
            batch_idx = np.arange(grad.shape[0])[:, None]
            seq_idx = np.arange(grad.shape[1])[None, :]
            grad[batch_idx, seq_idx, target_ids] -= 1.0
            pad_id = fra_embedding.mappings["<PAD>"]
            pad_mask = (target_ids != pad_id).astype(np.float32)
            grad *= pad_mask[:, :, None]
            grad /= probs.shape[0] * probs.shape[1]
            del probs
            gradients_decoder, gradients_encoder = transformer_train.backward(
                d_decoder=linear.backward(
                    grad.astype(np.float32), learning_rate=lr_new
                ),
                learning_rate=lr_new,
            )
            gradients_decoder = gradients_decoder.astype(np.float32)
            gradients_encoder = gradients_encoder.astype(np.float32)

            fra_embedding.backward(
                gradients=gradients_decoder,
                target_indices=fra_embedding.get_list_of_token_ids(
                    french_decoder_input
                ),
                learning_rate=lr_new,
            )

            eng_embedding.backward(
                gradients=gradients_encoder,
                target_indices=eng_embedding.get_list_of_token_ids(english_input),
                learning_rate=lr_new,
            )
            np.get_default_memory_pool().free_all_blocks()
            np.get_default_pinned_memory_pool().free_all_blocks()
            lr_new = D_MODEL ** (-0.5) * min(
                step ** (-0.5), step * (warmup_steps ** (-1.5))
            )

            pbar.set_description(
                f"Epoch {epoch} | Curr Loss: {float(loss_sum / len(df)):.4f} | Learning Rate: {float(lr_new):.20f} | Steps: {step}/{int(total_steps)}"
            )

        losses.append(float(loss_sum / len(df)))

        if epoch % 7 == 0:
            translate(
                transformer=transformer_train,
                francuski=fra_embedding,
                engleski=eng_embedding,
            )
            best_loss = min(losses)
            first_loss = losses[0]
            curr_loss = losses[-1]
            print(
                f"Current Best Loss: {best_loss} | Curr Loss: {curr_loss} | {f'Dropped {((first_loss - curr_loss) / first_loss) * 100:.2f}%'}"
            )

    print(losses)
