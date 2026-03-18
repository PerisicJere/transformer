from model.embedding import Embedding


def test_embeddings():
    tokens: list[str] = ['Transformer', 'Attention', 'FeedForward', 'Transformer']
    embed = Embedding(vocab_size=len(set(tokens)), embedding_size=4)

    embed.add_mapping(tokens)

    assert len(embed.embedding_weights) == len(embed.mappings)

    # Vector for Attention
    assert len(embed.get_embedding_vector('Transformer')) == 4
