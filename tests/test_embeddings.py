from model.embedding import Embedding


def test_embeddings():
    tokens: set[str] = {'Transformer', 'Attention', 'FeedForward', 'Transformer'}
    embed = Embedding(vocab_size=len(tokens), embedding_size=4)

    for token in tokens:
        embed.add_mapping(token)

    assert len(embed.embedding_weights) == len(embed.mappings)

    for token in tokens:
        embed.add_mapping(token)

    # Vector for Attention
    assert len(embed.get_embedding_vector("Attention")) == 4
