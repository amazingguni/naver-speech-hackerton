from __future__ import absolute_import, division, print_function, unicode_literals
from torch import nn
from transformer_model.embedding.positional_encoding import PositionalEmbedding
from transformer_model.embedding.token_embedding import TokenEmbedding


class Embeddings(nn.Module):
    def __init__(self, d_model, maxlen, vocab_size, pad_id, is_word):
        super(Embeddings, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, embed_size=d_model, pad_id=pad_id)
        self.pos_embedding = PositionalEmbedding(d_model = d_model, max_len=maxlen)
        self.is_word= is_word

    def forward(self, x):
        if self.is_word:
            x = self.token_embedding(x)

        pos_embed = self.pos_embedding(x)
        return x + pos_embed

def main():
    print("Embeddings")

if __name__ == '__main__':
    main()