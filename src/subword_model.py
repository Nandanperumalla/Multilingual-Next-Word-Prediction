"""Bi-LSTM whose word vectors also include hashed character n-gram (subword) vectors."""
import zlib
import torch
from torch import nn

MAX_SUBWORDS = 24


def subwords(word, buckets):
    """Hash character 3- and 4-grams into [1, buckets); 0 is padding."""
    marked = f'<{word}>'
    grams = [marked[i:i + size] for size in (3, 4) for i in range(len(marked) - size + 1)]
    return [1 + zlib.crc32(gram.encode('utf-8')) % (buckets - 1) for gram in grams[:MAX_SUBWORDS]]


class TeluguSubwordBiLSTM(nn.Module):
    """Ids at or above vocab_size are context-only words: no word vector, subwords only."""

    def __init__(self, vocab_size, subword_table, buckets=20000, embedding_dim=256,
                 hidden_dim=512, num_layers=2, dropout=0.3, word_dropout=0.1):
        super().__init__()
        self.vocab_size, self.word_dropout = vocab_size, word_dropout
        self.register_buffer('subword_table', subword_table, persistent=True)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.subword_embedding = nn.Embedding(buckets, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_dim * 2, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def embed(self, ids):
        known = ids.where(ids < self.vocab_size, torch.ones_like(ids))
        if self.training and self.word_dropout:
            # Hide some word vectors so rare words learn to rely on their subwords.
            hidden = torch.rand(ids.shape, device=ids.device) < self.word_dropout
            known = known.where(~(hidden & (known > 1)), torch.ones_like(ids))
        pieces = self.subword_table[ids]
        counts = pieces.gt(0).sum(-1, keepdim=True).clamp(min=1)
        return self.word_embedding(known) + self.subword_embedding(pieces).sum(2) / counts

    def forward(self, ids):
        _, (hidden, _) = self.lstm(self.dropout(self.embed(ids)))
        state = self.dropout(self.projection(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        # Output weights are shared with the input word vectors.
        return torch.nn.functional.linear(state, self.word_embedding.weight, self.bias)
