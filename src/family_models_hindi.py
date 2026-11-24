"""Model families used beyond LSTM/Bi-LSTM for Hindi and English: GRU and a small Transformer."""
import torch
from torch import nn
from model import model_from_checkpoint


class PrefixGRU(nn.Module):
    """Same shape as TeluguLSTM, with a GRU cell."""

    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=1, dropout=0.0, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        output, _ = self.gru(self.embedding(input_ids))
        return self.fc(output[:, -1])


class PrefixTransformer(nn.Module):
    """Pre-norm causal Transformer that predicts from the last position.

    Prefixes are left-padded, so the last position is always a real word. Padding keeps its zero
    embedding instead of a key-padding mask, which would leave padded positions nothing to attend to.
    """

    def __init__(self, vocab_size, embedding_dim=128, num_layers=2, num_heads=4, feedforward_dim=512,
                 dropout=0.1, seq_length=8, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.positions = nn.Embedding(seq_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, feedforward_dim, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.register_buffer('causal_mask', torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool), 1))
        self.register_buffer('position_ids', torch.arange(seq_length))

    def forward(self, input_ids):
        state = self.dropout(self.embedding(input_ids) + self.positions(self.position_ids))
        return self.fc(self.norm(self.encoder(state, mask=self.causal_mask))[:, -1])


def load_family(checkpoint):
    """Rebuild any trained model from its checkpoint."""
    architecture = checkpoint['architecture']
    if architecture == 'gru':
        model = PrefixGRU(len(checkpoint['word_to_idx']), checkpoint['embedding_dim'], checkpoint['hidden_dim'],
                          checkpoint['num_layers'], checkpoint['dropout'])
    elif architecture == 'transformer':
        model = PrefixTransformer(len(checkpoint['word_to_idx']), checkpoint['embedding_dim'], checkpoint['num_layers'],
                                  checkpoint['num_heads'], checkpoint['feedforward_dim'], checkpoint['dropout'],
                                  checkpoint['seq_length'])
    else:
        return model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
