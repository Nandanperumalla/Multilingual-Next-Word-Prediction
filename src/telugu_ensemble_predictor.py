"""Ensemble Telugu next-word predictions: LSTM, two Bi-LSTMs and long-context counts voting together."""
from __future__ import annotations
import json
import pickle
import torch
from model import model_from_checkpoint
from telugu_neural_predictor import TeluguNeuralPredictor
from text_utils import tokenize, project_root
from train_telugu_hybrid_long import long_distribution

OUTPUTS = project_root() / 'outputs'
CHECKPOINTS = {
    'lstm': OUTPUTS / 'telugu_comparison' / 'lstm_best.pt',
    'bilstm_v2': OUTPUTS / 'telugu_accuracy_v2' / 'bilstm_best.pt',
    'bilstm_v3': OUTPUTS / 'telugu_accuracy_v3' / 'bilstm_best.pt',
}
SELECTION = OUTPUTS / 'telugu_ensemble' / 'selection.json'
COUNTS = OUTPUTS / 'telugu_hybrid_v2' / 'counts.pkl'
FILES = [*CHECKPOINTS.values(), SELECTION, COUNTS]


class TeluguEnsemblePredictor(TeluguNeuralPredictor):
    def __init__(self):
        super().__init__(CHECKPOINTS['bilstm_v2'])
        self.weights = json.loads(SELECTION.read_text(encoding='utf-8'))['weights']  # Selected on validation.
        self.models = {key: self.model if key == 'bilstm_v2' else model_from_checkpoint(
            torch.load(path, weights_only=True, map_location='cpu')).eval() for key, path in CHECKPOINTS.items()}
        with COUNTS.open('rb') as f:  # Written by train_telugu_hybrid_long.py in this project.
            artifact = pickle.load(f)
        self.counts, self.unigram = artifact['counts'], artifact['unigram']

    def predict(self, text, top_k=5):
        words = tokenize(text, language='telugu')[-self.checkpoint['seq_length']:]
        if not words:
            return dict(predictions=[], unknown_words=[], context=[])
        unknown = [word for word in words if word not in self.vocab]
        if len(unknown) == len(words):
            return dict(predictions=[], unknown_words=unknown, context=words)
        ids = [self.vocab.get(word, 1) for word in words]
        ids = [0] * (self.checkpoint['seq_length'] - len(ids)) + ids
        with torch.inference_mode():
            probabilities = sum(self.weights[key] * model(torch.tensor([ids]))[0].softmax(-1)
                                for key, model in self.models.items())
        empirical = torch.zeros_like(probabilities)
        indices, values = zip(*long_distribution(ids, self.counts, self.unigram))
        empirical[list(indices)] = torch.tensor(values, dtype=probabilities.dtype)
        probabilities = probabilities + self.weights['counts'] * empirical
        ranked = probabilities.clone()
        ranked[:2] = -torch.inf
        top = ranked.topk(min(max(1, top_k), len(self.vocab) - 2)).indices
        return dict(predictions=[dict(word=self.checkpoint['idx_to_word'][int(i)],
                                      probability=float(probabilities[i])) for i in top],
                    unknown_words=unknown, context=words)
