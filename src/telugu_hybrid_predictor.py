"""Hybrid v2 Telugu next-word predictions: Bi-LSTM blended with long-context counts."""
from __future__ import annotations
import argparse
import json
import pickle
import torch
from telugu_neural_predictor import TeluguNeuralPredictor
from text_utils import tokenize, project_root, configure_utf8_stdio
from train_telugu_hybrid_long import long_distribution

OUT = project_root() / 'outputs' / 'telugu_hybrid_v2'
NEURAL = project_root() / 'outputs' / 'telugu_accuracy_v2' / 'bilstm_best.pt'


class TeluguHybridPredictor(TeluguNeuralPredictor):
    def __init__(self, counts_path=OUT / 'counts.pkl', checkpoint_path=NEURAL):
        super().__init__(checkpoint_path)
        with open(counts_path, 'rb') as f:
            artifact = pickle.load(f)  # Written by train_telugu_hybrid_long.py in this project.
        self.counts, self.unigram = artifact['counts'], artifact['unigram']
        selection = json.loads((OUT / 'selection.json').read_text(encoding='utf-8'))
        self.alpha = selection['configuration'][1]  # Selected on validation; 'long' mode.

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
            neural = self.model(torch.tensor([ids]))[0].softmax(-1)
        empirical = torch.zeros_like(neural)
        indices, values = zip(*long_distribution(ids, self.counts, self.unigram))
        empirical[list(indices)] = torch.tensor(values, dtype=neural.dtype)
        probabilities = (1 - self.alpha) * neural + self.alpha * empirical
        ranked = probabilities.clone()
        ranked[:2] = -torch.inf
        top = ranked.topk(min(max(1, top_k), len(self.vocab) - 2)).indices
        return dict(predictions=[dict(word=self.checkpoint['idx_to_word'][int(i)],
                                      probability=float(probabilities[i])) for i in top],
                    unknown_words=unknown, context=words)


def main():
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', required=True)
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(TeluguHybridPredictor().predict(args.text, args.top_k), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
