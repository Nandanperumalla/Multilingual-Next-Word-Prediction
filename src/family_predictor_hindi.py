"""Website predictions for the Hindi and English families and their combined model."""
from __future__ import annotations
import json
import pickle
import torch
from family_models_hindi import load_family
from text_utils import project_root, tokenize
from train_ensemble_hindi import checkpoint_path
from train_telugu_hybrid_long import long_distribution

OUTPUTS = project_root() / 'outputs'


class FamilyPredictor:
    def __init__(self, language, architecture):
        torch.set_num_threads(2)
        self.language = language
        self.checkpoint = torch.load(checkpoint_path(language, architecture), weights_only=True, map_location='cpu')
        self.model = load_family(self.checkpoint).eval()
        self.vocab = self.checkpoint['word_to_idx']
        self.surfaces = self.checkpoint.get('character_metadata', {}).get('surface_to_idx')

    def context(self, text):
        words = tokenize(text, language=self.language)[-self.checkpoint['seq_length']:]
        unknown = [word for word in words if word not in self.vocab]
        if not words or len(unknown) == len(words):
            return words, unknown, None
        ids = [self.vocab.get(word, 1) for word in words]
        return words, unknown, [0] * (self.checkpoint['seq_length'] - len(ids)) + ids

    def inputs(self, words):
        """Word-character models read spellings; a word never seen in any split falls back to the padding row."""
        if self.surfaces is None:
            ids = [self.vocab.get(word, 1) for word in words]
        else:
            ids = [self.surfaces.get(word, 0) for word in words]
        return torch.tensor([[0] * (self.checkpoint['seq_length'] - len(ids)) + ids])

    def distribution(self, words, ids):
        return self.model(self.inputs(words))[0].softmax(-1)

    def predict(self, text, top_k=5):
        words, unknown, ids = self.context(text)
        if ids is None:
            return dict(predictions=[], unknown_words=unknown, context=words)
        with torch.inference_mode():
            probabilities = self.distribution(words, ids)
        ranked = probabilities.clone()
        ranked[:2] = -torch.inf
        top = ranked.topk(min(max(1, top_k), len(self.vocab) - 2)).indices
        return dict(predictions=[dict(word=self.checkpoint['idx_to_word'][int(i)], probability=float(probabilities[i]))
                                 for i in top], unknown_words=unknown, context=words)


class CombinedPredictor(FamilyPredictor):
    def __init__(self, language):
        out = OUTPUTS / f'{language}_families'
        selection = json.loads((out / 'ensemble_selection.json').read_text(encoding='utf-8'))
        self.weights = selection['weights']
        members = [name for name, weight in self.weights.items() if name != 'counts' and weight > 0]
        super().__init__(language, members[0])
        self.members = {name: FamilyPredictor(language, name) for name in members}
        with (out / 'counts.pkl').open('rb') as f:  # Written by train_ensemble_hindi.py.
            artifact = pickle.load(f)
        self.counts, self.unigram = artifact['counts'], artifact['unigram']

    def distribution(self, words, ids):
        total = sum(self.weights[name] * member.distribution(words, ids) for name, member in self.members.items())
        empirical = torch.zeros_like(total)
        indices, values = zip(*long_distribution(ids, self.counts, self.unigram))
        empirical[list(indices)] = torch.tensor(values, dtype=total.dtype)
        return total + self.weights['counts'] * empirical
