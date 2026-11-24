"""Blend the best model of each family with n-gram counts; weights are chosen on validation only.

The best recurrent, CNN and Transformer models are picked by validation Top-1. A Hybrid (best recurrent
model plus counts) is selected the same way. The test split is scored once, after selection.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
import torch
from family_models_hindi import load_family
from text_utils import project_root
from train_families_hindi import loop_mask
from train_telugu_comparison import load_split, save_json
from train_telugu_hybrid_long import long_distribution

ROOT = project_root()
FAMILIES = dict(recurrent=['lstm', 'bilstm', 'gru'], cnn=['cnn', 'wordchar_cnn'], transformer=['transformer'])
HYBRID_ALPHAS = (0.1, 0.25, 0.5)
# Telugu reuses its earlier checkpoints read-only; their folders are never written.
TELUGU = dict(bilstm_v2='telugu_accuracy_v2/bilstm_best.pt', bilstm_v3='telugu_accuracy_v3/bilstm_best.pt',
              cnn='telugu_cnn/cnn_best.pt', cnn_v2='telugu_cnn_accuracy_v2/cnn_best.pt',
              wordchar_cnn='telugu_wordchar/wordchar_cnn_best.pt')


def data_dir(language):
    return ROOT / 'data_rebuilt' if language == 'telugu' else ROOT / f'data_{language}'


def families(language):
    if language != 'telugu':
        return FAMILIES
    return dict(recurrent=['lstm', 'bilstm', 'bilstm_v2', 'bilstm_v3', 'gru'], cnn=['cnn', 'cnn_v2', 'wordchar_cnn'],
                transformer=['transformer'])


def checkpoint_path(language, architecture):
    new = ROOT / 'outputs' / f'{language}_families' / f'{architecture}_best.pt'
    if new.exists():
        return new
    if language == 'telugu' and architecture in TELUGU:
        return ROOT / 'outputs' / TELUGU[architecture]
    return ROOT / 'outputs' / f'{language}_comparison' / f'{architecture}_best.pt'


def validation_top1(path):
    checkpoint = torch.load(path, weights_only=True, map_location='cpu')
    record = next((h for h in checkpoint.get('history', []) if h['epoch'] == checkpoint['epoch'] and h.get('validation')), None)
    for name in (f"{checkpoint['architecture']}_history.json", 'history.json'):  # Saved before this epoch was appended.
        if record is None and (path.parent / name).exists():
            history = json.loads((path.parent / name).read_text())
            history = history['history'] if isinstance(history, dict) else history
            record = next((h for h in history if h['epoch'] == checkpoint['epoch']), None)
    return record['validation']['top1_accuracy']


def build_counts(train, path):
    if path.exists():
        with path.open('rb') as f:  # Written by this script.
            return pickle.load(f)
    counts = {n: defaultdict(Counter) for n in (1, 2, 3, 4)}
    unigram = Counter()
    for row in train.numpy():
        target = int(row[-1])
        unigram[target] += 1
        for size in (1, 2, 3, 4):
            counts[size][tuple(map(int, row[-size - 1:-1]))][target] += 1
    artifact = dict(counts={size: {key: dict(c) for key, c in table.items() if sum(c.values()) >= 3}
                            for size, table in counts.items()},
                    unigram=[(k, v / len(train)) for k, v in unigram.items()], minimum_context_count=3)
    with path.open('wb') as f:
        pickle.dump(artifact, f)
    return artifact


def surface_rows(language, data, checkpoint):
    """Rebuild the word-character inputs, which are aligned with the shared sequence order."""
    from wordchar_data_hindi import prepare
    vocab = pickle.loads((data / 'vocab.pkl').read_bytes())
    (_, validation, test), _, _, metadata = prepare(data, vocab, language, f'{language}_raw.txt')
    saved = checkpoint['character_metadata']
    if any(metadata[key] != saved[key] for key in ('char_to_idx', 'surface_to_idx')):
        raise ValueError('Rebuilt spellings differ from the checkpoint')
    return dict(validation=validation, test=test)


class Scorer:
    def __init__(self, names, models, inputs, artifact):
        self.names, self.models, self.inputs, self.artifact = names, models, inputs, artifact

    @torch.inference_mode()
    def run(self, split, data, weights, limit=None, keep=None):
        """Score each weight vector; weights cover the neural members then the counts."""
        end = min(limit or len(data), len(data))
        parts = {w: dict(all=[0, 0, 0, 0.0, 0], kept=[0, 0, 0, 0.0, 0]) for w in weights}
        known = 0
        for offset in range(0, end, 512):
            batch = data[offset:min(offset + 512, end)]
            targets = batch[:, -1]
            known += targets.ne(1).sum().item()
            neural = [model(self.inputs.get(name, {}).get(split, data)[offset:offset + len(batch), :-1]).softmax(-1)
                      for name, model in zip(self.names, self.models)]
            counts = torch.zeros_like(neural[0])
            for i, row in enumerate(batch[:, :-1].tolist()):
                indices, values = zip(*long_distribution(row, self.artifact['counts'], self.artifact['unigram']))
                counts[i, list(indices)] = torch.tensor(values, dtype=counts.dtype)
            mask = keep[offset:offset + len(batch)] if keep is not None else None
            for w in weights:
                blend = sum(a * p for a, p in zip(w, neural + [counts]) if a)
                loss = -blend.gather(1, targets[:, None]).clamp_min(1e-12).log()[:, 0]
                blend[:, :2] = -torch.inf
                matches = blend.topk(5, dim=1).indices.eq(targets[:, None])
                groups = [('all', None)] + ([('kept', mask)] if mask is not None else [])
                for key, selector in groups:
                    m, l = (matches, loss) if selector is None else (matches[selector], loss[selector])
                    total = parts[w][key]
                    total[0] += m[:, 0].sum().item()
                    total[1] += m[:, :3].any(1).sum().item()
                    total[2] += m.any(1).sum().item()
                    total[3] += l.sum().item()
                    total[4] += len(l)
            if offset % 25600 == 0:
                print(json.dumps(dict(split=split, examples=offset, total=end)), flush=True)

        def metrics(v):
            return dict(examples=v[4], top1_accuracy=v[0] / v[4], top3_accuracy=v[1] / v[4], top5_accuracy=v[2] / v[4],
                        cross_entropy=v[3] / v[4], vocabulary_perplexity=math.exp(v[3] / v[4]))
        result = {}
        for w, value in parts.items():
            result[w] = metrics(value['all'])
            result[w]['target_oov_rate'] = 1 - known / result[w]['examples']
            if keep is not None:
                result[w]['without_loops'] = metrics(value['kept'])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=['hindi', 'english', 'telugu'], default='hindi')
    parser.add_argument('--screen-examples', type=int, default=60000)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data = data_dir(args.language)
    out = ROOT / 'outputs' / f'{args.language}_families'
    choices, scores = {}, {}
    for family, architectures in families(args.language).items():
        found = {a: checkpoint_path(args.language, a) for a in architectures}
        found = {a: p for a, p in found.items() if p.exists()}
        scores[family] = {a: validation_top1(p) for a, p in found.items()}
        choices[family] = max(found, key=lambda a: scores[family][a])
    print(json.dumps(dict(choices=choices, validation_top1=scores)), flush=True)
    names = list(choices.values())
    checkpoints = {a: torch.load(checkpoint_path(args.language, a), weights_only=True, map_location='cpu') for a in names}
    models = [load_family(checkpoints[a]).eval() for a in names]
    artifact = build_counts(load_split(data), out / 'counts.pkl')
    inputs = {a: surface_rows(args.language, data, checkpoints[a]) for a in names if a == 'wordchar_cnn'}
    scorer = Scorer(names, models, inputs, artifact)
    validation = load_split(data / 'validation')
    steps = [i / 10 for i in range(11)]
    grid = [w for w in itertools.product(steps, repeat=len(names) + 1) if abs(sum(w) - 1) < 1e-9]
    hybrids = [(1 - a, *[0.0] * (len(names) - 1), a) for a in HYBRID_ALPHAS]  # names[0] is the recurrent choice.
    grid = sorted(set(grid) | set(hybrids))
    screened = scorer.run('validation', validation, grid, limit=args.screen_examples)
    ranked = sorted(grid, key=lambda w: (screened[w]['top1_accuracy'], screened[w]['top5_accuracy']), reverse=True)
    finalists = sorted(set(ranked[:10]) | set(hybrids))
    full = scorer.run('validation', validation, finalists)
    best = max(finalists, key=lambda w: (full[w]['top1_accuracy'], full[w]['top5_accuracy']))
    hybrid = max(hybrids, key=lambda w: full[w]['top1_accuracy'])
    labels = names + ['counts']
    save_json(out / 'ensemble_selection.json', dict(
        family_choices=choices, family_validation_top1=scores, weights=dict(zip(labels, best)),
        hybrid_alpha=hybrid[-1], hybrid_neural=names[0], selection='validation_top1', candidates=len(grid),
        screened_on=args.screen_examples, validation=full[best], hybrid_validation=full[hybrid]))
    test = load_split(data / 'test')
    keep = loop_mask(data, args.language, f'{args.language}_raw.txt')
    tested = scorer.run('test', test, [best, hybrid], keep=keep)
    save_json(out / 'ensemble_test_metrics.json', dict(ensemble=tested[best], hybrid=tested[hybrid], weights=dict(zip(labels, best)),
              validation_top1=dict(ensemble=full[best]['top1_accuracy'], hybrid=full[hybrid]['top1_accuracy']),
              neural_parameter_count=sum(p.numel() for m in models for p in m.parameters())))
    print(json.dumps(dict(weights=dict(zip(labels, best)), ensemble=tested[best]), indent=2), flush=True)


if __name__ == '__main__':
    main()
