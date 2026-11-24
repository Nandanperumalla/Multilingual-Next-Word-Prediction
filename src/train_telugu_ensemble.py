"""Blend LSTM, Bi-LSTM v2, Bi-LSTM v3 and long-context counts; pick weights on validation only."""
import itertools
import json
import pickle
from pathlib import Path
import torch
from model import model_from_checkpoint
from text_utils import project_root
from train_telugu_comparison import load_split, save_json
from train_telugu_hybrid_long import long_distribution

ROOT = project_root()
DATA = ROOT / 'data_rebuilt'
OUT = ROOT / 'outputs' / 'telugu_ensemble'
CHECKPOINTS = {
    'lstm': ROOT / 'outputs/telugu_comparison/lstm_best.pt',
    'bilstm_v2': ROOT / 'outputs/telugu_accuracy_v2/bilstm_best.pt',
    'bilstm_v3': ROOT / 'outputs/telugu_accuracy_v3/bilstm_best.pt',
}
HYBRID_V2 = (0.0, 0.5, 0.0, 0.5)


def candidates():
    steps = [i / 10 for i in range(11)]
    grid = {w for w in itertools.product(steps, repeat=4) if abs(sum(w) - 1) < 1e-9
            and w[0] <= 0.3 and 0.2 <= w[3] <= 0.6}
    return sorted(grid | {HYBRID_V2})


def score(split, weights, models, artifact):
    data = load_split(DATA / split if split != 'train' else DATA)
    hits = {w: [0, 0, 0] for w in weights}
    with torch.inference_mode():
        for offset in range(0, len(data), 512):
            batch = data[offset:offset + 512]
            neural = [model(batch[:, :-1]).softmax(-1) for model in models]
            counts = torch.zeros_like(neural[0])
            for i, row in enumerate(batch[:, :-1].tolist()):
                indices, values = zip(*long_distribution(row, artifact['counts'], artifact['unigram']))
                counts[i, list(indices)] = torch.tensor(values, dtype=counts.dtype)
            for w in weights:
                blend = sum(a * p for a, p in zip(w, neural + [counts]))
                blend[:, :2] = -torch.inf
                matches = blend.topk(5, dim=1).indices.eq(batch[:, -1, None])
                hits[w][0] += matches[:, 0].sum().item()
                hits[w][1] += matches[:, :3].any(1).sum().item()
                hits[w][2] += matches.any(1).sum().item()
            if offset % 25600 == 0:
                print(json.dumps(dict(split=split, examples=offset, total=len(data))), flush=True)
    return {w: dict(examples=len(data), top1_accuracy=v[0] / len(data), top3_accuracy=v[1] / len(data),
                    top5_accuracy=v[2] / len(data)) for w, v in hits.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    checkpoints = {k: torch.load(p, weights_only=True, map_location='cpu') for k, p in CHECKPOINTS.items()}
    vocabularies = [c['word_to_idx'] for c in checkpoints.values()]
    if any(v != vocabularies[0] for v in vocabularies):
        raise ValueError('Models use different vocabularies')
    models = [model_from_checkpoint(c).eval() for c in checkpoints.values()]
    with (ROOT / 'outputs/telugu_hybrid_v2/counts.pkl').open('rb') as f:  # Written by train_telugu_hybrid_long.py.
        artifact = pickle.load(f)
    grid = candidates()
    print('candidates', len(grid), flush=True)
    validation = score('validation', grid, models, artifact)
    best = max(grid, key=lambda w: validation[w]['top1_accuracy'])
    names = list(CHECKPOINTS) + ['counts']
    save_json(OUT / 'selection.json', dict(weights=dict(zip(names, best)), selection='validation_top1',
                                           validation=validation[best], hybrid_v2_validation=validation[HYBRID_V2],
                                           candidates=len(grid)))
    tested = score('test', [best, HYBRID_V2], models, artifact)
    save_json(OUT / 'test_metrics.json', dict(ensemble=tested[best], hybrid_v2=tested[HYBRID_V2], weights=dict(zip(names, best))))
    print(json.dumps(dict(weights=dict(zip(names, best)), ensemble=tested[best], hybrid_v2=tested[HYBRID_V2]), indent=2), flush=True)


if __name__ == '__main__':
    main()
