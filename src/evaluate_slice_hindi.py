"""Score the best Hindi or English project models on the 500-example sample used for the pretrained models."""
from __future__ import annotations
import argparse
import json
import pickle
from family_predictor_hindi import CombinedPredictor, FamilyPredictor
from finetune_causal_lm_hindi import sample
from text_utils import configure_utf8_stdio, project_root

ROOT = project_root()


def main():
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=['hindi', 'english'], default='hindi')
    parser.add_argument('--examples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    data_dir = ROOT / f'data_{args.language}'
    vocab = pickle.loads((data_dir / 'vocab.pkl').read_bytes())['word_to_idx']
    rows = sample(data_dir, args.language, args.examples, args.seed)
    selection = json.loads((ROOT / 'outputs' / f'{args.language}_families' / 'ensemble_selection.json').read_text(encoding='utf-8'))
    scores = {name: score for family in selection['family_validation_top1'].values() for name, score in family.items()}
    best = max(scores, key=scores.get)
    results = {}
    for name, model in {f'best_neural_{best}': FamilyPredictor(args.language, best), 'combined': CombinedPredictor(args.language)}.items():
        hits = [0, 0, 0]
        for context, target in rows:
            ranked = [item['word'] for item in model.predict(' '.join(context), 5)['predictions']]
            if target in vocab:
                hits[0] += ranked[:1] == [target]
                hits[1] += target in ranked[:3]
                hits[2] += target in ranked[:5]
        results[name] = dict(examples=len(rows), top1_accuracy=hits[0] / len(rows), top3_accuracy=hits[1] / len(rows),
                             top5_accuracy=hits[2] / len(rows), target_oov_rate=sum(t not in vocab for _, t in rows) / len(rows))
        print(name, results[name], flush=True)
    out = ROOT / 'outputs' / f'{args.language}_causal_lm'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'project_models_on_sample.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
