"""Collect every Hindi or English result into one comparison.json, read by the website, report and README."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from text_utils import project_root
from train_ensemble_hindi import checkpoint_path

ROOT = project_root()
NAMES = dict(transformer='Transformer', bilstm='Bi-LSTM', bilstm_v2='Bi-LSTM v2', bilstm_v3='Bi-LSTM v3', wordchar_cnn='Word-character CNN',
             cnn='Gated CNN', cnn_v2='Gated CNN v2', gru='GRU', lstm='LSTM')


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def row(key, name, metrics, parameters, validation, clean):
    fields = 'examples', 'top1_accuracy', 'top3_accuracy', 'top5_accuracy', 'vocabulary_perplexity', 'target_oov_rate'
    return dict(key=key, name=name, parameters=parameters, validation_top1=validation, without_loops=clean,
                **{field: metrics.get(field) for field in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=['hindi', 'english', 'telugu'], required=True)
    args = parser.parse_args()
    families = ROOT / 'outputs' / f'{args.language}_families'
    baseline = ROOT / 'outputs' / f'{args.language}_comparison'
    selection = read(families / 'ensemble_selection.json')
    validation = {name: score for family in selection['family_validation_top1'].values() for name, score in family.items()}
    rows = []
    for architecture, name in NAMES.items():
        if architecture not in validation:
            continue
        path = checkpoint_path(args.language, architecture)
        checkpoint = torch.load(path, weights_only=True, map_location='cpu')
        metrics = read(path.parent / f"{checkpoint['architecture']}_test_metrics.json")
        parameters = metrics.get('parameter_count') or sum(v.numel() for v in checkpoint['model_state_dict'].values())
        clean = metrics.get('without_loops') or (read(baseline / f'{architecture}_test_without_loops.json') or {}).get('without_loops')
        rows.append(row(architecture, name, metrics, parameters, validation[architecture], clean))
    tested = read(families / 'ensemble_test_metrics.json')
    if tested:
        sizes = {r['key']: r['parameters'] for r in rows}
        used = sum(sizes[name] for name, weight in selection['weights'].items() if name != 'counts' and weight > 0)
        for key, name, parameters in (('ensemble', 'Combined' + (' (all families)' if args.language == 'telugu' else ''), used),
                                     ('hybrid', 'Hybrid' + (f" ({NAMES[selection['hybrid_neural']]} + counts)" if args.language == 'telugu' else ''), sizes[selection['hybrid_neural']])):
            rows.append(row(key, name, tested[key], parameters, tested['validation_top1'][key], tested[key].get('without_loops')))
    rows.sort(key=lambda r: r['validation_top1'], reverse=True)
    pretrained = ROOT / 'outputs' / f'{args.language}_causal_lm'
    sample = []
    if pretrained.exists():
        project = read(pretrained / 'project_models_on_sample.json') or {}
        sample += [dict(name={'combined': 'Combined'}.get(key, f"Best neural ({NAMES.get(key.replace('best_neural_', ''), key)})"),
                        **value) for key, value in project.items()]
        for folder in sorted(p for p in pretrained.iterdir() if p.is_dir()):
            for file, label in (('finetuned_metrics.json', 'LoRA fine-tuned'), ('zeroshot_metrics.json', 'no fine-tuning')):
                metrics = read(folder / file)
                if metrics:
                    sample.append(dict(name=f'{folder.name}, {label}', **metrics))
    result = dict(language=args.language, best=rows[0]['key'], best_by='validation Top-1', models=rows, sample=sample,
                  weights=selection['weights'], hybrid_alpha=selection['hybrid_alpha'], hybrid_neural=selection['hybrid_neural'])
    (families / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for r in rows:
        print(r['name'], '%.2f %.2f %.2f %.2f' % tuple(100 * r[k] for k in ('validation_top1', 'top1_accuracy', 'top3_accuracy', 'top5_accuracy')),
              r['vocabulary_perplexity'], r['parameters'])


if __name__ == '__main__':
    main()
