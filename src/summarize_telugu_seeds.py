"""Summarize repeated baseline runs as mean, sample standard deviation and range."""
import json
import statistics
from text_utils import project_root

ROOT = project_root() / 'outputs'
SEEDS = ROOT / 'telugu_seeds'
MODELS = {
    'LSTM': ('lstm', ROOT / 'telugu_comparison', '{}'),
    'Bi-LSTM': ('bilstm', ROOT / 'telugu_comparison', '{}'),
    'Word CNN': ('cnn', ROOT / 'telugu_cnn', '{}_cnn'),
    'Gated word-char CNN': ('wordchar_cnn', ROOT / 'telugu_wordchar', '{}_wordchar'),
}
METRICS = {'Top-1': 'top1_accuracy', 'Top-5': 'top5_accuracy', 'Perplexity': 'vocabulary_perplexity'}


def runs(architecture, original, suffix):
    found = {42: original / f'{architecture}_test_metrics.json'}
    for seed in (1, 2, 3, 4):
        found[seed] = SEEDS / suffix.format(f'seed_{seed}') / f'{architecture}_test_metrics.json'
    return {seed: json.loads(path.read_text(encoding='utf-8')) for seed, path in found.items() if path.exists()}


def main():
    summary = {}
    for name, (architecture, original, suffix) in MODELS.items():
        results = runs(architecture, original, suffix)
        summary[name] = dict(seeds=sorted(results))
        print(f'\n{name}: seeds {sorted(results)}')
        for label, key in METRICS.items():
            scale = 1 if label == 'Perplexity' else 100
            values = [metrics[key] * scale for metrics in results.values()]
            spread = statistics.stdev(values) if len(values) > 1 else 0.
            summary[name][label] = dict(mean=statistics.mean(values), std=spread, min=min(values), max=max(values))
            print(f'  {label}: {statistics.mean(values):.2f} ± {spread:.2f} (range {min(values):.2f}–{max(values):.2f})')
    (SEEDS / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
