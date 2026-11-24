"""Score the project's own models on the same test slice used for the pretrained causal LM."""
import argparse
import json
import pickle
from pathlib import Path
from finetune_telugu_causal_lm import test_examples
from telugu_hybrid_predictor import TeluguHybridPredictor
from telugu_neural_predictor import TeluguNeuralPredictor
from text_utils import project_root, configure_utf8_stdio

ROOT = project_root()


def main():
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data_rebuilt')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'telugu_causal_lm')
    parser.add_argument('--examples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    with (args.data_dir / 'vocab.pkl').open('rb') as f:  # Written locally by prepare_telugu_llm_data.py.
        vocab = pickle.load(f)['word_to_idx']
    rows = test_examples(args, vocab)
    models = {'hybrid_v2': TeluguHybridPredictor(),
              'bilstm_v2': TeluguNeuralPredictor(ROOT / 'outputs' / 'telugu_accuracy_v2' / 'bilstm_best.pt')}
    results = {}
    for name, model in models.items():
        hits1 = hits5 = 0
        for context, target in rows:
            ranked = [p['word'] for p in model.predict(' '.join(context), 5)['predictions']]
            if target in vocab:
                hits1 += ranked[:1] == [target]
                hits5 += target in ranked
        results[name] = dict(examples=len(rows), top1_accuracy=hits1 / len(rows), top5_accuracy=hits5 / len(rows))
        print(name, results[name], flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'project_models_on_slice.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
