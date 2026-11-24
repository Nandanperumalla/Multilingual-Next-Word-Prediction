"""Score the Hindi checkpoints with and without machine-translation repetition loops.

About 0.4% of Hindi sentences are loops such as one word repeated hundreds of times, and those easy
targets can flatter the score. The headline metrics keep them, so the method matches Telugu; this
reports the same test set with those sentences removed. Nothing here selects or tunes a model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from model import model_from_checkpoint
from preprocess import make_line_sequences
from text_utils import project_root, tokenize
from train_telugu_comparison import evaluate, save_json

ROOT = project_root()


def is_loop(words, run=4):
    """True for a run of identical words or a repeated two-word pattern."""
    if any(len(set(words[i:i + run])) == 1 for i in range(len(words) - run + 1)):
        return True
    return any(words[i:i + 2] == words[i + 2:i + 4] == words[i + 4:i + 6] for i in range(len(words) - 5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data_hindi')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'hindi_comparison')
    args = parser.parse_args()
    kept, dropped, loop_sentences, sentences = [], [], 0, 0
    for architecture in ('lstm', 'bilstm'):
        checkpoint = torch.load(args.output_dir / f'{architecture}_best.pt', weights_only=True, map_location='cpu')
        if isinstance(kept, list):
            lines = (args.data_dir / 'test' / 'hindi_raw.txt').read_text(encoding='utf-8').splitlines()
            for line in lines:
                words = tokenize(line, language='hindi')
                rows = [x + [y] for x, y in make_line_sequences(words, checkpoint['word_to_idx'], checkpoint['seq_length'])]
                sentences += 1
                if is_loop(words):
                    loop_sentences += 1
                    dropped += rows
                else:
                    kept += rows
            kept, dropped = torch.tensor(kept), torch.tensor(dropped)
        model = model_from_checkpoint(checkpoint)
        result = dict(all_examples=evaluate(model, torch.cat([kept, dropped]), 512), without_loops=evaluate(model, kept, 512),
                      loops_only=evaluate(model, dropped, 512))
        print(architecture, {k: round(v['top1_accuracy'], 4) for k, v in result.items()}, flush=True)
        save_json(args.output_dir / f'{architecture}_test_without_loops.json', dict(
            test_sentences=sentences, loop_sentences=loop_sentences, examples_removed=len(dropped), **result))


if __name__ == '__main__':
    main()
