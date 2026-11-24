"""LoRA fine-tune and score a pretrained causal LM for Hindi or English on the fixed 500-example test sample.

Follows finetune_telugu_causal_lm.py: targets outside the project vocabulary count as misses, and whole-word
rankings come from greedily completing the top word-start tokens, so they approximate word probabilities.
"""
from __future__ import annotations
import argparse
import json
import pickle
import random
import time
from pathlib import Path
import torch
import finetune_telugu_causal_lm as base
from text_utils import configure_utf8_stdio, project_root, tokenize

ROOT = project_root()


def script_check(language):
    if language == 'hindi':
        return lambda text: bool(text) and all('ऀ' <= ch <= 'ॿ' for ch in text)
    # Project English text is lower-case, so capitalized candidates could never match a target.
    return lambda text: bool(text) and text.replace("'", '').isalpha() and text.isascii() and text.islower()


def use_model(name):
    """GPT-2 has no pad token and different layer names, so adapt the shared helpers without editing them."""
    if 'gpt2' not in name:
        return
    import peft
    load, config = base.load_model, peft.LoraConfig

    def load_with_padding(*args, **kwargs):
        tokenizer, model = load(*args, **kwargs)
        tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, 'right'
        return tokenizer, model

    base.load_model = load_with_padding
    peft.LoraConfig = lambda **options: config(**{**options, 'target_modules': ['c_attn', 'c_proj'], 'fan_in_fan_out': True})


def sample(data_dir, language, examples, seed):
    rows = []
    for sentence in base.read_sentences(data_dir / 'test' / f'{language}_raw.txt'):
        words = tokenize(sentence, language=language)
        rows += [(words[max(0, i - 8):i], words[i]) for i in range(1, len(words))]
    random.Random(seed).shuffle(rows)
    return rows[:examples]


def main():
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'eval'])
    parser.add_argument('--language', choices=['hindi', 'english'], default='hindi')
    parser.add_argument('--model', default=base.DEFAULT_MODEL)
    parser.add_argument('--adapter', action='store_true')
    parser.add_argument('--train-sentences', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--micro-batch', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--examples', type=int, default=500)
    parser.add_argument('--starts', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.data_dir = ROOT / f'data_{args.language}'
    args.output_dir = ROOT / 'outputs' / f'{args.language}_causal_lm' / args.model.split('/')[-1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.is_telugu = script_check(args.language)  # WordRanker reads this module-level check.
    use_model(args.model)
    if args.command == 'train':
        raw = args.data_dir / f'{args.language}_raw.txt'
        link = args.output_dir / 'telugu_raw.txt'  # base.train reads this fixed file name.
        link.unlink(missing_ok=True)
        link.symlink_to(raw.resolve())
        args.data_dir, data = args.output_dir, args.data_dir
        try:
            base.train(args)
        finally:
            link.unlink(missing_ok=True)
        return
    tokenizer, model = base.load_model(args.model, args.output_dir / 'adapter' if args.adapter else None)
    vocab = pickle.loads((args.data_dir / 'vocab.pkl').read_bytes())['word_to_idx']
    rows = sample(args.data_dir, args.language, args.examples, args.seed)
    ranker = base.WordRanker(tokenizer, model, starts=args.starts)
    hits = [0, 0, 0]
    start = time.time()
    for n, (context, target) in enumerate(rows, 1):
        ranked = ranker.rank(context)
        if target in vocab:
            hits[0] += ranked[:1] == [target]
            hits[1] += target in ranked[:3]
            hits[2] += target in ranked[:5]
        if n % 50 == 0:
            print(json.dumps(dict(done=n, total=len(rows), top1=hits[0] / n, minutes=(time.time() - start) / 60)), flush=True)
    result = dict(model=args.model, fine_tuned=args.adapter, examples=len(rows), top1_accuracy=hits[0] / len(rows),
                  top3_accuracy=hits[1] / len(rows), top5_accuracy=hits[2] / len(rows), target_oov_rate=sum(t not in vocab for _, t in rows) / len(rows),
                  word_starts=args.starts,
                  seconds=time.time() - start, note='Sample only; words ranked by greedy completion; OOV targets are misses.')
    (args.output_dir / f"{'finetuned' if args.adapter else 'zeroshot'}_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
