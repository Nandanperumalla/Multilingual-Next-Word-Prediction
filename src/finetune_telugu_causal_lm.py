"""LoRA fine-tune a small pretrained multilingual causal LM and score it on next-word prediction.

Examples use the project's held-out test sentences: the previous eight words are the prompt and the
next word is the target. Targets outside the project's 20,000-word vocabulary count as misses, as in
the other models. Whole-word rankings come from greedily completing the top word-start tokens, so they
approximate (not exactly compute) the model's word probabilities.
"""
from __future__ import annotations
import argparse
import json
import pickle
import random
import time
from pathlib import Path
import torch
from text_utils import tokenize, project_root, configure_utf8_stdio

ROOT = project_root()
DEFAULT_MODEL = 'bigscience/bloom-560m'


def device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def read_sentences(path):
    return [line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def load_model(name, adapter=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return tokenizer, model.to(device()).eval()


def train(args):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(args.seed)
    tokenizer, model = load_model(args.model)
    model = get_peft_model(model.train(), LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                      target_modules=['query_key_value', 'dense'], task_type='CAUSAL_LM'))
    model.print_trainable_parameters()
    sentences = read_sentences(args.data_dir / 'telugu_raw.txt')
    random.Random(args.seed).shuffle(sentences)
    sentences = sentences[:args.train_sentences]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    steps = len(sentences) // args.batch_size
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    start, history = time.time(), []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(steps):
        batch = sentences[step * args.batch_size:(step + 1) * args.batch_size]
        # Micro-batches keep the 250k-token output layer within Apple-silicon GPU memory.
        loss = 0.
        for part in range(0, len(batch), args.micro_batch):
            enc = tokenizer(batch[part:part + args.micro_batch], return_tensors='pt', padding=True,
                            truncation=True, max_length=args.max_length).to(device())
            labels = enc.input_ids.masked_fill(enc.attention_mask == 0, -100)
            part_loss = model(**enc, labels=labels).loss * len(enc.input_ids) / len(batch)
            part_loss.backward()
            loss += part_loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); schedule.step(); optimizer.zero_grad()
        if device().type == 'mps':
            torch.mps.empty_cache()
        history.append(loss)
        recent = sum(history[-50:]) / len(history[-50:])
        if step >= 100 and recent > 1.5 * history[0]:
            raise SystemExit(f'Training diverged at step {step}: recent loss {recent:.2f}')
        if step and step % 250 == 0:
            model.save_pretrained(args.output_dir / 'adapter')  # Keep progress if a later step fails.
        if step % 25 == 0:
            print(json.dumps(dict(step=step, steps=steps, loss=sum(history[-25:]) / len(history[-25:]),
                                  minutes=(time.time() - start) / 60)), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir / 'adapter')
    (args.output_dir / 'train.json').write_text(json.dumps(dict(
        model=args.model, sentences=len(sentences), steps=steps, batch_size=args.batch_size, micro_batch=args.micro_batch, lr=args.lr,
        max_length=args.max_length, seed=args.seed, final_loss=sum(history[-50:]) / len(history[-50:]),
        minutes=(time.time() - start) / 60), indent=2))


def test_examples(args, vocab):
    rows = []
    for sentence in read_sentences(args.data_dir / 'test' / 'telugu_raw.txt'):
        words = tokenize(sentence, language='telugu')
        rows += [(words[max(0, i - 8):i], words[i]) for i in range(1, len(words))]
    random.Random(args.seed).shuffle(rows)
    return rows[:args.examples]


def is_telugu(text):
    return bool(text) and all('ఀ' <= ch <= '౿' for ch in text)


class WordRanker:
    """Rank whole next words: try the top Telugu word-start tokens, complete each greedily,
    and score by the summed log-probability of the word's tokens."""

    def __init__(self, tokenizer, model, starts=30, max_tokens=6):
        self.tokenizer, self.model, self.starts, self.max_tokens = tokenizer, model, starts, max_tokens
        pieces = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
        texts = [tokenizer.convert_tokens_to_string([p]) for p in pieces]
        # Tokens that begin a new Telugu word, and tokens that continue one.
        # The output layer can be padded beyond the tokenizer; padded ids are never words.
        size = model.get_output_embeddings().weight.shape[0]
        pad = [False] * (size - len(texts))
        self.word_start = torch.tensor([t.startswith(' ') and is_telugu(t[1:]) for t in texts] + pad)
        self.word_part = torch.tensor([is_telugu(t) for t in texts] + pad)

    @torch.inference_mode()
    def rank(self, context, top_k=5):
        ids = self.tokenizer(' '.join(context), return_tensors='pt').input_ids.to(self.model.device)
        logp = self.model(ids).logits[0, -1].float().log_softmax(-1).cpu()
        logp[~self.word_start] = -torch.inf
        first = logp.topk(self.starts)
        seqs = torch.cat([ids.repeat(self.starts, 1), first.indices[:, None].to(ids.device)], 1)
        scores, done = first.values.clone(), torch.zeros(self.starts, dtype=torch.bool)
        for _ in range(self.max_tokens - 1):
            step = self.model(seqs).logits[:, -1].float().log_softmax(-1).cpu()
            best = step.max(-1)
            # A word ends when the most likely next token is not a Telugu continuation.
            done |= ~self.word_part[best.indices]
            if done.all():
                break
            scores = torch.where(done, scores, scores + best.values)
            nxt = torch.where(done, torch.full_like(best.indices, self.tokenizer.pad_token_id), best.indices)
            seqs = torch.cat([seqs, nxt[:, None].to(seqs.device)], 1)
        words = {}
        for row, score in zip(seqs[:, ids.shape[1]:].tolist(), scores.tolist()):
            row = [t for t in row if t != self.tokenizer.pad_token_id]
            word = self.tokenizer.decode(row).strip()
            if is_telugu(word) and score > words.get(word, -float('inf')):
                words[word] = score
        return sorted(words, key=words.get, reverse=True)[:top_k]


def evaluate(args):
    tokenizer, model = load_model(args.model, args.output_dir / 'adapter' if args.adapter else None)
    with (args.data_dir / 'vocab.pkl').open('rb') as f:  # Written locally by prepare_telugu_llm_data.py.
        vocab = pickle.load(f)['word_to_idx']
    rows = test_examples(args, vocab)
    ranker = WordRanker(tokenizer, model, starts=args.starts)
    hits1 = hits5 = 0
    start = time.time()
    for n, (context, target) in enumerate(rows, 1):
        ranked = ranker.rank(context)
        if target in vocab:  # Out-of-vocabulary targets count as misses.
            hits1 += ranked[:1] == [target]
            hits5 += target in ranked[:5]
        if n % 50 == 0:
            print(json.dumps(dict(done=n, total=len(rows), top1=hits1 / n, top5=hits5 / n,
                                  minutes=(time.time() - start) / 60)), flush=True)
    result = dict(model=args.model, fine_tuned=bool(args.adapter), examples=len(rows),
                  top1_accuracy=hits1 / len(rows), top5_accuracy=hits5 / len(rows), word_starts=args.starts,
                  seconds=time.time() - start,
                  note='Words ranked by greedy completion of the top word-start tokens; OOV targets count as misses.')
    name = 'finetuned' if args.adapter else 'zeroshot'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f'{name}_metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'eval'])
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data_rebuilt')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'telugu_causal_lm')
    parser.add_argument('--adapter', action='store_true', help='eval: use the fine-tuned adapter')
    parser.add_argument('--train-sentences', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--micro-batch', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--examples', type=int, default=500)
    parser.add_argument('--starts', type=int, default=30, help='eval: word-start tokens to complete per example')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args) if args.command == 'train' else evaluate(args)


if __name__ == '__main__':
    main()
