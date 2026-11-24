"""Train a larger-vocabulary subword Bi-LSTM and score it on the project's same test examples."""
import argparse
import json
import math
import pickle
import time
from array import array
from pathlib import Path
import torch
from preprocess import build_vocab_from_counts, count_words
from subword_model import MAX_SUBWORDS, TeluguSubwordBiLSTM, subwords
from text_utils import iter_language_lines, project_root, tokenize
from train_telugu_comparison import load_split, save_json

ROOT = project_root()
SEQ_LENGTH = 8


def encode(path, extended, vocab_size):
    contexts, targets = array('i'), array('i')
    for line in iter_language_lines(path, language='telugu'):
        tokens = tokenize(line, language='telugu')
        if len(tokens) < 2:
            continue
        ids = [extended.setdefault(token, len(extended)) for token in tokens]
        for index in range(1, len(ids)):
            context = ids[max(0, index - SEQ_LENGTH):index]
            contexts.extend([0] * (SEQ_LENGTH - len(context)) + context)
            targets.append(ids[index] if ids[index] < vocab_size else 1)
    return (torch.tensor(contexts, dtype=torch.int32).view(-1, SEQ_LENGTH),
            torch.tensor(targets, dtype=torch.int32))


def prepare(args):
    cache = args.output_dir / 'data.pt'
    if cache.exists():
        return torch.load(cache, weights_only=False)
    counts, _ = count_words(args.data_dir / 'telugu_raw.txt', language='telugu')
    vocab = build_vocab_from_counts(counts, args.vocab, 3)  # Fitted on training text only.
    extended = dict(vocab)
    splits = {name: encode(args.data_dir / path, extended, len(vocab)) for name, path in (
        ('train', 'telugu_raw.txt'), ('validation', 'validation/telugu_raw.txt'), ('test', 'test/telugu_raw.txt'))}
    table = torch.zeros(len(extended), MAX_SUBWORDS, dtype=torch.int32)
    for word, index in extended.items():
        if index > 1:
            pieces = subwords(word, args.buckets)
            table[index, :len(pieces)] = torch.tensor(pieces, dtype=torch.int32)
    data = dict(splits=splits, table=table, vocab=vocab, extended_size=len(extended))
    torch.save(data, cache)
    return data


def verify_same_examples(args, data):
    """The new test rows must be the original rows, position by position."""
    original = load_split(args.data_dir / 'test')
    _, targets = data['splits']['test']
    with (args.data_dir / 'vocab.pkl').open('rb') as f:  # Written locally by prepare_telugu_llm_data.py.
        old = pickle.load(f)['word_to_idx']
    if len(original) != 200092 or len(original) != len(targets):
        raise ValueError(f'Test examples differ: {len(original)} vs {len(targets)}')
    mapping = torch.zeros(len(old), dtype=torch.long)
    for word, index in old.items():
        mapping[index] = data['vocab'][word] if index > 1 else index
    known = original[:, -1].ne(1)
    if not mapping[original[:, -1]][known].eq(targets[known].long()).all():
        raise ValueError('Test targets differ from the original rows')
    return int(known.sum()), int(targets.ne(1).sum())


def evaluate(model, contexts, targets, device, batch_size=2048):
    model.eval()
    loss = top1 = top3 = top5 = 0
    with torch.inference_mode():
        for start in range(0, len(targets), batch_size):
            x = contexts[start:start + batch_size].long().to(device)
            y = targets[start:start + batch_size].long().to(device)
            logits = model(x)
            loss += torch.nn.functional.cross_entropy(logits, y, reduction='sum').item()
            logits[:, :2] = -torch.inf  # Never credit <pad> or <unk> as a correct word.
            matches = logits.topk(5, dim=1).indices.eq(y[:, None])
            top1 += matches[:, 0].sum().item()
            top3 += matches[:, :3].any(1).sum().item()
            top5 += matches.any(1).sum().item()
    total = len(targets)
    return dict(examples=total, cross_entropy=loss / total, vocabulary_perplexity=math.exp(loss / total),
                top1_accuracy=top1 / total, top3_accuracy=top3 / total, top5_accuracy=top5 / total,
                target_oov_rate=float(targets.eq(1).float().mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data_rebuilt')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'telugu_subword')
    parser.add_argument('--vocab', type=int, default=50000)
    parser.add_argument('--buckets', type=int, default=20000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--max-batches', type=int, default=0, help='Stop early to measure speed')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare(args)
    old_known, new_known = verify_same_examples(args, data)
    print(json.dumps(dict(extended_words=data['extended_size'], vocabulary=len(data['vocab']),
                          test_targets_coverable_before=old_known, test_targets_coverable_now=new_known)), flush=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = TeluguSubwordBiLSTM(len(data['vocab']), data['table'].long(), args.buckets).to(device)
    print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    contexts, targets = data['splits']['train']
    best, stale, history = -1., 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        order = torch.randperm(len(targets), generator=torch.Generator().manual_seed(args.seed + epoch))
        loss_sum = count = 0
        for batch_index, indices in enumerate(order.split(args.batch_size), 1):
            x, y = contexts[indices].long().to(device), targets[indices].long().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(y)
            count += len(y)
            if batch_index % 100 == 0:
                save_json(args.output_dir / 'status.json', dict(status='training', epoch=epoch, batch=batch_index,
                          train_loss=loss_sum / count, elapsed_seconds=time.perf_counter() - started))
                print(json.dumps(dict(epoch=epoch, batch=batch_index, loss=round(loss_sum / count, 4),
                                      seconds=round(time.perf_counter() - started))), flush=True)
            if args.max_batches and batch_index >= args.max_batches:
                return
        metrics = evaluate(model, *data['splits']['validation'], device)
        history.append(dict(epoch=epoch, train_loss=loss_sum / count, learning_rate=optimizer.param_groups[0]['lr'],
                            validation=metrics, seconds=time.perf_counter() - started))
        save_json(args.output_dir / 'history.json', history)
        print(json.dumps(history[-1]), flush=True)
        if device.type == 'mps':
            torch.mps.empty_cache()
        if metrics['top1_accuracy'] > best:
            best, stale = metrics['top1_accuracy'], 0
            torch.save(dict(model_state_dict=model.state_dict(), vocab=data['vocab'], epoch=epoch,
                            buckets=args.buckets), args.output_dir / 'subword_bilstm_best.pt')
        else:
            stale += 1
            optimizer.param_groups[0]['lr'] /= 2
            if stale >= args.patience:
                break
    checkpoint = torch.load(args.output_dir / 'subword_bilstm_best.pt', weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    metrics = evaluate(model, *data['splits']['test'], device)
    metrics.update(best_epoch=checkpoint['epoch'], selection='validation_top1')
    save_json(args.output_dir / 'test_metrics.json', metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
