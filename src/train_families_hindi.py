"""Train LSTM, Bi-LSTM, GRU, gated CNN, gated word-character CNN or a small Transformer on a prepared language.

Uses the prepared vocabulary, split and test examples unchanged. The checkpoint is chosen by validation
cross-entropy, as for the baselines, and the test split is scored once on that checkpoint.
"""
from __future__ import annotations
import argparse
import json
import pickle
import time
from pathlib import Path
import torch
from evaluate_hindi_clean import is_loop
from family_models_hindi import PrefixGRU, PrefixTransformer, load_family
from model import TeluguBiLSTM, TeluguCNN, TeluguLSTM
from text_utils import iter_language_lines, project_root, tokenize
from train_telugu_comparison import evaluate, load_split, save_json, save_model

ROOT = project_root()
ARCHITECTURES = dict(lstm='lstm', bilstm='bilstm', gru='gru', cnn='cnn', wordchar='wordchar_cnn', transformer='transformer')


def loop_mask(data_dir, language, raw_name):
    """True for test examples outside repetition-loop sentences, in sequence order."""
    keep = []
    for line in iter_language_lines(data_dir / 'test' / raw_name, language=language):
        words = tokenize(line, language=language)
        if len(words) >= 2:
            keep += [not is_loop(words)] * (len(words) - 1)
    return torch.tensor(keep)


def report(model, test, mask):
    metrics = evaluate(model, test, 512)
    metrics['without_loops'] = evaluate(model, test[mask], 512)
    return metrics


def build(args, vocabulary_size, character_parts):
    if args.family in ('lstm', 'bilstm', 'gru'):
        config = dict(embedding_dim=64, hidden_dim=128, num_layers=1, dropout=0.0)
        return dict(lstm=TeluguLSTM, bilstm=TeluguBiLSTM, gru=PrefixGRU)[args.family](vocabulary_size, **config), config
    if args.family == 'transformer':
        config = dict(embedding_dim=128, num_layers=args.layers, num_heads=4, feedforward_dim=512, dropout=0.1)
        return PrefixTransformer(vocabulary_size, seq_length=8, **config), config
    config = dict(embedding_dim=64, hidden_dim=128, num_layers=3, dropout=0.0)
    if args.family == 'cnn':
        return TeluguCNN(vocabulary_size, **config), config
    from wordchar_model import TeluguWordCharCNN
    return TeluguWordCharCNN(vocabulary_size, *character_parts, **config), config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=ARCHITECTURES, required=True)
    parser.add_argument('--language', choices=['hindi', 'english', 'telugu'], default='hindi')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--layers', type=int, default=2, help='Transformer layers')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit-train', type=int, help='pilot only: train on this many examples')
    parser.add_argument('--limit-validation', type=int, help='pilot only: validate on this many examples')
    parser.add_argument('--no-test', action='store_true', help='pilot only: never touch the test split')
    args = parser.parse_args()
    data = args.data_dir or (ROOT / 'data_rebuilt' if args.language == 'telugu' else ROOT / f'data_{args.language}')
    out = args.output_dir or ROOT / 'outputs' / f'{args.language}_families'
    raw_name = f'{args.language}_raw.txt'
    torch.set_num_threads(args.threads)
    out.mkdir(parents=True, exist_ok=True)
    architecture = ARCHITECTURES[args.family]
    vocab = pickle.loads((data / 'vocab.pkl').read_bytes())
    character_parts, metadata = None, {}
    test = None
    if args.family == 'wordchar':
        from wordchar_data_hindi import prepare
        (train, validation, test), words, chars, metadata = prepare(data, vocab, args.language, raw_name)
        character_parts = (words, chars, len(metadata['char_to_idx']) + 2)
    else:
        train, validation = load_split(data), load_split(data / 'validation')
    torch.manual_seed(args.seed)
    if args.limit_train:
        train = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed))[:args.limit_train]]
    if args.limit_validation:
        validation = validation[:args.limit_validation]
    model, config = build(args, len(vocab['word_to_idx']), character_parts)
    model.to(args.device)
    parameters = sum(p.numel() for p in model.parameters())
    if args.family == 'transformer':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    experiment = dict(family=args.family, language=args.language, seed=args.seed, batch_size=args.batch_size,
                      epochs=args.epochs, learning_rate=args.lr, device=args.device, model_config=config,
                      train_examples=len(train), validation_examples=len(validation), parameter_count=parameters)
    save_json(out / f'{architecture}_experiment.json', experiment)
    best_path, latest_path = out / f'{architecture}_best.pt', out / f'{architecture}_latest.pt'
    best_loss, history, first_epoch = float('inf'), [], 1
    if latest_path.exists():
        latest = torch.load(latest_path, weights_only=True, map_location=args.device)
        if latest['experiment'] != experiment:
            raise ValueError('Resume settings differ; use a new output directory')
        model.load_state_dict(latest['model_state_dict'])
        optimizer.load_state_dict(latest['optimizer_state_dict'])
        best_loss, history, first_epoch = latest['best_val_loss'], latest['history'], latest['epoch'] + 1

    def bundle(epoch, with_optimizer=False):
        extra = dict(optimizer_state_dict=optimizer.state_dict()) if with_optimizer else {}
        return dict(model_state_dict={k: v.cpu() for k, v in model.state_dict().items()}, architecture=architecture,
                    **config, word_to_idx=vocab['word_to_idx'], idx_to_word=vocab['idx_to_word'],
                    seq_length=vocab['seq_length'], epoch=epoch, best_val_loss=best_loss, history=history,
                    experiment=experiment, language=args.language, character_metadata=metadata, **extra)

    status = out / f'{architecture}_status.json'
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + epoch))
        started, loss_sum, count = time.perf_counter(), 0.0, 0
        batches = -(-len(train) // args.batch_size)
        for i, indices in enumerate(order.split(args.batch_size), 1):
            batch = train[indices].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(batch[:, :-1]), batch[:, -1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            count += len(batch)
            if i % 200 == 0 or i == batches:
                save_json(status, dict(status='training', architecture=architecture, epoch=epoch, epochs=args.epochs,
                                       batch=i, batches=batches, train_loss=loss_sum / count,
                                       elapsed_seconds=time.perf_counter() - started))
        training_seconds = time.perf_counter() - started
        model.cpu()
        metrics = evaluate(model, validation, args.batch_size)
        model.to(args.device)
        history.append(dict(epoch=epoch, train_loss=loss_sum / count, validation=metrics, training_seconds=training_seconds))
        if metrics['cross_entropy'] < best_loss:
            best_loss = metrics['cross_entropy']
            save_model(best_path, bundle(epoch))
        save_model(latest_path, bundle(epoch, True))
        save_json(out / f'{architecture}_history.json', history)
        print(architecture, 'epoch', epoch, 'val top1 %.4f' % metrics['top1_accuracy'], int(training_seconds), 's', flush=True)
    if not args.no_test:
        checkpoint = torch.load(best_path, weights_only=True, map_location='cpu')
        if test is None:
            test = load_split(data / 'test')
        metrics = report(load_family(checkpoint), test, loop_mask(data, args.language, raw_name))
        metrics.update(architecture=architecture, best_epoch=checkpoint['epoch'], completed_epochs=len(history),
                       parameter_count=parameters, checkpoint=str(best_path),
                       validation=next(h['validation'] for h in history if h['epoch'] == checkpoint['epoch']))
        save_json(out / f'{architecture}_test_metrics.json', metrics)
        print('test', architecture, {k: metrics[k] for k in ('top1_accuracy', 'top3_accuracy', 'top5_accuracy')}, flush=True)
    latest_path.unlink(missing_ok=True)
    save_json(status, dict(status='complete', architecture=architecture))


if __name__ == '__main__':
    main()
