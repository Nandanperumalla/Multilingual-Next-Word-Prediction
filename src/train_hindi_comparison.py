"""Train and compare Hindi prefix-only LSTM / Bi-LSTM (identical settings to the Telugu baselines)."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import torch

from model import TeluguBiLSTM, TeluguLSTM, model_from_checkpoint
from text_utils import project_root
from train_telugu_comparison import evaluate, load_split, save_json, save_model


def audit_data(base: Path) -> dict:
    paths = {n: base / ("" if n == "train" else n) / "hindi_raw.txt" for n in ("train", "validation", "test")}
    sentences = {n: set(p.read_text(encoding="utf-8").splitlines()) for n, p in paths.items()}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if sentences[a] & sentences[b]:
            raise ValueError(f"Exact sentence leakage between {a} and {b}")
    return {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root() / "data_hindi")
    parser.add_argument("--output-dir", type=Path, default=project_root() / "outputs" / "hindi_comparison")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=["lstm", "bilstm"], default=["lstm", "bilstm"])
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    hashes = audit_data(args.data_dir)
    vocab = pickle.loads((args.data_dir / "vocab.pkl").read_bytes())
    provenance = json.loads((args.data_dir / "provenance.json").read_text(encoding="utf-8"))
    train = load_split(args.data_dir)
    validation = load_split(args.data_dir / "validation")
    config = dict(embedding_dim=64, hidden_dim=128, num_layers=1, dropout=0.0)
    experiment = dict(seed=args.seed, batch_size=args.batch_size, epochs=args.epochs, threads=args.threads,
                      train_examples=len(train), validation_examples=len(validation), data_hashes=hashes,
                      model_config=config, learning_rate=0.001, device="cpu", torch_version=str(torch.__version__))
    save_json(out / "experiment.json", experiment)
    classes = {"lstm": TeluguLSTM, "bilstm": TeluguBiLSTM}
    summary = {}
    for architecture in args.models:
        torch.manual_seed(args.seed)
        model = classes[architecture](len(vocab["word_to_idx"]), **config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        best_loss, history, best_path = float("inf"), [], out / f"{architecture}_best.pt"

        def bundle(epoch):
            return dict(model_state_dict=model.state_dict(), architecture=architecture, **config,
                        word_to_idx=vocab["word_to_idx"], idx_to_word=vocab["idx_to_word"],
                        seq_length=vocab["seq_length"], epoch=epoch, best_val_loss=best_loss,
                        history=history, experiment=experiment, dataset_provenance=provenance,
                        language="hindi", character_metadata={})

        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + epoch))
            started, loss_sum, count = time.perf_counter(), 0.0, 0
            batches = -(-len(train) // args.batch_size)
            for i, indices in enumerate(order.split(args.batch_size), 1):
                batch = train[indices]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(batch[:, :-1]), batch[:, -1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item() * len(batch)
                count += len(batch)
                if i % 200 == 0 or i == batches:
                    save_json(out / "status.json", dict(status="training", architecture=architecture, epoch=epoch,
                              epochs=args.epochs, batch=i, batches=batches, train_loss=loss_sum / count,
                              elapsed_seconds=time.perf_counter() - started))
            metrics = evaluate(model, validation, args.batch_size)
            history.append(dict(epoch=epoch, train_loss=loss_sum / count, validation=metrics,
                                training_seconds=time.perf_counter() - started))
            if metrics["cross_entropy"] < best_loss:
                best_loss = metrics["cross_entropy"]
                save_model(best_path, bundle(epoch))
            save_json(out / f"{architecture}_history.json", history)
            print(architecture, "epoch", epoch, "val top1 %.4f" % metrics["top1_accuracy"], flush=True)
        checkpoint = torch.load(best_path, weights_only=True, map_location="cpu")
        test = load_split(args.data_dir / "test")
        metrics = evaluate(model_from_checkpoint(checkpoint), test, args.batch_size)
        metrics.update(architecture=architecture, best_epoch=checkpoint["epoch"], completed_epochs=len(history),
                       parameter_count=sum(p.numel() for p in model.parameters()), checkpoint=str(best_path))
        save_json(out / f"{architecture}_test_metrics.json", metrics)
        summary[architecture] = metrics
        del test
    save_json(out / "comparison.json", summary)
    save_json(out / "status.json", dict(status="complete", results=summary))
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
