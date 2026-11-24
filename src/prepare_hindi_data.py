"""Prepare the Hindi Alpaca corpus for next-word prediction (same method as Telugu).

Source: iamshnoo/alpaca-cleaned-hindi (yahma/alpaca-cleaned translated with NLLB-1.3B).
Method matches prepare_telugu_llm_data.py: split by SHA256(seed 42, normalized instruction)
into 90/5/5, then remove exact sentence overlap; vocabulary fitted on training only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from preprocess import preprocess_corpus, write_sequence_shards
from sequence_data import load_pickle
from text_utils import clean_text, configure_utf8_stdio, project_root, split_sentences

LANG = "hindi"
FIELDS = ("instruction", "input", "output")
DATASET = "iamshnoo/alpaca-cleaned-hindi"


def split_for_key(text: str) -> str:
    value = int(hashlib.sha256(("42:" + text).encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if value < 5 else "validation" if value < 10 else "train"


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root() / "data_hindi")
    parser.add_argument("--source", type=Path, default=None, help="parquet file (default: data-dir/source/alpaca_hindi.parquet)")
    args = parser.parse_args()
    base = args.data_dir.resolve()
    source = args.source or base / "source" / "alpaca_hindi.parquet"
    if (base / "provenance.json").exists():
        raise SystemExit("Prepared corpus already exists; use a fresh --data-dir to rebuild.")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = pd.read_parquet(source)
    missing = set(FIELDS) - set(frame.columns)
    if missing:
        raise SystemExit(f"Missing columns {missing}; found {list(frame.columns)}")
    print(f"Read {len(frame):,} rows from {source.name}", flush=True)

    groups = {name: set() for name in ("train", "validation", "test")}
    empty_instruction_rows = 0
    for index, row in enumerate(frame.itertuples(index=False)):
        record = row._asdict()
        instruction = clean_text(str(record["instruction"] or ""), language=LANG)
        if not instruction:  # avoid sending every Latin-only instruction to one split
            empty_instruction_rows += 1
            instruction = f"row-{index}"
        split = split_for_key(instruction)
        for field in FIELDS:
            for sentence in split_sentences(str(record[field] or ""), language=LANG):
                if len(sentence.split()) >= 2:
                    groups[split].add(sentence)

    train_overlaps = groups["train"] & (groups["validation"] | groups["test"])
    val_overlaps = groups["validation"] & groups["test"]
    groups["train"].difference_update(train_overlaps)
    groups["validation"].difference_update(val_overlaps)
    for name, sentences in groups.items():
        if not sentences:
            raise SystemExit(f"Empty split: {name}")
        lines = sorted(sentences)
        random.Random(42).shuffle(lines)
        directory = base if name == "train" else base / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "hindi_raw.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts = {name: len(s) for name, s in groups.items()}
    del groups, train_overlaps, val_overlaps

    sentence_count, sequences, vocab_size = preprocess_corpus(
        base / "hindi_raw.txt", base / "sequences.pkl", base / "vocab.pkl", base / "ngram_model.pkl",
        max_vocab=20000, min_freq=2, seq_length=8, shard_size=50000,
        max_legacy_sequences=0, max_ngram_lines=15000, language=LANG,
    )
    vocabulary = load_pickle(base / "vocab.pkl")
    split_metrics = {"train": {"sentences": sentence_count, "sequences": sequences}}
    for name in ("validation", "test"):
        directory = base / name
        n_sentences, n_sequences, _ = write_sequence_shards(
            directory / "hindi_raw.txt", vocabulary["word_to_idx"], 8,
            directory / "sequences_manifest.json", directory / "sequence_shards",
            50000, directory / "sequences.pkl", 0, language=LANG,
        )
        split_metrics[name] = {"sentences": n_sentences, "sequences": n_sequences}
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [{"file": source.name, "dataset": DATASET, "sha256": digest, "rows": len(frame),
                     "url": f"https://huggingface.co/datasets/{DATASET}",
                     "note": "machine-translated from yahma/alpaca-cleaned with NLLB-1.3B; license not stated on the card, check upstream (yahma/alpaca-cleaned is CC BY 4.0)"}],
        "text_fields": list(FIELDS), "language": LANG,
        "task": "next-word LSTM training",
        "split_method": "SHA256(seed=42, normalized instruction): 90/5/5; exact sentence overlaps removed",
        "rows_with_empty_cleaned_instruction": empty_instruction_rows,
        "splits": split_metrics, "unique_sentences": counts, "vocab_size": vocab_size,
        "vocab_fit": "training split only", "ngram_training_sentence_limit": 15000,
    }
    (base / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(provenance, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
