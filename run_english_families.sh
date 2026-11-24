#!/bin/zsh
# English model families, one at a time, with the same epochs as Hindi. The Transformer settings are picked by a pilot.
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=outputs/english_families
mkdir -p $OUT
train() { $PY -u src/train_families_hindi.py --language english "$@" || exit 1; }
train --family lstm --epochs 3
train --family bilstm --epochs 3
train --family gru --epochs 3
train --family cnn --epochs 5 --device mps
train --family wordchar --epochs 5 --device mps
for layers in 2 4; do for lr in 0.001 0.0003; do
  train --family transformer --device mps --epochs 1 --layers $layers --lr $lr \
    --limit-train 1000000 --no-test --output-dir $OUT/transformer_pilot/l${layers}_lr$lr
done; done
