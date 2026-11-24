#!/bin/zsh
# Hindi model families, one at a time. CNNs and Transformer use the Apple GPU; results match CPU in the timing test.
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=outputs/hindi_families
mkdir -p $OUT
$PY -u src/train_families_hindi.py --family gru --epochs 3 && \
$PY -u src/train_families_hindi.py --family cnn --epochs 5 --device mps && \
$PY -u src/train_families_hindi.py --family wordchar --epochs 5 --device mps && \
for layers in 2 4; do for lr in 0.001 0.0003; do
  $PY -u src/train_families_hindi.py --family transformer --device mps --epochs 1 --layers $layers --lr $lr \
    --limit-train 1000000 --no-test --output-dir $OUT/transformer_pilot/l${layers}_lr$lr || exit 1
done; done
