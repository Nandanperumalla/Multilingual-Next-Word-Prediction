#!/bin/zsh
# Telugu GRU and Transformer as new files in outputs/telugu_families, then a new Combined model.
# Earlier Telugu checkpoints are only read, never written. The Transformer settings are picked by a pilot.
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=outputs/telugu_families
mkdir -p $OUT
train() { $PY -u src/train_families_hindi.py --language telugu "$@" || exit 1; }
pgrep -f "language telugu --family gru" > /dev/null || [[ -f $OUT/gru_test_metrics.json ]] || train --family gru --epochs 5
while pgrep -f finetune_causal_lm_hindi > /dev/null; do sleep 60; done  # Share the Apple GPU with no one.
for layers in 2 4; do for lr in 0.001 0.0003; do
  [[ -f $OUT/transformer_pilot/l${layers}_lr$lr/transformer_history.json ]] || \
  train --family transformer --device mps --epochs 1 --layers $layers --lr $lr \
    --limit-train 1000000 --no-test --output-dir $OUT/transformer_pilot/l${layers}_lr$lr
done; done
best=($($PY - <<'E'
import json, pathlib
runs = {p.parent.name: json.loads(p.read_text())[-1]['validation']['top1_accuracy']
        for p in pathlib.Path('outputs/telugu_families/transformer_pilot').glob('*/transformer_history.json')}
name = max(runs, key=runs.get)
print(name[1:].split('_lr')[0], name.split('_lr')[1])
E
))
echo "Pilot choice: ${best[1]} layers, lr ${best[2]}"
train --family transformer --device mps --epochs 5 --layers ${best[1]} --lr ${best[2]}
while pgrep -f "train_families_hindi.py --language telugu --family gru" > /dev/null; do sleep 60; done
$PY -u src/train_ensemble_hindi.py --language telugu || exit 1
$PY src/summarize_families_hindi.py --language telugu
