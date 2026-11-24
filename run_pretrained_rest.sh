#!/bin/zsh
# Hindi BLOOM LoRA diverged at learning rate 5e-5 (training loss guard, step 513), so it is retried at 2e-5.
cd "$(dirname "$0")"
PY=.venv/bin/python
run() { echo "== $*"; $PY -u "$@" || exit 1; }
run src/finetune_causal_lm_hindi.py train --language hindi --model bigscience/bloom-560m --lr 2e-5
run src/finetune_causal_lm_hindi.py eval --language hindi --model bigscience/bloom-560m --adapter
run src/finetune_causal_lm_hindi.py train --language english --model bigscience/bloom-560m
run src/finetune_causal_lm_hindi.py eval --language english --model bigscience/bloom-560m --adapter
