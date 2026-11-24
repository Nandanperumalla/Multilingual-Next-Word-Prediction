#!/bin/zsh
# Pretrained models on the fixed 500-example test sample (seed 42), one at a time.
# Project models first (fast), then no fine-tuning, then LoRA fine-tuning.
cd "$(dirname "$0")"
PY=.venv/bin/python
run() { echo "== $*"; $PY -u "$@" || exit 1; }
run src/evaluate_slice_hindi.py --language hindi
run src/evaluate_slice_hindi.py --language english
run src/finetune_causal_lm_hindi.py eval --language english --model gpt2
run src/finetune_causal_lm_hindi.py eval --language hindi --model bigscience/bloom-560m
run src/finetune_causal_lm_hindi.py eval --language english --model bigscience/bloom-560m
run src/finetune_causal_lm_hindi.py train --language english --model gpt2
run src/finetune_causal_lm_hindi.py eval --language english --model gpt2 --adapter
run src/finetune_causal_lm_hindi.py train --language hindi --model bigscience/bloom-560m
run src/finetune_causal_lm_hindi.py eval --language hindi --model bigscience/bloom-560m --adapter
run src/finetune_causal_lm_hindi.py train --language english --model bigscience/bloom-560m
run src/finetune_causal_lm_hindi.py eval --language english --model bigscience/bloom-560m --adapter
