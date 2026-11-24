#!/usr/bin/env bash
# Repeat the four baseline experiments over several seeds, two jobs at a time.
# Finished or interrupted jobs resume from their checkpoints when this is rerun.
# Usage: ./run_seed_experiments.sh [job ...]   Jobs look like 2:recurrent, 2:cnn or 2:wordchar.
set -u
cd "$(dirname "$0")"
OUT=outputs/telugu_seeds
mkdir -p "$OUT/logs"

run_job() {
  seed=${1%%:*}; kind=${1##*:}
  case "$kind" in
    recurrent) dir=$OUT/seed_$seed; flags= ;;
    cnn) dir=$OUT/seed_${seed}_cnn; flags=--cnn-only ;;
    wordchar) dir=$OUT/seed_${seed}_wordchar; flags=--wordchar-only ;;
  esac
  .venv/bin/python src/train_telugu_comparison.py --data-dir data_rebuilt --output-dir "$dir" \
    --epochs 5 --seed "$seed" $flags > "$OUT/logs/seed_${seed}_$kind.log" 2>&1
  echo "$1 exit=$?"
}

if [ "${RUN_JOB:-}" ]; then run_job "$RUN_JOB"; exit; fi
if [ $# -eq 0 ]; then
  # Recurrent jobs take about an hour; CNN jobs take several, so they queue last.
  set -- 1:recurrent 2:recurrent 3:recurrent 4:recurrent 1:cnn 1:wordchar 2:cnn 2:wordchar 3:cnn 3:wordchar 4:cnn 4:wordchar
fi
printf '%s\n' "$@" | RUN_JOB= xargs -P 2 -I{} env RUN_JOB={} "$PWD/$(basename "$0")"
