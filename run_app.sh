#!/usr/bin/env bash
# macOS/Linux launcher: creates .venv on first run, then starts the website.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -q -r requirements.txt
fi
echo "Open http://127.0.0.1:${1:-8501}"
.venv/bin/python src/server.py --port "${1:-8501}"
