#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Create venv if missing
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip setuptools wheel >/dev/null
pip install -r requirements.txt

# Run API (reload for local dev)
uvicorn main:app --reload