#!/usr/bin/env bash
set -e

export PYTHONPATH=src

python -m uvicorn analyser_ml.api:app \
  --app-dir src \
  --reload \
  --host 0.0.0.0 \
  --port 8000
