#!/usr/bin/env bash
set -e

export PYTHONPATH=src

python -m streamlit run dashboard/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
