#!/bin/sh
# Side study training queue (logs/decisions.md item 8): one epoch per model, frozen hyperparameters, MPS.
# Resumable: re-running skips finished models (result.json) and resumes the current one from latest.pt.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
$PY -m src.train.runs deep --n_blocks 2 --seeds 0 --device mps && \
$PY -m src.train.runs deep --n_blocks 4 --seeds 0 --device mps && \
$PY -m src.train.runs deep --n_blocks 2 --seeds 1,2,3,4 --device mps && \
$PY -m src.train.runs deep --n_blocks 4 --seeds 1,2,3,4 --device mps
