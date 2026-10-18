#!/bin/sh
# Readable-training side study, phase 3 (logs/readable_training.md): the development grid on seed 0 at 100M tokens,
# MPS. Resumable: re-running skips finished runs (result.json) and resumes the current one from latest.pt.
cd "$(dirname "$0")/../.." || exit 1
.venv/bin/python -m src.train.runs readable_dev --arms none,O,I,control --device mps
