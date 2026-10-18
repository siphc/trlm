#!/bin/sh
# Build probe_samples for each final model as soon as its final.pt exists.
cd "$(dirname "$0")/../.."
for m in attn_s0 attn_s1 attn_s2 attn_s3 attn_s4 mlp_s0 mlp_s1 mlp_s2 mlp_s3 mlp_s4; do
  until [ -f models/final/$m/result.json ]; do sleep 30; done
  .venv/bin/python -m harness.probes.make_probe_samples $m
done
