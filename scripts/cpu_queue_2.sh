#!/bin/sh
# Runs after scripts/cpu_queue_loose_ends.sh: the echo trajectory test (addendum D), CPU, 2 threads.
cd "$(dirname "$0")/.." || exit 1
while pgrep -f "cpu_queue_loose_ends.sh" > /dev/null; do sleep 60; done
export OMP_NUM_THREADS=2
echo "== D: echo trajectory $(date)"
nice -n 20 .venv/bin/python -m analysis.look_write_deep.echo_traj --tokens 5000000 --device cpu
echo "== done $(date)"
