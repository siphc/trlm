#!/bin/sh
# Resume the deep study after a shutdown or crash (safe to run any time: starts only what is not already running,
# and every step skips finished work). Sleep alone needs nothing: processes pause and continue on wake.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
# 1. training queue (skips finished models, resumes the current one from latest.pt) + battery watchdog
if ! pgrep -f "deep_queue.sh" > /dev/null; then
  nohup caffeinate -dimsu scripts/deep_study/deep_queue.sh >> logs/runs/deep_train.out 2>&1 &
  echo "started training queue"
fi
sleep 2
if ! pgrep -f "battery_watchdog.sh" > /dev/null; then
  nohup scripts/deep_study/battery_watchdog.sh > /dev/null 2>&1 &
  echo "started battery watchdog"
fi
# 2. CPU analysis (one process at a time, lowest priority): one-layer code checks, then the echo trajectory test
if ! pgrep -f "cpu_queue_loose_ends.sh|cpu_queue_2.sh|cpu_queue_resume.sh" > /dev/null; then
  nohup scripts/deep_study/cpu_queue_resume.sh >> logs/runs/cpu_queue_resume.log 2>&1 &
  echo "started CPU analysis queue"
fi
