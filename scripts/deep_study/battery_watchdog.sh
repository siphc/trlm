#!/bin/sh
# Stops the deep training queue if the laptop runs on battery below 25% (training resumes later by re-running
# scripts/deep_study/deep_queue.sh). Exits when the queue has finished.
cd "$(dirname "$0")/../.." || exit 1
while pgrep -f "deep_queue.sh" > /dev/null; do
  b=$(pmset -g batt | tail -1)
  pct=$(echo "$b" | grep -o '[0-9]*%' | tr -d '%')
  if echo "$b" | grep -q "discharging" && [ "${pct:-100}" -lt 25 ]; then
    echo "$(date) battery ${pct}% discharging: stopping the training queue" >> logs/runs/battery_watchdog.log
    pkill -f "deep_queue.sh"; pkill -f "src.train.runs deep"
    exit 0
  fi
  sleep 60
done
echo "$(date) queue finished; watchdog exits" >> logs/runs/battery_watchdog.log
