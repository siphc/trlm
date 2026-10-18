#!/bin/sh
# Stops readable-training runs if the laptop runs on battery below 25% (re-running the queue resumes them).
# Usage: scripts/readable/watchdog.sh <pattern of the queue process>   Exits when that queue has finished.
cd "$(dirname "$0")/../.." || exit 1
PAT=${1:-scripts/readable/}
while pgrep -f "$PAT" > /dev/null; do
  b=$(pmset -g batt | tail -1)
  pct=$(echo "$b" | grep -o '[0-9]*%' | tr -d '%')
  if echo "$b" | grep -q "discharging" && [ "${pct:-100}" -lt 25 ]; then
    echo "$(date) battery ${pct}% discharging: stopping readable training" >> logs/runs/battery_watchdog.log
    pkill -f "$PAT"; pkill -f "src.train.runs readable"
    exit 0
  fi
  sleep 60
done
echo "$(date) $PAT finished; watchdog exits" >> logs/runs/battery_watchdog.log
