#!/bin/sh
# Battery watchdog: stops a training queue if the laptop runs on battery below 25% (re-running the queue resumes it).
# Usage: nohup scripts/common/watchdog.sh <pattern of the queue process> > /dev/null 2>&1 &
# Exits when that queue has finished. Log: logs/runs/battery_watchdog.log.
cd "$(dirname "$0")/../.." || exit 1
PAT=${1:?usage: watchdog.sh <queue pattern>}
while pgrep -f "$PAT" > /dev/null; do
  b=$(pmset -g batt | tail -1)
  pct=$(echo "$b" | grep -o '[0-9]*%' | tr -d '%')
  if echo "$b" | grep -q "discharging" && [ "${pct:-100}" -lt 25 ]; then
    echo "$(date) battery ${pct}% discharging: stopping $PAT" >> logs/runs/battery_watchdog.log
    pkill -f "$PAT"; pkill -f "src.train.runs"
    exit 0
  fi
  sleep 60
done
echo "$(date) $PAT finished; watchdog exits" >> logs/runs/battery_watchdog.log
