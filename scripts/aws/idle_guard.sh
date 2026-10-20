#!/bin/sh
# Runs on the instance: stops (not terminates) the instance after 60 idle minutes (no training or analysis process),
# so a lost connection cannot burn the credits. The disk survives a stop.
idle=0
while true; do
  if pgrep -f "src.train|analysis\.|pytest|run_jobs" > /dev/null; then idle=0; else idle=$((idle + 5)); fi
  [ "$idle" -ge 60 ] && sudo shutdown -h now
  sleep 300
done
