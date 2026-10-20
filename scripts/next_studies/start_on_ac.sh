#!/bin/sh
# Waits until the laptop runs on the charger, then starts (or resumes) both training queues with their battery
# watchdogs: the readable-training confirmatory queue and the studies' development queue.
cd "$(dirname "$0")/../.." || exit 1
until pmset -g batt | head -1 | grep -q "AC Power"; do sleep 30; done
echo "$(date) on AC power: starting queues" >> logs/runs/battery_watchdog.log
nohup caffeinate -dimsu scripts/readable/confirm_queue.sh >> logs/runs/readable_confirm.out 2>&1 &
sleep 2
nohup scripts/readable/watchdog.sh scripts/readable/confirm_queue.sh > /dev/null 2>&1 &
nohup caffeinate -dimsu scripts/next_studies/dev_queue.sh mps >> logs/runs/next_studies_dev.out 2>&1 &
sleep 2
nohup scripts/next_studies/watchdog.sh scripts/next_studies/dev_queue.sh > /dev/null 2>&1 &
