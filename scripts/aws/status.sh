#!/bin/sh
# Instance state, GPU use, running jobs and disk on the instance.
cd "$(dirname "$0")/../.."
. scripts/aws/common.sh
$AWS ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].[State.Name,InstanceType,LaunchTime]' --output text
$SSH "$HOST" 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; uptime; df -h ~ | tail -1; pgrep -fa "src.train|analysis\." | cut -c1-150; tail -3 ~/trlm/logs/runs/aws/run_jobs.out 2>/dev/null' || true
