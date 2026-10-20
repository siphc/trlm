# Cloud instance for studies A and B (decision 12)

One g6.8xlarge (1 NVIDIA L4 24 GB, 32 vCPUs, 128 GiB; about USD 2.01/hour on demand in us-east-1, checked 2026-10-19
from third-party price lists) runs studies A and B: their confirmatory runs, twins and continuations, and their CPU
analysis. The readable-training study stays on the laptop. Budget: USD 200 of credits; the instance stops itself after
60 idle minutes (idle_guard.sh).

## Before the first launch (the human)

1. Account on the Paid plan (Free-plan accounts cannot launch GPU instances; credits carry over), an AWS Budgets alert,
   and the quota "Running On-Demand G and VT instances" = 32 vCPUs in us-east-1 (new accounts start at 0).
2. An IAM user with EC2 and SSM-read permissions; on the laptop run `~/.venvs/awscli/bin/aws configure` (region
   us-east-1) yourself, so the keys never pass through the chat.

## Sequence (the agent, after the human confirms the launch)

    scripts/aws/launch.sh                       # key pair, SSH-only security group, instance; writes instance.txt
    scripts/aws/push.sh                         # code (git-tracked files) + manifest data (1.1 GB) + dev models
    . scripts/aws/common.sh; $SSH $HOST 'cd trlm && bash scripts/aws/setup_remote.sh 2>&1 | tee -a logs/runs/aws_setup.log'
    # job file (one "<name> <command>" per line) -> on the instance:
    $SSH $HOST 'cd trlm && nohup .venv/bin/python scripts/aws/run_jobs.py <jobs> --parallel 6 > logs/runs/aws/run_jobs.out 2>&1 &'
    scripts/aws/status.sh                       # state, GPU use, jobs
    scripts/aws/pull.sh                         # results back (rules.json arrive as rules_from_instance.json)
    scripts/aws/stop.sh                         # stop billing for compute; `stop.sh terminate` deletes it (pull first)

## Rules

- The manifest (manifest.py; hashes in manifest.json) proves the instance has the laptop's data; the test split is
  never copied. Setup recomputes the corpus statistics and study A's usefulness tables and checks them against the
  committed prediction.
- Confirmatory runs start only after the study's FINAL pre-registration is committed (as on the laptop).
- Commit on the laptop; the instance gets code by push.sh, never by editing there.
