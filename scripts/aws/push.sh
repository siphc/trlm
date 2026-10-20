#!/bin/sh
# Copy the code (every file git tracks) and the manifest's data to the instance (~/trlm). Re-run after new commits.
set -e
cd "$(dirname "$0")/../.."
. scripts/aws/common.sh
{ git ls-files; .venv/bin/python scripts/aws/manifest.py list; } | grep -v -e 'test\.bin' -e 'test_full' > /tmp/trlm_push.txt
# development models of studies A and B (weights and results only), if any
find models/law_pressure models/fixed_votes \( -name final.pt -o -name result.json \) 2>/dev/null | grep -v scratch >> /tmp/trlm_push.txt || true
rsync -az --files-from=/tmp/trlm_push.txt -e "$SSH" ./ "$HOST:trlm/"
echo "pushed $(wc -l < /tmp/trlm_push.txt) files to $HOST"
