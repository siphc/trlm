# Sourced by the other scripts: instance address and SSH options from scripts/aws/instance.txt.
ID=$(cut -d' ' -f1 scripts/aws/instance.txt); IP=$(cut -d' ' -f2 scripts/aws/instance.txt)
HOST=ubuntu@$IP
SSH="ssh -i $HOME/.ssh/trlm-aws.pem -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=60"
AWS="${AWS_CLI:-$HOME/.venvs/awscli/bin/aws} --region ${AWS_REGION:-us-east-1}"
