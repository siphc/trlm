#!/bin/sh
# Launch the studies' instance (decision 12). COSTS MONEY: run only after the human has confirmed type and budget.
# Usage: scripts/aws/launch.sh [instance type, default g6.8xlarge]   (region: $AWS_REGION, default us-east-1)
# Creates (once) key pair trlm-aws (~/.ssh/trlm-aws.pem) and security group trlm-ssh (SSH from this machine's public IP
# only), launches the newest AWS Deep Learning Base OSS NVIDIA Driver GPU AMI (Ubuntu 24.04) with a 200 GB gp3 disk,
# and writes "<instance id> <public ip>" to scripts/aws/instance.txt (not in git).
set -e
cd "$(dirname "$0")/../.."
REGION=${AWS_REGION:-us-east-1}
TYPE=${1:-g6.8xlarge}
A="${AWS_CLI:-$HOME/.venvs/awscli/bin/aws} --region $REGION"
AMI=$($A ssm get-parameter --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id \
      --query Parameter.Value --output text)
KEY=$HOME/.ssh/trlm-aws.pem
if [ ! -f "$KEY" ]; then
  $A ec2 create-key-pair --key-name trlm-aws --query KeyMaterial --output text > "$KEY"; chmod 600 "$KEY"
fi
SG=$($A ec2 describe-security-groups --group-names trlm-ssh --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [ -z "$SG" ] || [ "$SG" = "None" ]; then
  SG=$($A ec2 create-security-group --group-name trlm-ssh --description "SSH to the trlm study instance" --query GroupId --output text)
fi
MYIP=$(curl -s https://checkip.amazonaws.com)
$A ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32" 2>/dev/null || true
ID=$($A ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name trlm-aws --security-group-ids "$SG" \
     --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}' \
     --instance-initiated-shutdown-behavior stop \
     --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=trlm-studies}]' \
     --query 'Instances[0].InstanceId' --output text)
$A ec2 wait instance-running --instance-ids "$ID"
IP=$($A ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "$ID $IP" > scripts/aws/instance.txt
echo "launched $ID ($TYPE, $AMI) at $IP"
