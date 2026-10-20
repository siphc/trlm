#!/bin/sh
# Stop the instance (billing for compute stops; the 200 GB disk remains, a few dollars a month). "terminate" as the
# first argument deletes the instance and its disk instead: pull results first.
cd "$(dirname "$0")/../.."
. scripts/aws/common.sh
if [ "$1" = "terminate" ]; then $AWS ec2 terminate-instances --instance-ids "$ID"; else $AWS ec2 stop-instances --instance-ids "$ID"; fi
