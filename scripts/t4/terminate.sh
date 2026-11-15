#!/bin/bash
# Terminate the launch.sh instance, wait for it, show its describe-instances state, delete the key pair and
# security group, and print the GPU spend (on-demand rate x billed seconds, 60 s minimum).
set -eu
export AWS_PROFILE=${AWS_PROFILE:-mesh-916} AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
state=${STATE:-.t4-state}; . "$state/env"
aws ec2 terminate-instances --instance-ids $iid --query 'TerminatingInstances[0].CurrentState.Name' --output text
aws ec2 wait instance-terminated --instance-ids $iid
aws ec2 describe-instances --instance-ids $iid --query 'Reservations[0].Instances[0].{InstanceId:InstanceId,Type:InstanceType,State:State.Name,Launch:LaunchTime,Reason:StateTransitionReason}' --output table
aws ec2 delete-security-group --group-id $sg && aws ec2 delete-key-pair --key-name $tag && rm -f "$state/key.pem"
python3 -c "
import time; s = max(60, time.time() - $launched); print(f'billed ~{s/60:.1f} min x \${$price}/hr + volume = \${s/3600*($price+0.011):.2f}')"
