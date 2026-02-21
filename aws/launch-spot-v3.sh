#!/bin/bash
# Launch g5.xlarge spot instance for V3 (multi-scale fusion) training
# Run this after uploading code to S3.
# Does NOT affect the existing V2 instance.

aws ec2 run-instances \
  --image-id ami-0fe59b4f6e7e66c3e \
  --instance-type g5.xlarge \
  --key-name 3body-compute \
  --security-group-ids sg-01db2d9932427a00a \
  --subnet-id subnet-0e58bf22b319ab3cc \
  --iam-instance-profile Name=vesuvius-training-profile \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":150,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --user-data file://aws/user-data-v3.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=vesuvius-v3-training},{Key=Project,Value=vesuvius-challenge},{Key=Pipeline,Value=v3}]' \
  --region us-east-1
