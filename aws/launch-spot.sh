#!/bin/bash
# Launch spot instance for Vesuvius training
# Run this after data upload to S3 is complete

aws ec2 run-instances \
  --image-id ami-0fe59b4f6e7e66c3e \
  --instance-type g5.xlarge \
  --key-name 3body-compute \
  --security-group-ids sg-01db2d9932427a00a \
  --subnet-id subnet-0e58bf22b319ab3cc \
  --iam-instance-profile Name=vesuvius-training-profile \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":150,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --user-data file://aws/user-data.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=vesuvius-training},{Key=Project,Value=vesuvius-challenge}]' \
  --region us-east-1
