#!/bin/bash

# Configuration variables
EC2_INSTANCE="i-08a58080616278d9c"
REGION="us-east-1"
IDENTITY_FILE="~/Downloads/clawsec2.pem"
SSH_USER="ec2-user"
SERVER_ALIVE_INTERVAL=115
REMOTE_FORWARD="52698:localhost:52698"
LOCAL_FORWARD_1="8888:localhost:8888"
LOCAL_FORWARD_2="16006:localhost:6006"

# Timeout configurations (in seconds)
WAITER_TIMEOUT=600  # 10 minutes
WAITER_DELAY=15     # 15 seconds between checks
INSTANCE_STARTUP_TIMEOUT=300  # 5 minutes
SSH_CONNECT_TIMEOUT=30  # 30 seconds

# Function to start the EC2 instance if it is stopped
start_instance_if_needed() {
    instance_state=$(aws ec2 describe-instances --instance-ids $EC2_INSTANCE --query "Reservations[0].Instances[0].State.Name" --output text --region $REGION)
    if [ "$instance_state" == "stopped" ] || [ "$instance_state" == "stopping" ]; then
        echo "Starting EC2 instance: $EC2_INSTANCE"
        aws ec2 start-instances --instance-ids $EC2_INSTANCE --region $REGION
        echo "Waiting for EC2 instance to start..."
        aws ec2 wait instance-running --instance-ids $EC2_INSTANCE --region $REGION
        echo "EC2 instance started."
    else
        echo "EC2 instance is already in state '$instance_state'."
    fi
}

# Function to get the public DNS of the EC2 instance
get_instance_dns() {
    public_dns=$(aws ec2 describe-instances --instance-ids $EC2_INSTANCE --query "Reservations[0].Instances[0].PublicDnsName" --output text --region $REGION)
    echo $public_dns
}

# Main script execution
start_instance_if_needed
public_dns=$(get_instance_dns)
echo "Connecting to EC2 instance at $public_dns"

# SSH into the instance and start an interactive terminal
ssh -i $IDENTITY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=$SERVER_ALIVE_INTERVAL -R $REMOTE_FORWARD -L $LOCAL_FORWARD_1 -L $LOCAL_FORWARD_2 $SSH_USER@$public_dns
