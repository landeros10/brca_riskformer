#!/bin/bash

# Script to create an SQS queue for SVS processing

set -e

# Load configuration
CONFIG_FILE="aws/config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found"
    exit 1
fi

# Extract values from config
REGION=$(jq -r '.region' "$CONFIG_FILE")
QUEUE_NAME=$(jq -r '.sqs.queue_name' "$CONFIG_FILE")
VISIBILITY_TIMEOUT=$(jq -r '.sqs.visibility_timeout' "$CONFIG_FILE")
MESSAGE_RETENTION_PERIOD=$(jq -r '.sqs.message_retention_period' "$CONFIG_FILE")

echo "Creating SQS queue: $QUEUE_NAME in region $REGION"

# Check if queue already exists
QUEUE_URL=$(aws sqs get-queue-url --queue-name "$QUEUE_NAME" --region "$REGION" --output text 2>/dev/null || echo "")

if [ -z "$QUEUE_URL" ]; then
    echo "Creating new SQS queue..."
    
    # Create the queue
    QUEUE_URL=$(aws sqs create-queue \
        --queue-name "$QUEUE_NAME" \
        --attributes "{\"VisibilityTimeout\":\"$VISIBILITY_TIMEOUT\",\"MessageRetentionPeriod\":\"$MESSAGE_RETENTION_PERIOD\"}" \
        --region "$REGION" \
        --output text)
    
    echo "Queue created: $QUEUE_URL"
else
    echo "Queue already exists: $QUEUE_URL"
    
    # Update queue attributes
    aws sqs set-queue-attributes \
        --queue-url "$QUEUE_URL" \
        --attributes "{\"VisibilityTimeout\":\"$VISIBILITY_TIMEOUT\",\"MessageRetentionPeriod\":\"$MESSAGE_RETENTION_PERIOD\"}" \
        --region "$REGION"
    
    echo "Queue attributes updated"
fi

# Get the queue ARN
QUEUE_ARN=$(aws sqs get-queue-attributes \
    --queue-url "$QUEUE_URL" \
    --attribute-names QueueArn \
    --region "$REGION" \
    --query 'Attributes.QueueArn' \
    --output text)

echo "Queue ARN: $QUEUE_ARN"
echo "Queue URL: $QUEUE_URL"

# Save queue URL and ARN to a file for reference
echo "{\"queue_url\": \"$QUEUE_URL\", \"queue_arn\": \"$QUEUE_ARN\"}" > "aws/sqs_queue_info.json"
echo "Queue information saved to aws/sqs_queue_info.json" 