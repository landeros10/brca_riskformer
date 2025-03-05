#!/bin/bash

# Script to test the Lambda function with a sample S3 event

set -e

# Load configuration
CONFIG_FILE="aws/config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found"
    exit 1
fi

# Extract values from config
REGION=$(jq -r '.region' "$CONFIG_FILE")
BUCKET_NAME=$(jq -r '.s3.svs_bucket' "$CONFIG_FILE")
FUNCTION_NAME=$(jq -r '.lambda.svs_processor.function_name' "$CONFIG_FILE")

echo "Testing Lambda function: $FUNCTION_NAME"

# Create a sample S3 event
SAMPLE_EVENT=$(cat <<EOF
{
  "Records": [
    {
      "eventVersion": "2.0",
      "eventSource": "aws:s3",
      "awsRegion": "$REGION",
      "eventTime": "$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")",
      "eventName": "ObjectCreated:Put",
      "s3": {
        "s3SchemaVersion": "1.0",
        "bucket": {
          "name": "$BUCKET_NAME",
          "arn": "arn:aws:s3:::$BUCKET_NAME"
        },
        "object": {
          "key": "test/sample.svs",
          "size": 1024
        }
      }
    }
  ]
}
EOF
)

# Save the sample event to a file
echo "$SAMPLE_EVENT" > /tmp/sample_s3_event.json

echo "Invoking Lambda function with sample S3 event..."
aws lambda invoke \
    --function-name "$FUNCTION_NAME" \
    --payload file:///tmp/sample_s3_event.json \
    --cli-binary-format raw-in-base64-out \
    --region "$REGION" \
    /tmp/lambda_output.json

echo "Lambda function response:"
cat /tmp/lambda_output.json | jq .

echo "Test complete!" 