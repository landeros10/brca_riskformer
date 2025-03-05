#!/bin/bash

# Script to configure S3 bucket to trigger Lambda function

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
ACCOUNT_ID=$(jq -r '.account_id' "$CONFIG_FILE")

echo "Configuring S3 bucket '$BUCKET_NAME' to trigger Lambda function '$FUNCTION_NAME'"

# Check if bucket exists
BUCKET_EXISTS=$(aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null && echo "true" || echo "false")

if [ "$BUCKET_EXISTS" = "false" ]; then
    echo "Error: S3 bucket '$BUCKET_NAME' does not exist"
    echo "Please create the bucket first or update the configuration"
    exit 1
fi

# Get Lambda function ARN
LAMBDA_ARN="arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$FUNCTION_NAME"
echo "Lambda ARN: $LAMBDA_ARN"

# Add permission for S3 to invoke Lambda
echo "Adding permission for S3 to invoke Lambda..."
aws lambda add-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id "s3-trigger-$BUCKET_NAME" \
    --action "lambda:InvokeFunction" \
    --principal s3.amazonaws.com \
    --source-arn "arn:aws:s3:::$BUCKET_NAME" \
    --source-account "$ACCOUNT_ID" \
    --region "$REGION"

# Create notification configuration
echo "Creating notification configuration..."
NOTIFICATION_CONFIG=$(cat <<EOF
{
    "LambdaFunctionConfigurations": [
        {
            "LambdaFunctionArn": "$LAMBDA_ARN",
            "Events": ["s3:ObjectCreated:*"],
            "Filter": {
                "Key": {
                    "FilterRules": [
                        {
                            "Name": "suffix",
                            "Value": ".svs"
                        }
                    ]
                }
            }
        }
    ]
}
EOF
)

# Save notification configuration to a temporary file
echo "$NOTIFICATION_CONFIG" > /tmp/notification.json

# Apply notification configuration to the bucket
echo "Applying notification configuration to bucket..."
aws s3api put-bucket-notification-configuration \
    --bucket "$BUCKET_NAME" \
    --notification-configuration file:///tmp/notification.json \
    --region "$REGION"

echo "S3 trigger configuration complete!"
echo "The Lambda function '$FUNCTION_NAME' will now be triggered when SVS files are uploaded to the bucket '$BUCKET_NAME'" 