#!/bin/bash

# Script to deploy Lambda functions using AWS CLI
# Usage: ./deploy_lambda.sh <function_name>

set -e

# Check if function name is provided
if [ -z "$1" ]; then
    echo "Error: Function name is required"
    echo "Usage: ./deploy_lambda.sh <function_name>"
    exit 1
fi

FUNCTION_NAME=$1
FUNCTION_DIR="aws/lambdas/$FUNCTION_NAME"

# Check if function directory exists
if [ ! -d "$FUNCTION_DIR" ]; then
    echo "Error: Function directory '$FUNCTION_DIR' does not exist"
    exit 1
fi

echo "Deploying Lambda function: $FUNCTION_NAME"

# Create a temporary build directory
BUILD_DIR="/tmp/lambda_build_$FUNCTION_NAME"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Copy function code
cp "$FUNCTION_DIR/lambda_function.py" "$BUILD_DIR/"

# Install dependencies if requirements.txt exists
if [ -f "$FUNCTION_DIR/requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r "$FUNCTION_DIR/requirements.txt" -t "$BUILD_DIR/"
fi

# Create deployment package
echo "Creating deployment package..."
cd "$BUILD_DIR"
zip -r "../$FUNCTION_NAME.zip" .
cd - > /dev/null

# Check if Lambda function exists
FUNCTION_EXISTS=$(aws lambda list-functions --query "Functions[?FunctionName=='$FUNCTION_NAME'].FunctionName" --output text)

if [ -z "$FUNCTION_EXISTS" ]; then
    echo "Creating new Lambda function: $FUNCTION_NAME"
    
    # Create Lambda function
    aws lambda create-function \
        --function-name "$FUNCTION_NAME" \
        --runtime python3.9 \
        --handler lambda_function.lambda_handler \
        --zip-file "fileb:///tmp/$FUNCTION_NAME.zip" \
        --role "arn:aws:iam::ACCOUNT_ID:role/LAMBDA_ROLE" \
        --timeout 30 \
        --memory-size 128 \
        --environment "Variables={SQS_QUEUE_URL=https://sqs.REGION.amazonaws.com/ACCOUNT_ID/svs-processing-queue}"
else
    echo "Updating existing Lambda function: $FUNCTION_NAME"
    
    # Update Lambda function code
    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --zip-file "fileb:///tmp/$FUNCTION_NAME.zip"
        
    # Update Lambda function configuration
    aws lambda update-function-configuration \
        --function-name "$FUNCTION_NAME" \
        --timeout 30 \
        --memory-size 128 \
        --environment "Variables={SQS_QUEUE_URL=https://sqs.REGION.amazonaws.com/ACCOUNT_ID/svs-processing-queue}"
fi

echo "Deployment complete!"
echo "NOTE: Before using this script, replace ACCOUNT_ID, REGION, and LAMBDA_ROLE with your actual values."
echo "You can set these in a config file or as environment variables for more secure handling." 