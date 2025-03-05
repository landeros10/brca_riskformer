# AWS Components for BRCA RiskFormer

This directory contains AWS infrastructure components for the BRCA RiskFormer project, focusing on automated processing of SVS files using AWS services.

## Directory Structure

```
aws/
├── config.json                # Configuration for AWS resources
├── create_sqs_queue.sh        # Script to create SQS queue
├── configure_s3_trigger.sh    # Script to configure S3 trigger for Lambda
├── ec2/                       # EC2-related scripts
│   └── run_preprocess_ec2.py  # Script for preprocessing on EC2
└── lambdas/                   # Lambda functions
    ├── deploy_lambda.sh       # Script to deploy Lambda functions
    └── svs_processor/         # Lambda function for processing SVS files
        ├── lambda_function.py # Lambda function code
        └── requirements.txt   # Lambda function dependencies
```

## Setup Instructions

### Prerequisites

1. AWS CLI installed and configured with appropriate credentials
2. jq installed for JSON processing in scripts
3. Python 3.9+ installed
4. An AWS account with appropriate permissions

### Configuration

1. Edit `config.json` to set your AWS account ID, region, and resource names.

### Deployment Steps

#### 1. Create SQS Queue

```bash
bash aws/create_sqs_queue.sh
```

This creates an SQS queue that will hold messages for SVS files to be processed.

#### 2. Deploy Lambda Function

```bash
bash aws/lambdas/deploy_lambda.sh svs_processor
```

This deploys the Lambda function that will be triggered when new SVS files are uploaded to S3.

#### 3. Configure S3 Trigger

```bash
bash aws/configure_s3_trigger.sh
```

This configures your S3 bucket to trigger the Lambda function when new SVS files are uploaded.

## Workflow

1. SVS files are uploaded to the configured S3 bucket
2. The S3 bucket triggers the Lambda function
3. The Lambda function sends a message to the SQS queue
4. EC2 instances or other compute resources can poll the SQS queue for files to process
5. After processing, results can be stored back in S3

## IAM Permissions

The Lambda function requires the following permissions:
- Read access to the S3 bucket
- Send message permissions for the SQS queue
- CloudWatch Logs permissions for logging

## Monitoring

You can monitor the workflow using:
- CloudWatch Logs for Lambda function logs
- SQS metrics for queue depth and processing rates
- S3 event notifications for file uploads

## Troubleshooting

- Check CloudWatch Logs for Lambda function errors
- Verify IAM permissions if access is denied
- Ensure the S3 bucket notification configuration is correct
- Verify the SQS queue URL in the Lambda function environment variables 