import boto3
import json
import os
import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    Lambda function that processes S3 events and sends SVS file information to SQS.
    
    This function is triggered by S3 events when new files are uploaded to the bucket.
    It filters for SVS files and sends their information to an SQS queue for processing.
    
    Args:
        event (dict): The event dict containing the S3 event details
        context (LambdaContext): The Lambda context object
        
    Returns:
        dict: Response containing status code and message
    """
    # Initialize SQS client
    sqs = boto3.client('sqs')
    
    # Get the queue URL from environment variables
    queue_url = os.environ.get('SQS_QUEUE_URL')
    if not queue_url:
        error_msg = "SQS_QUEUE_URL environment variable is not set"
        logger.error(error_msg)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': error_msg})
        }
    
    processed_files = 0
    skipped_files = 0
    
    try:
        # Process S3 event records
        for record in event['Records']:
            # Extract bucket and file information
            bucket = record['s3']['bucket']['name']
            file_key = record['s3']['object']['key']
            
            # Only process SVS files
            if not file_key.lower().endswith('.svs'):
                logger.info(f"Skipping non-SVS file: {file_key}")
                skipped_files += 1
                continue
            
            # Create message for SQS
            message = {
                'bucket': bucket,
                'file_key': file_key,
                'status': 'pending_processing',
                'timestamp': datetime.utcnow().isoformat(),
                'event_time': record['eventTime']
            }
            
            # Send message to SQS
            response = sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps(message)
            )
            
            logger.info(f"Sent message to SQS for file: {file_key}. MessageId: {response['MessageId']}")
            processed_files += 1
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Processing complete',
                'processed_files': processed_files,
                'skipped_files': skipped_files
            })
        }
        
    except Exception as e:
        error_msg = f"Error processing S3 events: {str(e)}"
        logger.error(error_msg)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': error_msg})
        } 