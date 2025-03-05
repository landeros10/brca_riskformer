#!/usr/bin/env python3
"""
sync_cloudwatch_logs.py

Syncs CloudWatch logs to local files for monitoring preprocessing progress.

Author: landeros10
Created: 2025-03-05
"""
import os
import time
import boto3
import logging
import argparse
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)

def get_cloudwatch_logs(
    log_group_name: str,
    start_time: datetime = None,
    region_name: str = None,
):
    """
    Retrieve logs from CloudWatch for the specified log group.
    
    Args:
        log_group_name (str): Name of the CloudWatch log group
        start_time (datetime): Only fetch logs after this time
        region_name (str): AWS region name
    """
    if not region_name:
        region_name = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    
    logs_client = boto3.client('logs', region_name=region_name)
    
    # Convert start_time to milliseconds since epoch
    start_time_ms = int(start_time.timestamp() * 1000) if start_time else None
    
    try:
        # Get all log streams in the group, sorted by last event time
        streams = logs_client.describe_log_streams(
            logGroupName=log_group_name,
            orderBy='LastEventTime',
            descending=True
        )
        
        for stream in streams['logStreams']:
            stream_name = stream['logStreamName']
            logger.info(f"Fetching logs from stream: {stream_name}")
            
            kwargs = {
                'logGroupName': log_group_name,
                'logStreamName': stream_name,
                'startFromHead': True
            }
            
            if start_time_ms:
                kwargs['startTime'] = start_time_ms
            
            while True:
                response = logs_client.get_log_events(**kwargs)
                
                for event in response['events']:
                    yield {
                        'timestamp': datetime.fromtimestamp(event['timestamp'] / 1000),
                        'message': event['message'],
                        'stream': stream_name
                    }
                
                # Check if we've reached the end of the stream
                if not response['events'] or \
                   kwargs.get('nextToken') == response['nextForwardToken']:
                    break
                    
                kwargs['nextToken'] = response['nextForwardToken']
                
    except Exception as e:
        logger.error(f"Error fetching CloudWatch logs: {e}")
        raise

def write_logs_to_file(logs, output_dir: str):
    """Write logs to local files organized by date."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group logs by date
    logs_by_date = {}
    for log in logs:
        date_str = log['timestamp'].strftime('%Y_%m_%d')
        if date_str not in logs_by_date:
            logs_by_date[date_str] = []
        logs_by_date[date_str].append(log)
    
    # Write each day's logs to a separate file
    for date_str, day_logs in logs_by_date.items():
        output_file = os.path.join(output_dir, f"cloudwatch_logs_{date_str}.log")
        with open(output_file, 'w') as f:
            for log in sorted(day_logs, key=lambda x: x['timestamp']):
                f.write(f"{log['timestamp']} - {log['stream']} - {log['message']}\n")
        logger.info(f"Wrote logs to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Sync CloudWatch logs to local files")
    parser.add_argument("--log-group", type=str, default="riskformer_preprocessing_ec2",
                       help="CloudWatch log group name")
    parser.add_argument("--output-dir", type=str, default="logs/cloudwatch",
                       help="Directory to store local log files")
    parser.add_argument("--region", type=str, help="AWS region name")
    parser.add_argument("--hours", type=int, default=24,
                       help="Number of hours of logs to fetch (default: 24)")
    parser.add_argument("--watch", action="store_true",
                       help="Continuously watch for new logs")
    args = parser.parse_args()
    
    setup_logging()
    
    start_time = datetime.now() - timedelta(hours=args.hours)
    logger.info(f"Fetching logs from {start_time}")
    
    while True:
        try:
            logs = list(get_cloudwatch_logs(
                args.log_group,
                start_time=start_time,
                region_name=args.region
            ))
            write_logs_to_file(logs, args.output_dir)
            
            if not args.watch:
                break
                
            logger.info("Waiting for new logs...")
            time.sleep(60)  # Wait 1 minute before checking for new logs
            
        except KeyboardInterrupt:
            logger.info("Stopping log sync")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            if not args.watch:
                break
            time.sleep(60)  # Wait before retrying

if __name__ == "__main__":
    main() 