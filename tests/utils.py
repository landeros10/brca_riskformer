import boto3
import botocore
from functools import lru_cache

@lru_cache(maxsize=1)
def check_aws_credentials():
    """
    Check if AWS credentials are available and valid.
    Uses boto3's STS service to verify credentials.
    Returns True if credentials are valid, False otherwise.
    """
    try:
        # Try to access AWS with a short timeout
        session = boto3.Session()
        sts_client = session.client('sts', 
                                  config=botocore.config.Config(
                                      connect_timeout=5, 
                                      retries={'max_attempts': 1}
                                  ))
        # This will raise an exception if credentials are invalid
        sts_client.get_caller_identity()
        return True
    except (botocore.exceptions.ClientError, 
            botocore.exceptions.NoCredentialsError,
            botocore.exceptions.ProfileNotFound):
        return False
    except Exception:
        # For any other unexpected errors, assume credentials are not available
        return False 