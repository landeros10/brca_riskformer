import boto3
import time
import subprocess
import os
import argparse
from datetime import datetime
import logging
import signal
import sys
import base64

DEBUG = True

EC2_INSTANCE = "i-08a58080616278d9c"
REGION = "us-east-1"

IDENTITY_FILE = "~/Downloads/clawsec2.pem"
SSH_USER = "ec2-user"
SERVER_ALIVE_INTERVAL = 115
REMOTE_FORWARD = "52698:localhost:52698"
LOCAL_FORWARD_1 = "8888:localhost:8888"
LOCAL_FORWARD_2 = "16006:localhost:6006"

# Timeout configurations (in seconds)
WAITER_TIMEOUT = 600  # 10 minutes
WAITER_DELAY = 15     # 15 seconds between checks
INSTANCE_STARTUP_TIMEOUT = 300  # 5 minutes
SSH_CONNECT_TIMEOUT = 30  # 30 seconds
SSH_PROCESS_TIMEOUT = 5  # 5 seconds
DOCKER_STOP_TIMEOUT = 10  # 10 seconds

REMOTE_COMMANDS = [
    "nvidia-smi",
    "cd ~/brca_riskformer",
    "ls -lR",
    "git pull origin main",
    "./orchestrators/run_preprocess.sh",
]

LOG_PATH = "./logs"
os.makedirs(LOG_PATH, exist_ok=True)
log_filename = f"log_{datetime.now().strftime('%m_%d_%y')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_PATH, log_filename)),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logger.info(f"Logger initialized{' in DEBUG mode' if DEBUG else ''}.")

# Global variables
global_ec2_client = None
ssh_process = None
public_dns = None
expanded_key_path = None
stop_instance_on_exit = False

def cleanup_ssh_process():
    """Clean up the SSH process if it's still running."""
    global ssh_process
    if ssh_process and ssh_process.poll() is None:
        logger.info("Cleaning up SSH process...")
        try:
            ssh_process.terminate()
            ssh_process.wait(timeout=SSH_PROCESS_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.warning(f"SSH process didn't terminate within {SSH_PROCESS_TIMEOUT} seconds, forcing kill...")
            ssh_process.kill()
        except Exception as e:
            logger.error(f"Error cleaning up SSH process: {e}")

# Signal handler for graceful termination
def signal_handler(sig, frame):
    global ssh_process, global_ec2_client, stop_instance_on_exit, public_dns, expanded_key_path
    
    logger.info(f"Received interrupt signal ({sig}). Sending interrupt to remote GPU processes...")
    
    # Try to kill GPU processes on the remote machine
    if public_dns and expanded_key_path:
        try:
            # Use nvidia-smi to find and kill GPU processes
            kill_cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-i", expanded_key_path,
                f"{SSH_USER}@{public_dns}",
                # The following command:
                # 1. Uses nvidia-smi to get PIDs of GPU processes
                # 2. Kills those processes with SIGKILL (-9)
                # 3. Falls back to stopping Docker containers if needed
                "echo 'Killing GPU processes...'; "
                "sudo nvidia-smi --query-compute-apps=pid --format=csv,noheader | "
                "xargs -r sudo kill -9 || echo 'No GPU processes found or failed to kill'; "
                "echo 'Stopping any related Docker containers...'; "
                f"docker ps --filter 'ancestor=*riskformer*' --format '{{{{.ID}}}}' | "
                f"xargs -r docker stop --time={DOCKER_STOP_TIMEOUT} || true; "
                f"docker ps --filter 'ancestor=*brca*' --format '{{{{.ID}}}}' | "
                f"xargs -r docker stop --time={DOCKER_STOP_TIMEOUT} || true; "
                "echo 'Forcing kill any remaining containers...'; "
                "docker ps --filter 'ancestor=*riskformer*' --format '{{.ID}}' | xargs -r docker kill || true; "
                "docker ps --filter 'ancestor=*brca*' --format '{{.ID}}' | xargs -r docker kill || true"
            ]
            logger.info("Sending kill commands to remote GPU processes...")
            subprocess.run(kill_cmd, timeout=15)
            logger.info("Kill commands sent to remote GPU processes.")
            
            # Wait a moment to let the commands take effect
            time.sleep(2)
            
            # Check if any relevant processes are still running and provide info
            status_cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-i", expanded_key_path,
                f"{SSH_USER}@{public_dns}",
                "echo 'Checking for GPU processes:'; "
                "nvidia-smi || echo 'Could not run nvidia-smi'; "
                "echo 'Checking for Docker containers:'; "
                "docker ps || echo 'No Docker containers running'"
            ]
            logger.info("Checking status of remote processes...")
            subprocess.run(status_cmd, timeout=10)
        except Exception as e:
            logger.error(f"Failed to send kill commands to remote processes: {e}")
    
    # Clean up the SSH process
    cleanup_ssh_process()
    
    # Only stop the EC2 instance if the flag was provided
    if stop_instance_on_exit and global_ec2_client:
        logger.info("--stop_instance flag was provided, stopping EC2 instance...")
        stop_instance(global_ec2_client)
    else:
        logger.info("Not stopping EC2 instance (--stop_instance flag not provided).")
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def stop_instance(ec2_client):
    try:
        logger.info(f"Stopping EC2 instance: {EC2_INSTANCE}")
        ec2_client.stop_instances(InstanceIds=[EC2_INSTANCE])

        waiter = ec2_client.get_waiter("instance_stopped")
        waiter.wait(
            InstanceIds=[EC2_INSTANCE],
            WaiterConfig={
                'Delay': WAITER_DELAY,
                'MaxAttempts': WAITER_TIMEOUT // WAITER_DELAY
            }
        )
        logger.info("Successfully stopped EC2 instance.")
    except Exception as e:
        logger.error(f"Failed to stop EC2 instance: {e}")


def boot_instance():
    try:
        ec2_client = boto3.client("ec2", region_name=REGION)
        logger.info("Successfully created EC2 client.")
    except Exception as e:
        logger.error(f"Couldn't create EC2 client: {e}")
        return None

    logger.info(f"Pulling instance info for {EC2_INSTANCE}...")
    try:
        desc = ec2_client.describe_instances(InstanceIds=[EC2_INSTANCE])
        if not desc["Reservations"] or not desc["Reservations"][0]["Instances"]:
            logger.error("No instance found with the specified ID")
            return None
            
        instance_state = desc["Reservations"][0]["Instances"][0]["State"]["Name"].lower()
        logger.info(f"Current instance state: {instance_state}")
    except Exception as e:
        logger.error(f"Couldn't pull EC2 instance state: {e}")
        return None

    if instance_state in ["stopped", "stopping"]:
        start_time = time.time()
        logger.info(f"Starting EC2 instance: {EC2_INSTANCE}")
        try:
            ec2_client.start_instances(InstanceIds=[EC2_INSTANCE])
        except Exception as e:
            logger.error(f"Failed to start EC2 instance: {e}")
            return None
            
        logger.info("Waiting for EC2 instance finish initializing...")
        waiter = ec2_client.get_waiter("instance_running")
        waiter.wait(
            InstanceIds=[EC2_INSTANCE],
            WaiterConfig={
                'Delay': WAITER_DELAY,
                'MaxAttempts': INSTANCE_STARTUP_TIMEOUT // WAITER_DELAY
            }
        )
        logger.info("Successfully started EC2 instance.")
        logger.info(f"Time elapsed: {(time.time() - start_time) / 60:.2f} minutes")
    else:
        logger.info(f"Instance is already in state '{instance_state}'. Skipping start.")
    return ec2_client


def wait_for_ssh(public_dns, expanded_key_path, max_attempts=10):
    """Wait for SSH to become available on the instance."""
    for attempt in range(max_attempts):
        try:
            ssh_cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=5",
                "-i", expanded_key_path,
                f"{SSH_USER}@{public_dns}",
                "echo", "SSH connection successful"
            ]
            subprocess.run(ssh_cmd, check=True, timeout=SSH_CONNECT_TIMEOUT)
            logger.info("SSH connection successful.")
            return True
        except Exception as e:
            if attempt < max_attempts - 1:
                logger.warning(f"SSH connection attempt {attempt + 1} failed: {e}")
                time.sleep(5)
            else:
                logger.error(f"Failed to connect to SSH after {max_attempts} attempts: {e}")
                return False


def get_ecr_token():
    """Get ECR authentication token."""
    try:
        logger.info("Getting ECR authentication token...")
        ecr_client = boto3.client('ecr', region_name=REGION)
        response = ecr_client.get_authorization_token()
        token = response['authorizationData'][0]['authorizationToken']
        # Decode the base64 token and remove the "AWS:" prefix
        decoded_token = base64.b64decode(token).decode('utf-8')
        if decoded_token.startswith('AWS:'):
            decoded_token = decoded_token[4:]  # Remove "AWS:" prefix
        logger.info("Successfully obtained ECR token")
        return decoded_token
    except Exception as e:
        logger.error(f"Failed to get ECR token: {e}")
        return None


def main():    
    global global_ec2_client, ssh_process, public_dns, expanded_key_path
    
    # Expand the key path at the start
    expanded_key_path = os.path.expanduser(IDENTITY_FILE)
    
    ### 1. Boot EC2 Instance ###
    logger.info("Creating EC2 client...")
    ec2_client = boot_instance()
    if ec2_client is None:
        logger.error("Failed to boot EC2 client. Exiting.")
        return None
    
    global_ec2_client = ec2_client
    
    ### 2. Collect DNS for SSH ###
    logger.info("Pulling instance DNS...")
    try:
        desc = ec2_client.describe_instances(InstanceIds=[EC2_INSTANCE])
        if not desc["Reservations"] or not desc["Reservations"][0]["Instances"]:
            logger.error("No instance found with the specified ID")
            return None
            
        public_dns = desc["Reservations"][0]["Instances"][0]["PublicDnsName"]
        logger.info(f"EC2 instance DNS: {public_dns}")
    except Exception as e:
        logger.error(f"Couldn't pull EC2 instance DNS: {e}")
        return None

    ### 3. Get AWS credentials ###
    logger.info("Getting AWS credentials...")
    try:
        # Get the current session's credentials
        session = boto3.Session()
        credentials = session.get_credentials().get_frozen_credentials()
        aws_access_key = credentials.access_key
        aws_secret_key = credentials.secret_key
        aws_session_token = credentials.token
        logger.info("Successfully obtained AWS credentials")
    except Exception as e:
        logger.error(f"Failed to get AWS credentials: {e}")
        return None

    ### 4. Get ECR token ###
    ecr_token = get_ecr_token()
    if not ecr_token:
        logger.error("Failed to get ECR token. Exiting.")
        return None

    ### 5. Connect to EC2 instance ###
    logger.info(f"Connecting to EC2 instance with key: {expanded_key_path}")
    
    # Wait for SSH to become available
    if not wait_for_ssh(public_dns, expanded_key_path):
        logger.error("Failed to establish SSH connection. Exiting.")
        return None
    
    ### 6. Run orchestrator script ###
    logger.info("Running orchestrator script...")
    
    # Construct the remote command with AWS credentials and ECR token
    remote_cmd = [
        "ssh",
        "-i", expanded_key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", f"ServerAliveInterval={SERVER_ALIVE_INTERVAL}",
        "-R", REMOTE_FORWARD,
        "-L", LOCAL_FORWARD_1,
        "-L", LOCAL_FORWARD_2,
        f"{SSH_USER}@{public_dns}",
        (f"export ECR_TOKEN='{ecr_token}' && "
         f"export AWS_ACCESS_KEY_ID='{aws_access_key}' && "
         f"export AWS_SECRET_ACCESS_KEY='{aws_secret_key}' && "
         f"export AWS_SESSION_TOKEN='{aws_session_token}' && "
         f"export AWS_DEFAULT_REGION='{REGION}' && "
         + " && ".join(REMOTE_COMMANDS))
    ]
    
    logger.info(f"SSH command: {' '.join(remote_cmd)}")
    ssh_process = subprocess.Popen(remote_cmd)
    ssh_process.wait()
    return ec2_client


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing on EC2 instance")
    parser.add_argument("--stop_instance", action="store_true", help="Stop the EC2 instance after running the script")
    args = parser.parse_args()
    
    # Set the global flag based on the command-line argument
    stop_instance_on_exit = args.stop_instance
    
    ec2_client = None
    try:
        ec2_client = main()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        # Clean up SSH process if it's still running
        cleanup_ssh_process()
        # Stop instance if requested
        if args.stop_instance and ec2_client:
            stop_instance(ec2_client)