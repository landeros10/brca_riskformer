import boto3
import time
import subprocess
import os
import argparse
from datetime import datetime
import logging
import signal

DEBUG = True

AWS_CREDS = "~/.aws/credentials"

EC2_INSTANCE = "i-08a58080616278d9c"
REGION = "us-east-1"
PROFILE = "651340551631_AWSPowerUserAccess"

IDENTITY_FILE = "~/Downloads/clawsec2.pem"
SSH_USER = "ec2-user"
SERVER_ALIVE_INTERVAL = 115
REMOTE_FORWARD = "52698:localhost:52698"
LOCAL_FORWARD_1 = "8888:localhost:8888"
LOCAL_FORWARD_2 = "16006:localhost:6006"

REMOTE_COMMANDS = [
    "export AWS_SHARED_CREDENTIALS_FILE=~/.aws/credentials",
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


def stop_instance():
    try:
        logger.info(f"Stopping EC2 instance: {EC2_INSTANCE}")
        ec2_client.stop_instances(InstanceIds=[EC2_INSTANCE])

        waiter = ec2_client.get_waiter("instance_stopped")
        waiter.wait(InstanceIds=[EC2_INSTANCE])
        logger.info("Successfully stopped EC2 instance.")
    except Exception as e:
        logger.error(f"Failed to stop EC2 instance: {e}")


def boot_instance():
    try:
        ec2_client = boto3.client("ec2", region_name=REGION)
        logger.info("Successfully created EC2 client.")
    except Exception as e:
        logger.error(f"Couldn't create EC2 client: {e}")
        return

    logger.info(f"Pulling instance info for {EC2_INSTANCE}...")
    try:
        desc = ec2_client.describe_instances(InstanceIds=[EC2_INSTANCE])
        instance_state = desc["Reservations"][0]["Instances"][0]["State"]["Name"].lower()
        logger.info(f"Current instance state: {instance_state}")
    except Exception as e:
        logger.error(f"Couldn't pull EC2 instance state: {e}")
        return ec2_client

    if instance_state in ["stopped", "stopping"]:
        start_time = time.time()
        logger.info(f"Starting EC2 instance: {EC2_INSTANCE}")
        try:
            ec2_client.start_instances(InstanceIds=[EC2_INSTANCE])
        except Exception as e:
            logger.error(f"Failed to start EC2 instance: {e}")
            return ec2_client
        logger.info("Waiting for EC2 instance finish initializing...")
        waiter = ec2_client.get_waiter("instance_running")
        waiter.wait(InstanceIds=[EC2_INSTANCE])
        logger.info("Successfully started EC2 instance.")
        logger.info(f"Time elapased: {(time.time() - start_time) / 60:.2f} minutes")
    else:
        logger.info(f"Instance is already in state '{instance_state}'. Skipping start.")
    return ec2_client


def main():    
    ### 1. Boot EC2 Instance ###
    logger.info("Creating EC2 client...")
    ec2_client = boot_instance()

    ### 2.Collect DNS for SSH ###
    logger.info("Pulling instance DNS...")
    try:
        desc = ec2_client.describe_instances(InstanceIds=[EC2_INSTANCE])
        public_dns = desc["Reservations"][0]["Instances"][0]["PublicDnsName"]
        logger.info(f"EC2 instance DNS: {public_dns}")
    except Exception as e:
        logger.error(f"Couldn't pull EC2 instance DNS: {e}")
        return ec2_client
    
    ### 3. Copy AWS credentials to EC2 instance ###
    logger.info("Copying AWS credentials to EC2 instance...")
    expanded_key_path = os.path.expanduser(IDENTITY_FILE)
    expanded_creds = os.path.expanduser(AWS_CREDS)
    if os.path.exists(expanded_creds):
        scp_cmd = [
            "scp",
            "-i", expanded_key_path,
            expanded_creds,
            f"{SSH_USER}@{public_dns}:/home/ec2-user/.aws/credentials"
        ]
        try:
            subprocess.run(scp_cmd, check=True)
            logger.info("AWS credentials successfully uploaded.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to upload AWS credentials: {e}")
            return ec2_client
    else:
        logger.error(f"AWS credentials file not found: {AWS_CREDS}")
        return ec2_client
            
    ### 4. SSH into EC2 instance and run script ###
    logger.info(f"Connecting to EC2 instance with key: {expanded_key_path}")
    logger.info("Running orchestrator script...")
    ssh_cmd = [
        "ssh",
        "-i", expanded_key_path,
        "-o", f"ServerAliveInterval={SERVER_ALIVE_INTERVAL}",
        "-R", REMOTE_FORWARD,
        "-L", LOCAL_FORWARD_1,
        "-L", LOCAL_FORWARD_2,
        f"{SSH_USER}@{public_dns}",
        " && ".join(REMOTE_COMMANDS),
    ]
    logger.info(f"SSH command: {' '.join(ssh_cmd)}")
    ssh_process = subprocess.Popen(ssh_cmd)
    def handle_interrupt(signum, frame):
        """Handle CTRL+C and terminate SSH process."""
        logger.info("CTRL+C detected. Terminating SSH connection...")
        ssh_process.terminate()  # Gracefully terminate SSH
        ssh_process.wait()  # Wait for process to exit
        logger.info("SSH connection closed. Exiting.")
        exit(0)
    signal.signal(signal.SIGINT, handle_interrupt)
    ssh_process.wait()  # Wait for the SSH process to finish
    return ec2_client


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing on EC2 instance")
    parser.add_argument("--stop_instance", action="store_true", help="Stop the EC2 instance after running the script")
    args = parser.parse_args()

    ec2_client = None
    try:
        ec2_client = main()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        if args.stop_instance and ec2_client:
            stop_instance(ec2_client)