import boto3
import time
import subprocess
import os
from datetime import datetime
import logging

DEBUG = True

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
    "cd ~/notebooks/brca_riskformer",
    "git pull origin main",
    "./orchestrators/run_preprocess.sh"
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

# 1. Boot EC2 Instance
ec2_client = boto3.client("ec2", region_name=REGION)
logger.info(f"Spinning up EC2 instance {EC2_INSTANCE} in {REGION}...")
desc = ec2_client.describe_instances(InstanceIds=[EC2_INSTANCE])
instance_state = desc["Reservations"][0]["Instances"][0]["State"]["Name"].lower()
logger.info(f"Current instance state: {instance_state}")

# Only start if stopped/stopping
if instance_state in ["stopped", "stopping"]:
    start_time = time.time()
    logger.info(f"Starting EC2 instance: {EC2_INSTANCE}")
    ec2_client.start_instances(InstanceIds=[EC2_INSTANCE])
    waiter = ec2_client.get_waiter("instance_running")
    waiter.wait(InstanceIds=[EC2_INSTANCE])
    logger.info("EC2 instance is now running.")
    logger.info(f"Time elapased: {(time.time() - start_time) / 60:.2f} minutes")
else:
    logger.info(f"Instance is already in state '{instance_state}'. Skipping start.")
logger.info("Successfully started EC2 instance.")

# 2. Connect by SSH & Run Orchestrator Script
desc = ec2_client.describe_instances(InstanceIds=[EC2_INSTANCE])
public_dns = desc["Reservations"][0]["Instances"][0]["PublicDnsName"]
logger.info(f"EC2 instance DNS: {public_dns}")

expanded_key_path = os.path.expanduser(IDENTITY_FILE)
logger.info("Connecting to EC2 instance via SSH and running orchestrator script...")
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
result = subprocess.run(ssh_cmd)
if result.returncode != 0:
    logger.error(f"SSH command failed with error: {result.stderr}")
else:
    logger.info(f"SSH command succeeded with output: {result.stdout}")
