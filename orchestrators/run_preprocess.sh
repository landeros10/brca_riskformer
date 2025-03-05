#!/bin/bash

# ==============================
# Configuration Variables
# ==============================

# Exit script on error and print each command
set -e
set -x
cleanup() {
    docker logout "$ECR_ID" || true
}
trap cleanup EXIT

YQ_VERSION=$(yq --version)
echo "yq version: $YQ_VERSION"

# Root directory of the project
PROJECT_ROOT="/home/ec2-user/brca_riskformer"
CONFIG_FILE="$PROJECT_ROOT/configs/preprocessing/ec2_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at $CONFIG_FILE"
    exit 1
fi

# AWS credentials
PROFILE=$(yq '.aws.profile' "$CONFIG_FILE")
ECR_ID=$(yq '.aws.ecr_id' "$CONFIG_FILE")
REGION=$(yq '.aws.region' "$CONFIG_FILE")

if [ -z "$PROFILE" ] || [ -z "$ECR_ID" ] || [ -z "$REGION" ]; then
    echo "Error: Required AWS configuration not found in $CONFIG_FILE"
    exit 1
fi

# Export AWS profile and region
export AWS_PROFILE="$PROFILE"
export AWS_DEFAULT_REGION="$REGION"

# Docker image name
IMAGE_NAME="$ECR_ID/$(yq '.docker.image_name' "$CONFIG_FILE")"
WORKSPACE_ROOT=$(yq '.docker.workspace_root' "$CONFIG_FILE")

# Paths to data and configs
RESOURCES_DIR="$PROJECT_ROOT/$(yq '.project.directories.resources' "$CONFIG_FILE")"
CONFIGS_DIR="$PROJECT_ROOT/$(yq '.project.directories.configs' "$CONFIG_FILE")"
OUTPUTS_DIR="$PROJECT_ROOT/$(yq '.project.directories.outputs' "$CONFIG_FILE")"
RISKFORMER_DIR="$PROJECT_ROOT/$(yq '.project.directories.riskformer' "$CONFIG_FILE")"
ENTRYPOINTS_DIR="$PROJECT_ROOT/$(yq '.project.directories.entrypoints' "$CONFIG_FILE")"
ORCHESTRATORS_DIR="$PROJECT_ROOT/$(yq '.project.directories.orchestrators' "$CONFIG_FILE")"
LOGS_DIR="$PROJECT_ROOT/$(yq '.project.directories.logs' "$CONFIG_FILE")"

# ==============================
# Run the Docker Container
# ==============================
# Use the provided ECR token
if [ -z "$ECR_TOKEN" ]; then
    echo "Error: ECR_TOKEN environment variable not set"
    exit 1
fi

echo "$ECR_TOKEN" | docker login --username AWS --password-stdin "$ECR_ID"
docker pull "$IMAGE_NAME"

docker run --rm --gpus all --runtime=nvidia\
    --user root \
    --ipc=host \
    --memory=0 \
    --cpus="$(nproc)" \
    --privileged \
    --cap-add=SYS_ADMIN --cap-add=SYS_RAWIO \
    --device=/dev/nvidiactl --device=/dev/nvidia0 \
    --device=/dev/nvidia-modeset --device=/dev/nvidia-uvm \
    -e AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY \
    -e AWS_SESSION_TOKEN \
    -e AWS_DEFAULT_REGION \
    -v "$RISKFORMER_DIR":"$WORKSPACE_ROOT/riskformer" \
    -v "$ENTRYPOINTS_DIR":"$WORKSPACE_ROOT/entrypoints" \
    -v "$ORCHESTRATORS_DIR":"$WORKSPACE_ROOT/orchestrators" \
    -v "$RESOURCES_DIR":"$WORKSPACE_ROOT/resources" \
    -v "$CONFIGS_DIR":"$WORKSPACE_ROOT/configs" \
    -v "$LOGS_DIR":"$WORKSPACE_ROOT/logs" \
    -w "$WORKSPACE_ROOT" \
    "$IMAGE_NAME" \
    -m orchestrators.run_preprocess \
    --config "$CONFIG_FILE"