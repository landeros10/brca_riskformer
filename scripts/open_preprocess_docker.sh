#!/bin/bash

# ==============================
# Configuration Variables
# ==============================

# Exit script on error and print each command
set -e
set -x

# Docker image name (using ECR URL for consistency with orchestrator)
IMAGE_NAME="651340551631.dkr.ecr.us-east-1.amazonaws.com/brca-riskformer/pytorch-svs-preprocess"

# Root directory of the project
PROJECT_ROOT="$(pwd)"

# Paths to data and configs
RESOURCES_DIR="$PROJECT_ROOT/resources"
CONFIGS_DIR="$PROJECT_ROOT/configs"
MODELS_DIR="$PROJECT_ROOT/models"
OUTPUTS_DIR="$PROJECT_ROOT/outputs"
SRC_DIR="$PROJECT_ROOT/src"

# Workspace configuration
WORKSPACE_ROOT="/workspace"

# Test configuration
TEST_SVS_FILE="$WORKSPACE_ROOT/resources/TCGA-Z7-A8R6-01Z-00-DX1.CE4ED818-D762-4324-9DEA-2ACB38B9B0B9.svs"
FOREGROUND_CONFIG="$WORKSPACE_ROOT/configs/preprocessing_foreground_config.yaml"
FOREGROUND_CLEANUP_CONFIG="$WORKSPACE_ROOT/configs/preprocessing_foreground_cleanup_config.yaml"
TILING_CONFIG="$WORKSPACE_ROOT/configs/preprocessing_tiling_config.yaml"
OUTPUT_DIR="$WORKSPACE_ROOT/outputs/"

# Model details
MODEL_TYPE="uni"
MODEL_DIR="$WORKSPACE_ROOT/models/uni2-h/"

# Processing parameters
NUM_WORKERS=1
PREFETCH_FACTOR=1
BATCH_SIZE=1
DEBUG_FLAG="--debug"

# Interactive mode for debugging
docker run --rm --gpus all --runtime=nvidia \
    --user root \
    --ipc=host \
    --memory=0 \
    --privileged \
    --cap-add=SYS_ADMIN --cap-add=SYS_RAWIO \
    --device=/dev/nvidiactl --device=/dev/nvidia0 \
    --device=/dev/nvidia-modeset --device=/dev/nvidia-uvm \
    -it --entrypoint /bin/bash \
    -v "$SRC_DIR":"$WORKSPACE_ROOT/src" \
    -v "$RESOURCES_DIR":"$WORKSPACE_ROOT/resources" \
    -v "$CONFIGS_DIR":"$WORKSPACE_ROOT/configs" \
    -v "$MODELS_DIR":"$WORKSPACE_ROOT/models" \
    -v "$OUTPUTS_DIR":"$WORKSPACE_ROOT/outputs" \
    -w "$WORKSPACE_ROOT" \
    "$IMAGE_NAME"