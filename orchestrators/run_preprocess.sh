#!/bin/bash

# ==============================
# Configuration Variables
# ==============================

# Exit script on error and print each command
set -e
set -x

# Docker image name
IMAGE_NAME="651340551631.dkr.ecr.us-east-1.amazonaws.com/brca-riskformer/pytorch-svs-preprocess"
# IMAGE_NAME="pytorch-svs-preprocess"

# Root directory of the project
PROJECT_ROOT="/home/ec2-user/brca_riskformer"

# Paths to data and configs
RESOURCES_DIR="$PROJECT_ROOT/resources"
CONFIGS_DIR="$PROJECT_ROOT/configs"
OUTPUTS_DIR="$PROJECT_ROOT/outputs"
RISKFORMER_DIR="$PROJECT_ROOT/riskformer"
ENTRYPOINTS_DIR="$PROJECT_ROOT/entrypoints"
ORCHESTRATORS_DIR="$PROJECT_ROOT/orchestrators"
LOGS_DIR="$PROJECT_ROOT/logs"

# Specific files
WORKSPACE_ROOT="/opt/ml/processing"
METADATA="$WORKSPACE_ROOT/resources/riskformer_slide_samples.json"
FOREGROUND_CONFIG="$WORKSPACE_ROOT/configs/preprocessing_foreground_config.yaml"
FOREGROUND_CLEANUP_CONFIG="$WORKSPACE_ROOT/configs/preprocessing_foreground_cleanup_config.yaml"
TILING_CONFIG="$WORKSPACE_ROOT/configs/preprocessing_tiling_config.yaml"

# Model details
MODEL_BUCKET="tcga-riskformer-preprocessing-models"
MODEL_TYPE="uni"
MODEL_KEY="uni/uni2-h"

# Processing parameters
BATCH_SIZE=64
NUM_WORKERS=2
PREFETCH_FACTOR=2
DEBUG_FLAG="--debug"
STOP_ON_FAIL="--stop_on_fail"


# AWS credentials
PROFILE="651340551631_AWSPowerUserAccess"
DATA_BUCKET="tcga-riskformer-data-2025"
REGION="us-east-1"
INPUT_DIR="raw"
OUTPUT_DIR="preprocessed"
AWS_CREDENTIALS="$HOME/.aws/credentials"
INSTNCE_ID="i-08a58080616278d9c"

# ==============================
# Run the Docker Container
# ==============================

docker run --rm --gpus all --runtime=nvidia\
    --user root \
    --ipc=host \
    --memory=0 \
    --privileged \
    --cap-add=SYS_ADMIN --cap-add=SYS_RAWIO \
    --device=/dev/nvidiactl --device=/dev/nvidia0 \
    --device=/dev/nvidia-modeset --device=/dev/nvidia-uvm \
    -v "$AWS_CREDENTIALS":"/root/.aws/credentials" \
    -v "$RISKFORMER_DIR":"$WORKSPACE_ROOT/riskformer" \
    -v "$ENTRYPOINTS_DIR":"$WORKSPACE_ROOT/entrypoints" \
    -v "$ORCHESTRATORS_DIR":"$WORKSPACE_ROOT/orchestrators" \
    -v "$RESOURCES_DIR":"$WORKSPACE_ROOT/resources" \
    -v "$CONFIGS_DIR":"$WORKSPACE_ROOT/configs" \
    -v "$LOGS_DIR":"$WORKSPACE_ROOT/logs" \
    -w "$WORKSPACE_ROOT" \
    "$IMAGE_NAME" \
    -m orchestrators.run_preprocess \
        --profile "$PROFILE" \
        --bucket "$DATA_BUCKET" \
        --region "$REGION" \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --metadata_file "$METADATA" \
        --foreground_config "$FOREGROUND_CONFIG" \
        --foreground_cleanup_config "$FOREGROUND_CLEANUP_CONFIG" \
        --tiling_config "$TILING_CONFIG" \
        --model_type "$MODEL_TYPE" \
        --model_bucket "$MODEL_BUCKET" \
        --model_key "$MODEL_KEY" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --prefetch_factor "$PREFETCH_FACTOR" \
        $STOP_ON_FAIL \
        $DEBUG_FLAG \