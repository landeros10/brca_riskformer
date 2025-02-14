#!/bin/bash

# ==============================
# Configuration Variables
# ==============================

# Exit script on error and print each command
set -e
set -x

# Docker image name
IMAGE_NAME="pytorch-svs-preprocess"

# Root directory of the project
PROJECT_ROOT=~/notebooks/brca_riskformer

# Paths to data and configs
RESOURCES_DIR="$PROJECT_ROOT/resources"
CONFIGS_DIR="$PROJECT_ROOT/configs"
MODELS_DIR="$PROJECT_ROOT/models"
OUTPUTS_DIR="$PROJECT_ROOT/outputs"
SRC_DIR="$PROJECT_ROOT/src"

# Specific files
WORKSPACE_ROOT="/workspace"
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
BATCH_SIZE=8
DEBUG_FLAG="--debug"


# ==============================
# Run the Docker Container
# ==============================

docker run --rm \
    --user root \
    -v "$SRC_DIR":"$WORKSPACE_ROOT/src" \
    -v "$RESOURCES_DIR":"$WORKSPACE_ROOT/resources" \
    -v "$CONFIGS_DIR":"$WORKSPACE_ROOT/configs" \
    -v "$MODELS_DIR":"$WORKSPACE_ROOT/models" \
    -v "$OUTPUTS_DIR":"$WORKSPACE_ROOT/outputs" \
    -w "$WORKSPACE_ROOT" \
    "$IMAGE_NAME" \
    -m src.scripts.preprocess \
        --foreground_config "$FOREGROUND_CONFIG" \
        --foreground_cleanup_config "$FOREGROUND_CLEANUP_CONFIG" \
        --tiling_config "$TILING_CONFIG" \
        --model_type "$MODEL_TYPE" \
        --model_dir "$MODEL_DIR" \
        --input_filename "$TEST_SVS_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --num_workers "$NUM_WORKERS" \
        --prefetch_factor "$PREFETCH_FACTOR" \
        $DEBUG_FLAG \
        --batch_size "$BATCH_SIZE"
