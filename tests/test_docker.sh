#!/bin/bash

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

# Function to test container health
test_container_health() {
    echo "Testing container health..."
    
    # Test if container can start
    docker run --rm "$IMAGE_NAME" python3 --version
    
    # Test if GPU is accessible
    docker run --rm --gpus all "$IMAGE_NAME" nvidia-smi
    
    # Test if required system libraries are present
    docker run --rm "$IMAGE_NAME" ldconfig -p | grep -E "libopenslide|libtiff"
    
    echo "Container health check passed"
}

# Function to test Python dependencies
test_python_dependencies() {
    echo "Testing Python dependencies..."
    
    # Test if key Python packages are installed
    docker run --rm "$IMAGE_NAME" python3 -c "
import timm
import pydantic
import skimage
import scipy
import sklearn
import openslide
import histomicstk
import zarr
print('All required Python packages are installed')
"
    
    echo "Python dependencies check passed"
}

# Function to test volume mounts
test_volume_mounts() {
    echo "Testing volume mounts..."
    
    # Test if volumes can be mounted and accessed
    docker run --rm \
        -v "$SRC_DIR":"$WORKSPACE_ROOT/src" \
        -v "$RESOURCES_DIR":"$WORKSPACE_ROOT/resources" \
        -v "$CONFIGS_DIR":"$WORKSPACE_ROOT/configs" \
        -v "$MODELS_DIR":"$WORKSPACE_ROOT/models" \
        -v "$OUTPUTS_DIR":"$WORKSPACE_ROOT/outputs" \
        "$IMAGE_NAME" \
        ls -la "$WORKSPACE_ROOT"
    
    echo "Volume mounts check passed"
}

# Function to test basic command execution
test_basic_commands() {
    echo "Testing basic command execution..."
    
    # Test if we can run a simple Python script
    docker run --rm "$IMAGE_NAME" python3 -c "print('Basic command execution test passed')"
    
    # Test if we can access the workspace directory
    docker run --rm "$IMAGE_NAME" ls -la /opt/ml/processing
    
    echo "Basic command execution check passed"
}

# Function to test environment variables
test_environment_variables() {
    echo "Testing environment variables..."
    
    # Test if Python path is set correctly
    docker run --rm "$IMAGE_NAME" python3 -c "import sys; print(sys.path)"
    
    # Test if CUDA environment variables are set
    docker run --rm --gpus all "$IMAGE_NAME" env | grep -E "CUDA|NVIDIA"
    
    echo "Environment variables check passed"
}

# Main test execution
main() {
    echo "Starting Docker container tests..."
    
    test_container_health
    test_python_dependencies
    test_volume_mounts
    test_basic_commands
    test_environment_variables
    
    echo "All Docker container tests passed successfully"
}

# Run main function
main 