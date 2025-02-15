#!/bin/bash
# Docker image name
IMAGE_NAME="651340551631.dkr.ecr.us-east-1.amazonaws.com/brca-riskformer/pytorch-svs-preprocess"

docker run --rm --gpus all --runtime=nvidia\
    --user root \
    --ipc=host \
    --memory=0 \
    --privileged \
    --cap-add=SYS_ADMIN --cap-add=SYS_RAWIO \
    --device=/dev/nvidiactl --device=/dev/nvidia0 \
    --device=/dev/nvidia-modeset --device=/dev/nvidia-uvm \
    -it --entrypoint /bin/bash "651340551631.dkr.ecr.us-east-1.amazonaws.com/brca-riskformer/pytorch-svs-preprocess"
