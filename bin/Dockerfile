# Use AWS Deep Learning Container for PyTorch as the base
FROM public.ecr.aws/sagemaker/sagemaker-distribution:2.3-gpu

# Switch to root user to install system dependencies
USER root

# Install system dependencies required for HistomicsTK and large_image
RUN apt-get update && apt-get install -y \
    libopenslide-dev \
    libtiff-dev \
    nvidia-utils-535 \
    nvidia-settings \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies, including OpenSlide and HistomicsTK
RUN pip install --upgrade pip && \
    pip install \
    timm \
    pydantic \
    scikit-image \
    scipy \
    scikit-learn \
    openslide-bin \
    histomicstk --find-links https://girder.github.io/large_image_wheels \
    zarr==2.18.3

# Set the working directory inside the container (SageMaker mounts script and data here)
WORKDIR /opt/ml/processing

# Set the entrypoint for SageMaker Processing Job
ENTRYPOINT ["python3"]