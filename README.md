# BRCA RiskFormer

A deep learning model for predicting BRCA-related cancer risks using transformer architecture. This project implements a novel approach to cancer risk prediction using transformer-based models to analyze genetic and clinical data.

## Project Status Disclaimer

**⚠️ Work in Progress:** This repository represents ongoing research work that is being refactored for deployment standards. The codebase is currently transitioning from a research prototype to an implementation ready for production use.

**Current Implementation Status:**
- ✅ Data preprocessing pipeline is implemented and functional
- ✅ Basic AWS infrastructure integration (S3, EC2) is set up
- ✅ Docker containerization is configured
- ✅ Core transformer model architecture is defined
- ⚠️ Training pipeline is partially implemented
- ⚠️ MLOps deployment infrastructure is partially implemented
- ❌ Inference API is not yet implemented
- ❌ CI/CD pipeline is not yet set up

This project demonstrates MLOps practices and AWS cloud integration for large-scale model training and deployment, though some components are still under active development.

## Security Notice

This repository uses AWS infrastructure and requires appropriate credentials for operation. When using this code:

- **Do not** hard-code any AWS credentials directly in the source code
- **Do not** commit `.env` files or credential files to the repository
- **Do** use AWS credential provider chain (environment variables, AWS profiles, or IAM roles)
- **Do** review code for sensitive information before committing


## Project Structure

```
brca_riskformer/
│── aws/                # AWS infrastructure components
│   ├── config.json     # AWS configuration
│   ├── lambdas/        # Lambda functions
│   ├── ec2/            # EC2 scripts
│   └── *.sh            # AWS deployment scripts
│── bin/                 # Job execution scripts
│── configs/             # Configuration files
│   ├── aws/            # AWS-related configurations
│   ├── preprocessing/  # Preprocessing configurations
│   ├── training/       # Model training configurations
│   ├── inference/      # Inference configurations
│── entrypoints/        # Main workflow scripts
│── logs/               # Execution logs
│── models/             # Trained model checkpoints
│── notebooks/          # Jupyter notebooks
│   ├── experiments/    # Training experiments
│   ├── testing/        # Debugging notebooks
│── orchestrators/      # Job orchestration scripts
│   ├── batch/          # AWS Batch job definitions
│── resources/          # Static dataset files
│── riskformer/         # Core package
│   ├── data/          # Dataset operations
│   ├── training/      # Training logic
│   ├── utils/         # Utility functions
│── scripts/           # Standalone scripts
│── testing/           # Local testing scripts
│── tests/             # Unit and integration tests
│── docker/            # Docker-related files
│── docs/              # Documentation
```

## Features

- Transformer-based architecture for cancer risk prediction
- Support for processing genetic and clinical data
- AWS Batch integration for scalable processing
- Comprehensive preprocessing pipeline
- Docker containerization for consistent environments
- Extensive testing and documentation

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brca-riskformer.git
   cd brca-riskformer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Usage

### Preprocessing

The preprocessing pipeline handles data preparation and feature engineering:

```bash
python entrypoints/preprocess.py --config configs/preprocessing/config.yaml
```

### Training

Train the RiskFormer model:

```bash
python entrypoints/train.py --config configs/training/config.yaml
```

### Inference

Generate predictions using trained models:

```bash
python entrypoints/inference.py --config configs/inference/config.yaml
```

## Development

### Testing

Run tests:
```bash
python -m pytest tests/
```

## Docker

Build the Docker image:
```bash
docker build -t brca-riskformer -f docker/Dockerfile .
```

Run the container:
```bash
docker run -it brca-riskformer
```

## AWS Integration

The project includes AWS integration for scalable processing:

### SQS-Based Job Processing

The project uses AWS SQS for job queue management:

```bash
# Create SQS queue
bash aws/create_sqs_queue.sh
```

### Lambda Functions for Event-Driven Processing

Lambda functions are used to process S3 events:

```bash
# Deploy Lambda function
bash aws/lambdas/deploy_lambda.sh svs_processor
```

### S3 Event Triggers

S3 buckets are configured to trigger Lambda functions:

```bash
# Configure S3 trigger
bash aws/configure_s3_trigger.sh
```

For more details on AWS components, see [aws/README.md](aws/README.md).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue in the GitHub repository.
