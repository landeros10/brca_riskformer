# BRCA RiskFormer

A deep learning model for predicting BRCA-related cancer risks using transformer architecture. This project implements a novel approach to cancer risk prediction using transformer-based models to analyze genetic and clinical data.

## Project Structure

```
brca_riskformer/
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

The project includes AWS Batch integration for scalable processing:

- Job definitions are located in `orchestrators/batch/`
- AWS configurations are in `configs/aws/`
- Batch processing scripts are in `bin/`

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
