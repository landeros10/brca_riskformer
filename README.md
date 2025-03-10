# BRCA RiskFormer

## Project Overview

RiskFormer is a deep learning framework for predicting breast cancer risk from histopathology whole slide images (WSIs) using transformer-based architectures. This project addresses the critical clinical need for more accurate risk prediction models to improve screening and early detection of breast cancer.


Breast cancer risk prediction is essential for early intervention, especially for BRCA1/2 mutation carriers who face elevated lifetime risk (45-87%). While genetic testing approaches like Oncotype DX™ are standard for risk stratification, they remain costly and inaccessible in many settings. RiskFormer leverages deep learning to analyze histopathology images directly, offering a potentially cost-effective alternative that captures tissue-level morphological patterns associated with cancer development.

## Technical Approach

RiskFormer employs a hierarchical architecture specifically designed to handle the extreme size of whole slide images (often exceeding 100,000 × 100,000 pixels) while capturing relevant features at multiple scales:

### 1. **Pre-processing Pipeline**:
Converts gigapixel WSIs into manageable representations through tissue segmentation, patch extraction, and uses pre-trained vision models to extract high-dimensional feature representations from tissue patches.
<div align="center">
  <img src="docs/images/f1.png" width="80%" alt="RiskFormer Pipeline Overview">
  <p><em style="font-size: 0.9em;">Figure 1: Patient Slide Pre-Processing Pipeline</em></p>
</div>

The workflow consists of:
- **Tissue Segmentation**: Isolates relevant tissue regions from the slide background
- **Patch Extraction**: Splits identified tissue into smaller image tiles at high resolution
- **Feature Embedding**: Processes tiles through a pre-trained encoder to create variably sized arrays of tile embeddings
- **Region Formatting**: Splits and/or pads the embedding arrays into uniformly sized regions for consistent processing


### 2. **Hierarchical Transformer Architecture**:
Implements a multi-scale transformer designed to handle the complex spatial relationships in whole slide images. Each patient is treated as a batch of "large-scale regions". Each large-scale region is analyzed by a transformer for intra-region analysis, and attention pooling is used to conduct inter-region analysis (between distant large-scale regions in the slide).
<div align="center">
  <img src="docs/images/f2.png" width="80%" alt="Transformer Architecture">
  <p><em style="font-size: 0.9em;">Figure 2: Risk Prediction Model Architecture.</em></p>
</div>

The workflow consists of:
- **Dimensionality Reduction**: phi (φ) to standardize embedding dimensions
- **Multi-Scale Processing**: Deploys specialized transformer blocks with convolution operations to consolidate features spatially.
- **Feature Consolidation**: Concatenates average and maximum region-level pooling of transformed token arrays to capture both typical and salient features
- **Attention Mechanism**: Implements an attention-weighted averaging system where region embeddings receive learned attention weights, enabling the model to focus on the most informative regions
- **Dual Prediction Paths**: Generates both region-level and patient-level risk scores, with the final score derived from attention-weighted embeddings


### 3. **Risk Assessment & Visualization**:
The model produces an overall risk prediction on a scale from 0 to 1, which correlates with recurrence risk categories used in genetic tests like Oncotype DX™. 

<div align="center">
  <img src="docs/images/f4.png" width="80%" alt="Feature Visualization">
  <p><em style="font-size: 0.9em;">Figure 3: Visualizing High-Risk Regions in sample slides.</em></p>
</div>

The model also uses multiple explainability methods to identify high-risk areas in slides. These visualization methods include: 
- **Tile dropout**: Measures which region occlusions lead to reduced risk outputs
- **Region-level prediction**: These are sub-slide predictions directly integrated into the architecture of the model
- **Attention maps**: Combines fine-scale transformer block attention weights with region-level attention weights from the attention-pooling step.

## Notebooks

> ⚠️ **Coming Soon**: The following notebooks are under development and will be available in the near future.

The `notebooks/` directory will contain Jupyter notebooks that demonstrate key functionality of the RiskFormer pipeline:

### Dataset Exploration
- **`01_dataset_loading.ipynb`**: Demonstrates how to load and preprocess whole slide images (WSIs) for the RiskFormer pipeline
- **`02_embedding_visualization.ipynb`**: Visualizes tile embeddings and explains their spatial organization

### Model Usage
- **`03_simple_inference.ipynb`**: Step-by-step walkthrough of running inference on new WSIs
- **`04_risk_visualization.ipynb`**: Examples of generating and interpreting risk heatmaps from model outputs

## Implementation

The project is implemented in PyTorch with comprehensive MLOps integration:

- **Distributed Training**: Support for multi-GPU and cloud-based training on AWS
- **Data Pipeline**: Efficient data loading and preprocessing of large whole slide images
- **Model Versioning**: Tracking of experiments and model iterations
- **Containerization**: Docker-based deployment for consistent environments
- **Cloud Integration**: AWS batch processing for scalable inference


### Project Status Disclaimer

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

### Project Structure

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


## Usage

### Preprocessing

The preprocessing pipeline handles data preparation and feature engineering:

```bash
python entrypoints/preprocess.py test.svs --config configs/preprocessing/config.yaml 
```

### Training

Train the RiskFormer model:

```bash
python entrypoints/train.py --config configs/training/config.yaml
```

### Inference

Generate predictions using trained models:

```bash
python entrypoints/inference.py test.svs
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue in the GitHub repository.