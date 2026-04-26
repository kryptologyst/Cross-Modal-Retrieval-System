# Cross-Modal Retrieval System

A research-ready implementation of cross-modal retrieval using CLIP-based models for image-text matching and retrieval tasks.

## Overview

This project implements a sophisticated cross-modal retrieval system that enables:
- **Image → Text Retrieval**: Find relevant text descriptions for given images
- **Text → Image Retrieval**: Find relevant images for given text queries
- **Bidirectional Search**: Seamless switching between modalities
- **Real-time Demo**: Interactive Streamlit interface for testing

## Features

- **Modern Architecture**: Enhanced CLIP model with additional projection layers
- **Contrastive Learning**: InfoNCE loss with temperature scaling
- **Comprehensive Evaluation**: Recall@K, mAP, median rank metrics
- **Device Support**: Automatic CUDA → MPS → CPU fallback
- **Reproducible**: Deterministic seeding and configuration management
- **Interactive Demo**: Streamlit interface with visualization
- **Production Ready**: Clean code, type hints, comprehensive testing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Cross-Modal-Retrieval-System.git
cd Cross-Modal-Retrieval-System

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pre-commit install
```

### Dataset Setup

The system automatically creates a toy dataset for testing. For real datasets, organize your data as follows:

```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations.json
```

The `annotations.json` should contain:
```json
{
  "train": [
    {"image": "image1.jpg", "text": "A beautiful landscape"},
    {"image": "image2.jpg", "text": "A cute cat playing"}
  ],
  "val": [...],
  "test": [...]
}
```

### Training

```bash
# Train with default configuration
python scripts/train.py

# Train with custom configuration
python scripts/train.py --config configs/custom_config.yaml
```

### Evaluation

```bash
# Run evaluation on test set
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
```

### Demo

```bash
# Launch interactive demo
streamlit run demo/app.py
```

## Project Structure

```
0928_Cross-modal_Retrieval_System/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architectures
│   ├── losses/            # Loss functions
│   ├── eval/              # Evaluation metrics
│   ├── viz/               # Visualization utilities
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── demo/                  # Interactive demo
├── tests/                 # Unit tests
├── assets/                # Generated artifacts
├── data/                  # Dataset directory
├── checkpoints/           # Model checkpoints
└── logs/                  # Training logs
```

## Configuration

The system uses YAML configuration files. Key parameters:

```yaml
model:
  name: "clip-vit-base-patch32"
  temperature: 0.07
  projection_dim: 512

data:
  batch_size: 32
  max_samples: 1000

training:
  epochs: 10
  learning_rate: 1e-4
  weight_decay: 0.01
```

## Model Architecture

The system extends CLIP with:

1. **Dual Encoders**: Separate image and text encoders
2. **Enhanced Projections**: Additional projection layers for better representations
3. **Contrastive Learning**: InfoNCE loss with temperature scaling
4. **Cross-Modal Alignment**: Shared embedding space for both modalities

## Evaluation Metrics

- **Recall@K**: Percentage of queries where correct result is in top-K
- **Mean Average Precision (mAP)**: Average precision across all queries
- **Median Rank**: Median rank of correct results
- **Cross-Modal**: Separate metrics for Image→Text and Text→Image retrieval

## Demo Features

The Streamlit demo provides:

- **Interactive Search**: Upload images or enter text queries
- **Real-time Results**: Instant retrieval with similarity scores
- **Visualization**: Display of retrieved images and descriptions
- **Gallery Browser**: Browse the dataset and test queries
- **Performance Metrics**: Display of model performance

## Safety and Limitations

### Disclaimer

This is a research/educational project. The system:

- May not be accurate for all types of queries
- Should not be used for critical decision-making
- Requires verification of important information
- May have biases present in training data

### Use Cases

Suitable for:
- Research and education
- Prototype development
- Content recommendation systems
- Image annotation tools

Not suitable for:
- Medical diagnosis
- Legal decisions
- Security applications
- Critical infrastructure

## Development

### Code Quality

- **Type Hints**: Full type annotation coverage
- **Documentation**: Google-style docstrings
- **Formatting**: Black + Ruff for consistent code style
- **Testing**: Comprehensive unit tests

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Run linting and tests
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with CLIP model licensing terms.

## Acknowledgments

- OpenAI CLIP model and team
- Hugging Face Transformers library
- Streamlit for the demo interface
- PyTorch ecosystem for deep learning tools
# Cross-Modal-Retrieval-System
