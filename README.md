# EnginBERT

EnginBERT is a domain-specific BERT model designed to create high-quality text embeddings for engineering literature. It's specifically trained on engineering research papers to better understand technical and scientific content.

## Features

- Domain-specific BERT model trained on engineering papers
- Automated data collection from arXiv
- Custom preprocessing pipeline for academic papers
- Fine-tuning with Masked Language Modeling (MLM)
- Sequence classification capabilities
- Evaluation metrics for clustering, citations, and information retrieval
- Command-line interface for easy training and evaluation

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Basic Installation

To install EnginBERT locally, use:

```bash
pip install -e .
```

### Development Installation

For development purposes, install with additional dependencies:

```bash
pip install -e ".[dev]"
```

This includes testing, linting, and development tools.

### Documentation Installation

To build and work with the documentation:

```bash
pip install -e ".[docs]"
```

## Usage

EnginBERT provides a convenient CLI for all major operations:

### Training

Train the model from scratch:
```bash
enginbert train
```

Skip specific training steps:
```bash
enginbert train --skip data preprocess
```

### Evaluation

Run all evaluation metrics:
```bash
enginbert evaluate
```

Run specific evaluation metrics:
```bash
enginbert evaluate --metrics clustering ir citations
```

### Complete Pipeline

Run the entire pipeline (training and evaluation):
```bash
enginbert run-all
```

## Project Structure

```
EnginBERT/
├── scripts/
│   ├── data_processing/    # Data collection and preprocessing
│   ├── evaluation_metrics/ # Model evaluation tools
│   ├── helpers/           # Utility functions
│   ├── tokenizer/         # Custom tokenization
│   └── train/            # Training scripts
```

## Development

### Code Style

The project follows the Black code style. To format your code:

```bash
black .
```

### Linting

Run flake8 for code quality checks:

```bash
flake8 .
```


## License

MIT license


## Contact

Jan Heimann - jan_heimann@icloud.com
Tristan Cruise

