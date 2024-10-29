# EnginBERT: A Text Embedding Model for Engineering Literature

EnginBERT is a domain-specific text embedding model designed for processing and understanding engineering literature. Built using the BERT architecture, EnginBERT is fine-tuned on engineering-specific texts to create embeddings that are more accurate and relevant for engineering-related tasks such as information retrieval, clustering, and semantic similarity analysis.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Data Collection and Preparation](#data-collection-and-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## Introduction

In fields like engineering, traditional NLP models often fall short due to the specialized terminology and complex structures in the text. EnginBERT aims to bridge this gap by providing a model specifically trained on engineering literature, enabling more effective information retrieval, document classification, and other NLP tasks within the engineering domain.

EnginBERT follows a pipeline similar to the one used in **PhysBERT**, but with a focus on engineering texts. It is pre-trained using a custom tokenizer and fine-tuned on engineering tasks to provide highly relevant embeddings.

## Features

- Custom vocabulary and tokenizer tailored to engineering terminology
- Pre-trained model fine-tuned on engineering-specific datasets
- Support for downstream tasks, including:
  - Information retrieval
  - Category clustering
  - Citation classification
  - Subdomain fine-tuning for specialized engineering fields

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/janMagnusHeimann/EnginBERT.git
   cd EnginBERT
   ```

2. **Install dependencies**:
   It's recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

   Dependencies include:
   - [Transformers](https://huggingface.co/transformers/) for BERT modeling
   - [Tokenizers](https://github.com/huggingface/tokenizers) for custom tokenization
   - [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for data handling
   - [PDFMiner](https://pypi.org/project/pdfminer.six/) for text extraction from PDFs

## Data Collection and Preparation

To build EnginBERT, a large dataset of engineering literature is required. Here's the process:

1. **Data Collection**: Gather engineering research papers and technical documents from sources like arXiv, IEEE Xplore, and Semantic Scholar. Save them in plain text or PDF format.

2. **Data Preprocessing**:
   - Run `scripts/data_arxiv.py` to download engineering-related articles from arXiv via the API.
   - Run `scripts/preprocess_data.py` to clean and tokenize the data, creating a balanced and ready-to-train dataset.

3. **Custom Tokenizer**: Use `scripts/train_tokenizer.py` (not provided here, but you can create one based on Hugging Face or SentencePiece) to create a vocabulary specifically suited to engineering terms. Save the tokenizer in `engineering_tokenizer`.

## Training

The training pipeline includes pre-training and fine-tuning phases, similar to BERT:

1. **Pre-training**:
   - Run `main.py` to execute the pipeline, starting from data collection to pre-training and fine-tuning the model.
   - The training process uses Masked Language Modeling (MLM) on the engineering corpus to create a pre-trained BERT model specifically for engineering.

2. **Fine-tuning**:
   - Fine-tune the model on engineering-specific tasks using `scripts/train_bert.py`, which loads the pre-trained EnginBERT model and applies additional supervised training on tasks like category clustering and citation classification.

## Evaluation

The effectiveness of EnginBERT is evaluated using the following tasks:

- **Category Clustering**: Evaluates how well the model groups similar engineering topics.
- **Information Retrieval**: Measures the model's ability to retrieve relevant documents based on engineering-specific queries.
- **Citation Classification**: Determines the model’s accuracy in identifying citation relationships between papers.
- **Subdomain Fine-tuning**: Additional fine-tuning on specific engineering subdomains like aerospace or mechanical engineering for targeted applications.

Evaluation results are logged and stored in the `results/` directory.

## Usage

Once trained, EnginBERT can be used for various NLP tasks related to engineering. Here’s a quick example of how to use the trained model for generating embeddings:

```python
from transformers import BertTokenizer, BertModel
import torch

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('path/to/engineering_tokenizer')
model = BertModel.from_pretrained('path/to/bert_classification_model')

# Sample text
text = "The torque on this engine exceeds 50 Nm."

# Tokenize and generate embeddings
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
embeddings = outputs.last_hidden_state.mean(dim=1)

print(embeddings)
```

## Project Structure

```plaintext
EnginBERT/
│
├── data/ 
│   ├── processed_papers.csv         # Preprocessed data saved as CSV
│   ├── cleaned_processed_papers.csv # Cleaned and balanced data
│
├── scripts/
│   ├── data_arxiv.py                # Script for downloading data from arXiv
│   ├── preprocess_data.py           # Script for data cleaning and preprocessing
│   ├── train_bert.py                # Script for training the BERT model
│   └── train_tokenizer.py           # Script for building a custom tokenizer (create as needed)
│
├── bert_classification_model/       # Directory for saving trained BERT model
│
├── main.py                          # Main script to execute the pipeline
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── results/                         # Directory for storing evaluation results
```

## Future Work

- **Expand Dataset**: Add more engineering-specific datasets from diverse fields (e.g., biomedical engineering, structural engineering).
- **Refine Tokenizer**: Continuously improve the custom tokenizer to capture more complex engineering terms and symbols.
- **Additional Downstream Tasks**: Implement tasks like question answering or summarization specific to engineering research.
- **Open-source Model**: Consider releasing EnginBERT on platforms like Hugging Face for community use.

## Acknowledgments

Special thanks to [PhysBERT](https://arxiv.org/abs/2408.09574) for inspiring this project. We also acknowledge the open-source NLP tools and datasets that made this project possible.

---

This README provides an overview of EnginBERT’s goals, setup instructions, and usage details. Feel free to modify it based on additional features or changes you implement in the project.
