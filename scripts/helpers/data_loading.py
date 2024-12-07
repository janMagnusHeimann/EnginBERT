# scripts/helpers/data_loading.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import re
import logging
from transformers import PreTrainedTokenizer
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalTermDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Technical term patterns
        self.term_patterns = [
            r'\b[A-Z][a-z]+(?:[ -][A-Z][a-z]+)*\b',  # CamelCase terms
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b(?:[A-Z][a-z]+){2,}\b',  # Concatenated terms
            r'\b\w+(?:[-/]\w+)*(?:\s+\w+(?:[-/]\w+)*)*\b'  # Hyphenated terms
        ]
        self.term_regex = re.compile('|'.join(self.term_patterns))

        # Load and preprocess data
        logger.info(f"Loading data from {file_path}")
        self.df = pd.read_csv(file_path)

        # Pre-process and store sequences and their labels
        self.processed_data = []
        for text in self.df['text_sequences']:
            sequences = self._process_text(str(text))
            self.processed_data.extend(sequences)

        logger.info(f"Processed {len(self.processed_data)} sequences")

    def _process_text(self, text: str) -> List[Tuple[str, List[int]]]:
        """Process text into chunks with technical term labels."""
        # Split into smaller chunks if needed
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            if current_length + len(word_tokens) >= self.max_length - 2:
                # Account for [CLS] and [SEP]
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word_tokens)
            else:
                current_chunk.append(word)
                current_length += len(word_tokens)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Process each chunk
        processed_chunks = []
        for chunk in chunks:
            # Find technical terms
            terms = list(self.term_regex.finditer(chunk))

            # Create labels
            encoding = self.tokenizer(
                chunk,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_offsets_mapping=True
            )

            # Create label tensor (0 for non-technical, 1 for technical terms)
            labels = [0] * len(encoding['input_ids'])
            offset_mapping = encoding.pop('offset_mapping')

            for term_match in terms:
                term_start, term_end = term_match.span()
                # Find tokens that overlap with the term
                for idx, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start != token_end:  # Skip padding tokens
                        if (token_start >= term_start and
                            token_start < term_end) or \
                           (token_end > term_start and token_end <= term_end):
                            labels[idx] = 1

            processed_chunks.append((chunk, labels))

        return processed_chunks

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        text, labels = self.processed_data[idx]
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        # Remove the batch dimension
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'term_labels': torch.tensor(labels)
        }

        return item


def load_training_data(
        file_path: str, tokenizer:
        PreTrainedTokenizer) -> TechnicalTermDataset:
    """
    Load and prepare data for technical term prediction

    Args:
        file_path: Path to the training data CSV
        tokenizer: Tokenizer to use for text processing

    Returns:
        TechnicalTermDataset instance with processed data and labels
    """
    try:
        dataset = TechnicalTermDataset(file_path, tokenizer)
        logger.info(f"Successfully loaded {len(dataset)} training examples")
        return dataset
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise