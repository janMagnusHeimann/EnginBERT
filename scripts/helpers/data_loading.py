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


class EquationUnderstandingDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(f"Loading equation data from {file_path}")
        self.df = pd.read_csv(file_path)

        self.processed_data = []
        for text in self.df['text_sequences']:
            sequences = self._process_text(str(text))
            self.processed_data.extend(sequences)

        logger.info(f"Processed {len(self.processed_data)} equation sequences")

    def _process_text(self, text: str) -> List[Tuple[str, List[int]]]:
        """Process text into chunks with equation labels."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        # Split text into chunks for model input length constraints
        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            if current_length + len(word_tokens) >= self.max_length - 2:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word_tokens)
            else:
                current_chunk.append(word)
                current_length += len(word_tokens)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        processed_chunks = []

        # We'll consider inline equations defined by $...$ for simplicity.
        for chunk in chunks:
            encoding = self.tokenizer(
                chunk,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_offsets_mapping=True
            )
            offset_mapping = encoding.pop('offset_mapping')
            labels = [0] * len(encoding['input_ids'])

            # Identify equation regions by $...$
            # Find all '$' signs
            dollar_positions = [m.start() for m in re.finditer(r'\$', chunk)]
            # Pair them up as start and end of equations
            for i in range(0, len(dollar_positions), 2):
                if i + 1 < len(dollar_positions):
                    eq_start = dollar_positions[i]
                    eq_end = dollar_positions[i+1]

                    # Mark tokens within these spans as 1
                    for idx, (token_start, token_end
                              ) in enumerate(offset_mapping):
                        if token_start != token_end:
                            # Check if token overlaps with the equation range
                            if token_start >= eq_start and token_end <= eq_end:
                                labels[idx] = 1

            processed_chunks.append((chunk, labels))

        return processed_chunks

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        text, labels = self.processed_data[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'eq_labels': torch.tensor(labels)
        }

        return item


class ComponentRelationDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(f"Loading component data from {file_path}")
        self.df = pd.read_csv(file_path)

        self.processed_data = []
        for text in self.df['component_sequences']:
            sequences = self._process_text(str(text))
            self.processed_data.extend(sequences)

        logger.info("Processed " +
                    f"{len(self.processed_data)} component sequences")

    def _process_text(self, text: str) -> List[Tuple[str, List[int]]]:
        """Process text into chunks with component relation labels.
           This is a placeholder logic.
             Adjust it based on how you identify component relations."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            if current_length + len(word_tokens) >= self.max_length - 2:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word_tokens)
            else:
                current_chunk.append(word)
                current_length += len(word_tokens)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        processed_chunks = []
        for chunk in chunks:
            encoding = self.tokenizer(
                chunk,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_offsets_mapping=True
            )
            # offset_mapping = encoding.pop('offset_mapping')

            # Placeholder: all zeros as labels. Replace with actual logic.
            labels = [0] * len(encoding['input_ids'])

            processed_chunks.append((chunk, labels))

        return processed_chunks

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        text, labels = self.processed_data[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'relation_labels': torch.tensor(labels)
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


def load_equation_data(
        file_path: str, tokenizer:
        PreTrainedTokenizer) -> EquationUnderstandingDataset:
    """
    Load and prepare data for equation understanding
    """
    try:
        dataset = EquationUnderstandingDataset(file_path, tokenizer)
        logger.info(f"Successfully loaded {len(dataset)} training examples")
        return dataset
    except Exception as e:
        logger.error(f"Error loading equation data: {str(e)}")
        raise


def load_component_data(
        file_path: str, tokenizer:
        PreTrainedTokenizer) -> ComponentRelationDataset:
    """
    Load and prepare data for component relation prediction
    """
    try:
        dataset = ComponentRelationDataset(file_path, tokenizer)
        logger.info(f"Successfully loaded {len(dataset)} training examples")
        return dataset
    except Exception as e:
        logger.error(f"Error loading component data: {str(e)}")
        raise
