import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, List
import logging
from transformers import PreTrainedTokenizer
import re

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

        # Load the data
        logger.info(f"Loading data from {file_path}")
        self.df = pd.read_csv(file_path)

        # Pre-process to ensure sequences aren't too long
        self.processed_texts = []
        for text in self.df['text_sequences']:
            # Tokenize and truncate if necessary
            tokens = self.tokenizer.tokenize(str(text))
            if len(tokens) > max_length - 2:  # Account for [CLS] and [SEP]
                tokens = tokens[:(max_length - 2)]
            self.processed_texts.append(self.tokenizer.convert_tokens_to_string(tokens))

        logger.info(f"Processed {len(self.processed_texts)} sequences")

    def __len__(self):
        return len(self.processed_texts)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        text = self.processed_texts[idx]
        
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
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        
        # Add labels
        item['term_labels'] = torch.zeros_like(item['input_ids'])
        
        return item

def load_training_data(file_path: str, tokenizer: PreTrainedTokenizer) -> TechnicalTermDataset:
    """Load and prepare data for technical term prediction"""
    try:
        dataset = TechnicalTermDataset(file_path, tokenizer)
        logger.info(f"Successfully loaded {len(dataset)} training examples")
        return dataset
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise


# Additional utility functions
def create_custom_term_detector(patterns: List[str]) -> re.Pattern:
    """
    Create a custom technical term detector from patterns

    Args:
        patterns: List of regex patterns for technical terms

    Returns:
        Compiled regex pattern
    """
    return re.compile('|'.join(patterns))
