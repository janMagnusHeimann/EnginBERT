import logging
from transformers import BertTokenizer

# Set up logging for debug output with line numbers
logging.basicConfig(level=logging.DEBUG, format='%(lineno)d: %(message)s')

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a sample sentence for testing
sentence = "Machine learning is fascinating!"

# Step 1: Tokenize the sentence
tokens = tokenizer.tokenize(sentence)
logging.debug(f"Tokens (tokenized words/subwords): {tokens}")

# Step 2: Check for subword tokens (WordPiece tokenization)
subword_sentence = "unbelievable"
subword_tokens = tokenizer.tokenize(subword_sentence)
logging.debug(f"Subword Tokens (WordPiece tokenization): {subword_tokens}")

# Step 3: Convert tokens to IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
logging.debug(f"Token IDs (numeric representation of tokens): {token_ids}")

# Step 4: Add special tokens ([CLS] and [SEP])
encoded_with_special_tokens = tokenizer(sentence, add_special_tokens=True)
logging.debug(f"With Special Tokens ([CLS] and [SEP] included): {encoded_with_special_tokens['input_ids']}")

# Step 5: View the complete encoding process with padding, truncation, and attention mask
encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
logging.debug("Complete Encoding Process (including padding and attention mask):")
logging.debug(f"Input IDs: {encoded_input['input_ids']}")
logging.debug(f"Attention Mask: {encoded_input['attention_mask']}")

# Step 6: Explore the vocabulary to understand tokenization rules
# Display a small sample of the vocabulary
sample_vocab = dict(list(tokenizer.vocab.items())[:1000000])  # Display first 10 items for brevity
logging.debug(f"Sample of BERT Vocabulary (first 10 items): {sample_vocab}")
