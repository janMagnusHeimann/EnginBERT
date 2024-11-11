from transformers import BertTokenizer, BertModel
import pandas as pd
import torch

# Load dataset, base model, and base tokenizer
df = pd.read_csv('data/cleaned_processed_papers.csv')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Load custom embeddings and tokenization

# from transformers import BertTokenizerFast, BertModel
# import pandas as pd
# import torch
# import numpy as np

# # Load dataset
# df = pd.read_csv('data/cleaned_processed_papers.csv')

# # Load custom tokenizer
# # Replace 'path/to/custom_tokenizer'
# with the actual path to your saved tokenizer
# tokenizer = BertTokenizerFast.from_pretrained('path/to/custom_tokenizer')

# # Load base BERT model
# model = BertModel.from_pretrained('bert-base-uncased')
# model.eval()

# # Move model to the available device (CPU or GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Load custom embeddings
# # Replace 'path/to/custom_embeddings.pt' with the actual path to
#  your saved embeddings file
# custom_embeddings = torch.load('path/to/custom_embeddings.pt')

# # Set the model's input embeddings to the custom embeddings
# model.get_input_embeddings().weight.data.copy_(custom_embeddings)

# # Optionally, freeze the embedding layer to prevent further training
# model.get_input_embeddings().weight.requires_grad = False

# # Verify the setup
# print("Tokenizer vocab size:", tokenizer.vocab_size)
# print("Custom embeddings shape:", custom_embeddings.shape)
# print("Model moved to device:", device)
