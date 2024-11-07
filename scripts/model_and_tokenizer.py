from transformers import BertTokenizer, BertModel
import pandas as pd
import torch

# Load dataset, model, and tokenizer
df = pd.read_csv('data/cleaned_processed_papers.csv')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)