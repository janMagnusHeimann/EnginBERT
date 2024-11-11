# embedding_extraction.py
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd

# Load fine-tuned model and tokenizer
model = BertModel.from_pretrained('path/to/fine_tuned_enginbert')
tokenizer = BertTokenizer.from_pretrained('path/to/fine_tuned_enginbert')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Define embedding extraction function
def get_embeddings(text, model, tokenizer, device):
    inputs = tokenizer(text,
                       return_tensors="pt",
                       padding=True,
                       truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  
        # Pooling strategy: mean of last layer
    return embeddings.cpu().numpy()


# Load data
df = pd.read_csv('data/cleaned_processed_papers.csv')
texts = df['text_column'].tolist()  # Ensure 'text_column' contains text data

# Generate embeddings for each text
for text in texts:
    embeddings = get_embeddings(text, model, tokenizer, device)
    print("Embedding:", embeddings)
    # Store or use embeddings as needed for downstream tasks
