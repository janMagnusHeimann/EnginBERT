import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

# Load fine-tuned model and tokenizer
model = BertModel.from_pretrained('model/fine_tuned_enginbert')
tokenizer = BertTokenizer.from_pretrained('model/fine_tuned_enginbert')
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
texts = df['full_text'].tolist()
# Make sure this column contains your text data

# Generate and save embeddings for each text
all_embeddings = []
for text in texts:
    embeddings = get_embeddings(text, model, tokenizer, device)
    all_embeddings.append(embeddings)

# Convert to array and save
all_embeddings = np.array(all_embeddings)

# Save embeddings to a .npy file
np.save('data/eng_embeddings.npy', all_embeddings)
# Saves embeddings in a NumPy file

# (Optional) Save embeddings to a CSV file along with document IDs or labels
# Reshape for DataFrame
df_embeddings = pd.DataFrame(all_embeddings.reshape(len(texts), -1))
df_embeddings.to_csv('data/eng_embeddings.csv', index=False)
