import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset, model, and tokenizer
df = pd.read_csv('data/cleaned_processed_papers.csv')
tokenizer = BertTokenizer.from_pretrained('bert_classification_model')
model = BertModel.from_pretrained('bert_classification_model')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Helper function to get embeddings for a text
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       padding='max_length', max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding


# Generate embeddings for all documents
print("Generating embeddings for citation evaluation...")
embeddings = np.vstack([get_embedding(text) for text in df['full_text']])

# Assume 'citation_references' column contains lists of IDs
# or titles of cited papers
# Here we create a dummy reference list for demonstration;
# replace with actual citation data if available
df['citation_references'] = df['title'].apply(lambda x: [x])  # Dummy data;
# replace with actual citations

# Define top-k for evaluation
top_k = 5


# Calculate precision@k for citation retrieval
def precision_at_k(query_idx, top_k):
    query_embedding = embeddings[query_idx].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings).flatten()

    # Exclude the query document itself
    similarities[query_idx] = -1

    # Get the indices of the top-k most similar papers
    top_k_indices = similarities.argsort()[-top_k:][::-1]

    # Retrieve titles for top-k documents
    retrieved_titles = df.iloc[top_k_indices]['title'].tolist()
    true_references = set(df.iloc[query_idx]['citation_references'])

    # Calculate precision@k
    relevant_retrieved = sum(
        1 for title in retrieved_titles if title in true_references)
    precision = relevant_retrieved / top_k
    return precision


# Evaluate precision@k for all documents
precision_scores = [precision_at_k(i, top_k) for i in range(len(df))]
average_precision_at_k = np.mean(precision_scores)

print("Average Precision@{top_k} for Citation Evaluation: "
      f"{average_precision_at_k:.4f}")
