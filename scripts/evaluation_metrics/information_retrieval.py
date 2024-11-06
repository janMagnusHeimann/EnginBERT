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


# Helper function to get embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       padding='max_length', max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding


# Generate embeddings for all documents
print("Generating embeddings for information retrieval evaluation...")
embeddings = np.vstack([get_embedding(text) for text in df['full_text']])

# Define top-k for evaluation
top_k = 5

# Dummy ground truth relevance data
# For demonstration, assume each document should ideally retrieve
# others in the same 'label' as relevant
# In real-world scenarios, replace with actual ground truth pairs
df['relevant_docs'] = df['labels'].apply(
    lambda label: df[df['labels'] == label].index.tolist())


# Calculate precision@k and MRR
def precision_at_k_and_mrr(query_idx, top_k):
    query_embedding = embeddings[query_idx].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings).flatten()

    # Exclude the query document itself
    similarities[query_idx] = -1

    # Get indices of the top-k most similar documents
    top_k_indices = similarities.argsort()[-top_k:][::-1]

    # Retrieve indices of relevant documents
    relevant_docs = df.iloc[query_idx]['relevant_docs']
    relevant_retrieved = sum(
        1 for idx in top_k_indices if idx in relevant_docs)

    # Calculate precision@k
    precision = relevant_retrieved / top_k

    # Calculate MRR
    # (reciprocal rank of the first relevant document in the top-k)
    mrr = 0
    for rank, idx in enumerate(top_k_indices, start=1):
        if idx in relevant_docs:
            mrr = 1 / rank
            break

    return precision, mrr


# Evaluate precision@k and MRR for all documents
precision_scores = []
mrr_scores = []
for i in range(len(df)):
    precision, mrr = precision_at_k_and_mrr(i, top_k)
    precision_scores.append(precision)
    mrr_scores.append(mrr)

# Calculate average precision@k and MRR
average_precision_at_k = np.mean(precision_scores)
mean_reciprocal_rank = np.mean(mrr_scores)

print(f"Average Precision@{top_k} for Information Retrieval: "
      f"{average_precision_at_k:.4f}")
print(f"Mean Reciprocal Rank (MRR) for Information Retrieval: "
      f"{mean_reciprocal_rank:.4f}")
