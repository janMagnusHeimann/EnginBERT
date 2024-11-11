import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scripts.helpers.model_and_tokenizer import df, tokenizer, model, device


# Helper function to get embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding="max_length", max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding


# Generate embeddings for all documents
print("Generating embeddings for information retrieval evaluation...")
embeddings = np.vstack([get_embedding(text) for text in df["full_text"]])

# Define top-k for evaluation
top_k = 10  # Adjusted to 10 for nDCG@10

# Dummy ground truth relevance data
# Assume each document should ideally retrieve others
#  in the same 'label' as relevant
df["relevant_docs"] = df["labels"].apply(
    lambda label: df[df["labels"] == label].index.tolist())


# Function to calculate DCG and nDCG
def dcg_at_k(relevance_scores, k):
    return sum((rel / np.log2(idx + 2)) for idx, rel in enumerate(
        relevance_scores[:k]))


def ndcg_at_k(query_idx, top_k):
    query_embedding = embeddings[query_idx].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings).flatten()

    # Exclude the query document itself
    similarities[query_idx] = -1

    # Get indices of the top-k most similar documents
    top_k_indices = similarities.argsort()[-top_k:][::-1]

    # Retrieve indices of relevant documents
    relevant_docs = df.iloc[query_idx]["relevant_docs"]
    relevance_scores = [
        1 if idx in relevant_docs else 0 for idx in top_k_indices]

    # Calculate DCG and iDCG
    dcg = dcg_at_k(relevance_scores, top_k)
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_relevance_scores, top_k)

    # Calculate nDCG@k
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg


# Evaluate nDCG@10 for all documents
ndcg_scores = [ndcg_at_k(i, top_k) for i in range(len(df))]

# Calculate average nDCG@10
average_ndcg_at_k = np.mean(ndcg_scores)

print(f"Average nDCG@{top_k} " +
      f"for Information Retrieval: {average_ndcg_at_k:.4f}")
