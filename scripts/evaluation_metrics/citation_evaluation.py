import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

from scripts.model_and_tokenizer import tokenizer, model, device
# Load pre-trained BERT model and tokenizer

# Load the dataset with citation references populated
# Ensure this CSV includes 'title', 'full_text', and 'citation_references'
df = pd.read_csv("data/cleaned_processed_papers_with_citations.csv")


# Helper function to generate an embedding for each document's full text
def get_embedding(text):
    # Tokenize the text and prepare it for input into the model
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding="max_length",
                       max_length=512).to(device)

    # Perform a forward pass through the model without computing gradients
    with torch.no_grad():
        outputs = model(**inputs)

        # Use the mean of the hidden states for the embedding
        #  (alternatively, use CLS token)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding


# Generate embeddings for all documents' full text and stack them into an array
print("Generating embeddings for citation evaluation...")
embeddings = np.vstack([get_embedding(text) for text in df["full_text"]])

# Define top-k for evaluation;
# determines the number of retrieved documents for each query
top_k = 5


# Function to calculate precision@k for a specific document (query_idx)
def precision_at_k(query_idx, top_k):
    # Get the embedding for the query document and compute similarity
    #  with all other embeddings
    query_embedding = embeddings[query_idx].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings).flatten()

    # Set similarity with itself to -1 to exclude it
    #  from the top-k retrieved documents
    similarities[query_idx] = -1

    # Get indices of top-k most similar documents based on cosine similarity
    top_k_indices = similarities.argsort()[-top_k:][::-1]

    # Retrieve titles of the top-k documents to compare against true references
    retrieved_titles = df.iloc[top_k_indices]["title"].tolist()

    # Get true citation references for the query document
    true_references = set(df.iloc[query_idx]["citation_references"])

    # Calculate precision@k by counting how many of the retrieved
    #  documents are in the true references
    relevant_retrieved = sum(
        1 for title in retrieved_titles if title in true_references)
    precision = relevant_retrieved / top_k
    return precision


# Calculate precision@k for each document in the dataset and store scores
precision_scores = [precision_at_k(i, top_k) for i in range(len(df))]

# Calculate the average Precision@k across all documents
#   as the final evaluation metric
average_precision_at_k = np.mean(precision_scores)

# Print the result, showing the model's effectiveness at
#  retrieving relevant (cited) documents
print(f"Average Precision@{top_k} for Citation Evaluation: "
      f"{average_precision_at_k:.4f}")
