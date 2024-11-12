import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Load fine-tuned BERT model and tokenizer for embedding extraction
from scripts.helpers.model_and_tokenizer import load_model_and_data

# Load model, tokenizer, device, and data
tokenizer, model, device, df = load_model_and_data()


# Improved helper function to determine if a citation is a valid title
def is_valid_citation_title(citation):
    pattern = r"(Publishers|Press|Academic|Boston|ISBN|Symposium|" \
              r"Conference|Proceedings|Laboratory|University|Vol|" \
              r"pp|Series|Edition)"
    return len(citation.split()) > 3 and not re.search(pattern, citation)


# Classification and diagnostic print for the first 10 documents
print("Title Classification for the First 10 Documents:")
for i in range(min(10, len(df))):
    title = df.loc[i, 'title']
    citations = df.loc[i, 'citation_references']

    has_valid_citations = (
        isinstance(citations, str) and
        bool(eval(citations)) and
        any(is_valid_citation_title(citation) for citation in eval(citations))
    )

    print(f"Document {i+1} Title: {title}")
    print(f"Extracted Citations: {citations}")
    print("Contains Valid Citations: " +
          f"{'Yes' if has_valid_citations else 'No'}\n")

# Filter out documents that lack valid citations
df['citation_references'] = df['citation_references'].apply(eval)
df = df[df["citation_references"].apply(
    lambda citations: isinstance(
        citations, list) and any(is_valid_citation_title(
            citation) for citation in citations))]

# Check if the filtered DataFrame is empty
if df.empty:
    print("No documents with valid citations were found. Skipping evaluation.")
else:
    # Helper function to generate an embedding for each document's full text
    def get_embedding(text):
        inputs = tokenizer(text,
                           return_tensors="pt",
                           truncation=True,
                           padding="max_length",
                           max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embedding

    # Generate embeddings for all documents' full text
    #  and stack them into an array
    print("Generating embeddings for citation evaluation...")
    embeddings = np.vstack([get_embedding(text) for text in df["full_text"]])

    # Define top-k for evaluation
    top_k = 5

    # Function to calculate precision@k for a specific document (query_idx)
    def precision_at_k(query_idx, top_k):
        query_embedding = embeddings[query_idx].reshape(1, -1)
        similarities = cosine_similarity(query_embedding, embeddings).flatten()

        # Set similarity with itself to -1 to exclude it from top-k
        similarities[query_idx] = -1

        # Get indices of top-k most similar documents
        top_k_indices = similarities.argsort()[-top_k:][::-1]
        retrieved_titles = df.iloc[top_k_indices]["title"].tolist()

        # True citation references
        true_references = set(df.iloc[query_idx]["citation_references"])

        # Calculate precision@k
        relevant_retrieved = sum(
            1 for title in retrieved_titles if title in true_references)
        precision = relevant_retrieved / top_k
        return precision

    # Calculate precision@k for each document with valid citations
    precision_scores = [precision_at_k(i, top_k) for i in range(len(df))]

    # Calculate average Precision@k across all valid documents
    average_precision_at_k = np.mean(precision_scores)

    print(f"Average Precision@{top_k} for " +
          f"Citation Evaluation: {average_precision_at_k:.4f}")
