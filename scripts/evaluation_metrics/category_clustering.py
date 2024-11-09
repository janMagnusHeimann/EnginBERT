import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import sys
import numpy as np
# Import model, tokenizer, and dataset from model_and_tokenizer module
from scripts.model_and_tokenizer import df, tokenizer, model, device
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
#from scripts import df, tokenizer, model, device


# Helper function to get embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       padding='max_length', max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding


# Generate embeddings for all documents
print("Generating embeddings for category clustering...")
embeddings = np.vstack([get_embedding(text) for text in df['full_text']])

# Number of clusters based on unique labels
num_clusters = df['labels'].nunique()

# Check if we have more than one cluster
if num_clusters > 1:
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    # Check for unique clusters in the result
    unique_clusters = len(set(clusters))
    if unique_clusters > 1:
        # Calculate silhouette score
        silhouette_avg = silhouette_score(embeddings, clusters)
        print(f"Silhouette Score [Category Clustering]: {silhouette_avg:.4f}")
    else:
        print("Silhouette Score cannot be computed") + (
            "because only one unique cluster was found.")
else:
    print("Clustering evaluation skipped due to") + (
        "insufficient unique labels in the dataset.")
