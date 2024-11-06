import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset and model
df = pd.read_csv('data/cleaned_processed_papers.csv')
tokenizer = BertTokenizer.from_pretrained('bert_classification_model')
model = BertModel.from_pretrained('bert_classification_model')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Helper function to get embeddings
def get_embedding(text):
    inputs = tokenizer(
        text, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding


# Get embeddings for all documents
embeddings = np.vstack([get_embedding(text) for text in df['full_text']])

# Category Clustering
num_clusters = len(df['labels'].unique())
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Check unique clusters and calculate silhouette score
unique_clusters = len(set(clusters))
if unique_clusters > 1:
    silhouette_avg = silhouette_score(embeddings, clusters)
    print(f"Silhouette Score for Category Clustering: {silhouette_avg:.4f}")
else:
    print(
        "Silhouette Score cannot be computed due to only one unique cluster.")


# Information Retrieval
def retrieve_similar_documents(query, embeddings, df, top_k=5):
    query_embedding = get_embedding(query)
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    similar_indices = similarities.argsort()[-top_k:][::-1]
    return df.iloc[similar_indices][['title', 'full_text', 'labels']]


# Example query
query_text = "Structural analysis in aerospace engineering"
similar_docs = retrieve_similar_documents(query_text, embeddings, df)
print("\nTop 5 Documents for Query:")
print(similar_docs.to_string(index=False))
