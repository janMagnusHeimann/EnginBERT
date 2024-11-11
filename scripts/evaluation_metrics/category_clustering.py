import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel
import torch
from collections import Counter

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


# Function to generate embeddings for sentences
def get_embeddings(sentences):
    inputs = tokenizer(sentences,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    # Average over tokens for each sentence
    return embeddings


# Load your dataset
data = pd.read_csv("data/processed_papers_with_citations.csv")
# Adjust the path if needed
sentences = data['title'].tolist()
labels = data['labels'].astype("category").cat.codes
# Convert labels to categorical integer codes

# Generate embeddings for each sentence
embeddings = get_embeddings(sentences)

# Calculate the smallest class size
class_counts = Counter(labels)
min_class_size = min(class_counts.values())

# Adjust n_splits to be at least 2, if possible
n_splits = max(2, min(10, len(sentences), min_class_size))
kf = StratifiedKFold(n_splits=n_splits)
v_measure_scores = []

for train_index, test_index in kf.split(embeddings, labels):
    # Split into train and test for each fold
    X_train, X_test = embeddings[train_index], embeddings[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Standardize embeddings based on the training set
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Set the number of clusters to the smaller value
    #  of the unique labels in the training set or len(y_train)
    num_clusters = min(len(np.unique(y_train)), len(y_train))

    if num_clusters < 2:
        print("Skipping fold due to insufficient data for clustering.")
        continue

    # Fit KMeans to training data
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X_train)

    # Predict cluster labels on the test set
    y_pred = kmeans.predict(X_test)

    # Calculate V-measure score for the test set
    v_measure = v_measure_score(y_test, y_pred)
    v_measure_scores.append(v_measure)

# Final metric: Mean V-measure score across all valid folds
if v_measure_scores:
    mean_v_measure = np.mean(v_measure_scores)
    print(f"Mean V-measure Score: {mean_v_measure:.4f}")
else:
    print("No valid folds for evaluation.")
