import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def category_clustering(sentences, labels, model):
    """
    Evaluates an embedding model's ability to uncover semantic structure using clustering.

    Parameters:
    - sentences: List of sentences to be clustered.
    - labels: Ground truth labels indicating the category of each sentence.
    - model: Pre-trained embedding model to generate vector representations.

    Returns:
    - mean_v_measure: Mean V-measure score across all folds.
    """
    
    # Step 1: Generate embeddings for each sentence
    logging.debug("Generating embeddings for sentences...")
    embeddings = np.array([model.encode(sentence) for sentence in sentences])

    # Step 2: Initialize cross-validation
    logging.debug("Initializing stratified 10-fold cross-validation...")
    skf = StratifiedKFold(n_splits=10)
    v_measure_scores = []

    # Step 3: Perform cross-validation
    for train_index, test_index in skf.split(embeddings, labels):
        logging.debug("Splitting data into training and test sets...")
        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]

        # Step 4: Standardize the embeddings
        logging.debug("Standardizing the embeddings...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Step 5: Fit KMeans model
        logging.debug("Fitting KMeans model...")
        kmeans = KMeans(n_clusters=len(np.unique(labels)), init='k-means++', n_init=10)
        kmeans.fit(X_train)

        # Step 6: Predict clusters for the test set
        logging.debug("Predicting clusters for the test set...")
        y_pred = kmeans.predict(X_test)

        # Step 7: Calculate V-measure score
        v_measure = v_measure_score(y_test, y_pred)
        logging.debug(f"V-measure score for this fold: {v_measure:.2f}")
        v_measure_scores.append(v_measure)

    # Step 8: Calculate mean V-measure score
    mean_v_measure = np.mean(v_measure_scores)
    logging.debug(f"Mean V-measure score: {mean_v_measure:.2f}")
    return mean_v_measure

# Commentary on Missing Elements:
# 1. **Embedding Model**: Ensure your separate script provides a robust embedding model, and the `model.encode`
#    method generates meaningful embeddings for the input sentences.
# 2. **Standardization**: This script uses `StandardScaler` from scikit-learn. Verify if this standardization
#    method is suitable for your data.
# 3. **KMeans Sensitivity**: KMeans may be sensitive to the initialization of cluster centers. The `k-means++`
#    strategy helps, but you might need to experiment with the number of initializations.
# 4. **Performance Metrics**: The main metric is the mean V-measure score, but you could also consider
#    reporting additional insights based on your use case.