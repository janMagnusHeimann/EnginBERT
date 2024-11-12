import pandas as pd
from sklearn.utils import resample

# Load the dataset
df = pd.read_csv('data/processed_papers_with_citations.csv')

# Check if 'labels' column exists
if 'labels' not in df.columns:
    raise ValueError("The dataset is missing the 'labels' column.")

# Count occurrences of each label to check for class imbalance
class_counts = df['labels'].value_counts()
minority_class = class_counts.idxmin()
majority_class = class_counts.idxmax()

# Print initial class distribution for reference
print(f"Initial class distribution: {class_counts.to_dict()}")

# Perform upsampling if there is a significant imbalance
if abs(
    class_counts[minority_class] - class_counts[majority_class]
       ) > 0.1 * len(df):
    print(f"Balancing classes by upsampling label {minority_class}")
    df_minority = df[df['labels'] == minority_class]
    df_majority = df[df['labels'] == majority_class]

    # Upsample the minority class to match the majority class size
    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    # Combine majority class with upsampled minority class
    df = pd.concat([df_majority, df_minority_upsampled])
    print(f"Class distribution after balancing: {df['labels'].value_counts().to_dict()}")

else:
    print("Class distribution is balanced; no upsampling needed.")

# Save the processed dataset
df.to_csv('data/cleaned_processed_papers.csv', index=False)
print("Data preprocessing complete and " +
      "saved to 'data/cleaned_processed_papers.csv'")
