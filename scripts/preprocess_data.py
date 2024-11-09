import pandas as pd
from sklearn.utils import resample

# Load the dataset
df = pd.read_csv('data/processed_papers.csv')

# Check if 'labels' column exists
if 'labels' not in df.columns:
    raise ValueError("The dataset is missing the 'labels' column.")

# Count occurrences of each label
class_counts = df['labels'].value_counts()
minority_class = class_counts.idxmin()
majority_class = class_counts.idxmax()

# Print initial class distribution
print(f"Initial class distribution: {class_counts.to_dict()}")

# Check for imbalance and perform upsampling if necessary
if abs(class_counts[
      minority_class] - class_counts[majority_class]) > 0.1 * len(df):
    # Upsample the minority class if imbalance is greater than 10%
    print(f"Balancing classes by upsampling label {minority_class}")
    df_minority = df[df['labels'] == minority_class]
    df_majority = df[df['labels'] == majority_class]

    # Perform upsampling
    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )
    df = pd.concat([df_majority, df_minority_upsampled])
    print(f"Class distribution after balancing: "
          f"{df['labels'].value_counts().to_dict()}")


# Save the processed data
df.to_csv('data/cleaned_processed_papers.csv', index=False)
print("Data preprocessing complete and saved to "
      "'data/cleaned_processed_papers.csv'")
