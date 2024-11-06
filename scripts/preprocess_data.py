import pandas as pd
from sklearn.utils import resample

# Load the dataset
df = pd.read_csv('data/processed_papers.csv')

# Check if 'labels' column exists
if 'labels' not in df.columns:
    raise ValueError("The dataset is missing the 'labels' column.")

# Count label occurrences, with a fallback for missing labels
class_counts = df['labels'].value_counts().to_dict()

# Set the counts to 0 if the label is missing
count_0 = class_counts.get(0, 0)
count_1 = class_counts.get(1, 0)

# Only perform upsampling if both classes have samples
if count_0 > 0 and count_1 > 0:
    # Check for imbalance and balance if necessary
    if abs(count_0 - count_1) > 0.1 * len(df):
        # If there's more than 10% imbalance
        # Upsample the minority class
        minority_class = 0 if count_0 < count_1 else 1
        majority_class = 1 if minority_class == 0 else 0

        df_minority = df[df['labels'] == minority_class]
        df_majority = df[df['labels'] == majority_class]

        df_minority_upsampled = resample(
            df_minority, replace=True, n_samples=len(
                df_majority), random_state=42)
        df = pd.concat([df_majority, df_minority_upsampled])

# Save the processed data
df.to_csv('data/cleaned_processed_papers.csv', index=False)
print("Data preprocessing complete and saved to "
      "'data/cleaned_processed_papers.csv'")
