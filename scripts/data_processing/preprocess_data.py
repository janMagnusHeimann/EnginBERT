import pandas as pd
import re
from sklearn.utils import resample

# Load the dataset
df = pd.read_csv('data/processed_papers_with_citations.csv')

# Check if 'labels' column exists
if 'labels' not in df.columns:
    raise ValueError("The dataset is missing the 'labels' column.")


def clean_text(text):
    # Updated cleaning for figure/table captions, headers/footers,
    # references section, etc.
    text = re.sub(r'(Figure\s*\d+|Table\s*\d+):.*?(?=\n\n|$)',
                  '',
                  text,
                  flags=re.IGNORECASE)
    text = re.sub(r'(^|\n)\s*\d{1,3}\s*(\n|$)', '', text)  # Page numbers
    text = re.sub(r'\n\s*References\s*[\s\S]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# Apply the enhanced cleaning function to the 'full_text' column
df['cleaned_text'] = df['full_text'].apply(clean_text)

# Check and balance class distribution if needed
class_counts = df['labels'].value_counts()
minority_class = class_counts.idxmin()
majority_class = class_counts.idxmax()
print(f"Initial class distribution: {class_counts.to_dict()}")
if abs(class_counts[minority_class] - class_counts[majority_class
                                                   ]) > 0.1 * len(df):
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
    print("Class distribution after balancing:",
          df['labels'].value_counts().to_dict())
else:
    print("Class distribution is balanced; no upsampling needed.")


# Prepare text for BERT by segmenting into chunks
def split_into_sequences(text, max_length=512):
    words = text.split()
    return [' '.join(words[
        i:i + max_length]) for i in range(0, len(words), max_length)]


# Apply segmentation to cleaned text
df['text_sequences'] = df['cleaned_text'].apply(split_into_sequences)

# Explode segmented text into individual rows
train_df = df.explode('text_sequences').dropna(
    subset=['text_sequences']).reset_index(drop=True)

# Retain 'full_text' in the training data for compatibility
train_df['full_text'] = train_df['cleaned_text']

# Prepare the evaluation DataFrame with citation references
evaluation_df = df[['title',
                    'cleaned_text',
                    'labels',
                    'citation_references']].copy()

# Save the training data (chunked for BERT with 'full_text' retained)
train_df[['text_sequences', 'labels', 'full_text']].to_csv(
    'data/cleaned_processed_papers.csv', index=False)

# Save the evaluation data with citation references
evaluation_df.to_csv('data/evaluation_processed_papers.csv', index=False)

print("Training data saved to 'data/cleaned_processed_papers.csv'")
print("Evaluation data with citation references " +
      "saved to 'data/evaluation_processed_papers.csv'")

# Verify that the 'title' column exists in the evaluation file
eval_df_check = pd.read_csv('data/evaluation_processed_papers.csv')
print("Columns in evaluation_processed_papers.csv:", eval_df_check.columns)
