import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# Load the processed dataset
df = pd.read_csv('data/cleaned_processed_papers.csv')
df['labels'] = df['labels'] - 1
print(df['labels'].unique())  # Should output [0, 1, 2] now

# Save the adjusted dataset to avoid this issue in future runs
df.to_csv('data/cleaned_processed_papers.csv', index=False)
# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load BERT model for classification with 3 labels
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=3
)


# Custom dataset class
class PaperDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['full_text']
        label = self.data.iloc[index]['labels']
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Define max length and create dataset and dataloader
max_len = 512
dataset = PaperDataset(df, tokenizer, max_len)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Set up the model, optimizer, and device
model.train()
optimizer = AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
epochs = 3
for epoch in range(epochs):
    print(f"Starting epoch {epoch + 1}/{epochs}")
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(
        f"Epoch {epoch + 1}/{epochs} completed. Average Loss: {avg_loss:.4f}")

# Save the model and tokenizer
model.save_pretrained('bert_classification_model')
tokenizer.save_pretrained('bert_classification_model')
print("Model and tokenizer saved to 'bert_classification_model/'")
