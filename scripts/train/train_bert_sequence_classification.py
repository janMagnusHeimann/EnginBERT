# train_bert_sequence_classification.py
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# Load the processed dataset
df = pd.read_csv('data/cleaned_processed_papers.csv')
df['labels'] = df['labels'] - 1  # Adjust labels if necessary

# Initialize the fine-tuned tokenizer and model
tokenizer = BertTokenizer.from_pretrained('model/fine_tuned_enginbert')
model = BertForSequenceClassification.from_pretrained(
    'model/fine_tuned_enginbert',  # Load from fine-tuned MLM model
    num_labels=3
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
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs} completed. " +
          f"Average Loss: {avg_loss:.4f}")

# Save the model and tokenizer
model.save_pretrained('model/bert_classification_model')
tokenizer.save_pretrained('model/bert_classification_model')
print("Model and tokenizer saved to 'model/bert_classification_model/'")
