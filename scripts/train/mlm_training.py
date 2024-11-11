from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding="max_length",
                                  truncation=True, max_length=self.max_length,
                                  return_tensors="pt")
        # Flatten each returned tensor to remove the extra dimension
        return {key: val.squeeze(0) for key, val in encoding.items()}

# Load data and tokenizer
df = pd.read_csv('data/cleaned_processed_papers.csv')
texts = df['full_text'].tolist()  # Ensure 'full_text' contains text data

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create dataset and dataloader
dataset = TextDataset(texts, tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=data_collator)

# Initialize model for MLM
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1} completed. Loss: {total_loss / len(dataloader)}")

# Save model and tokenizer
model.save_pretrained('model/fine_tuned_enginbert')
tokenizer.save_pretrained('model/fine_tuned_enginbert')
print("Model and tokenizer saved at 'model/fine_tuned_enginbert'")
