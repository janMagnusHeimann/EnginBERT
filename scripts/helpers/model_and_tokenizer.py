import torch
import pandas as pd
from transformers import BertTokenizer, BertModel


# Load fine-tuned BERT model and tokenizer for embedding extraction
def load_model_and_data(model_path='model/fine_tuned_enginbert',
                        data_path='data/cleaned_processed_papers.csv'):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    model.eval()  # Set model to evaluation mode

    # Define the device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the dataset
    df = pd.read_csv(data_path)

    return tokenizer, model, device, df
