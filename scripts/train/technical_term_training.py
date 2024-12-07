# scripts/train/technical_term_training.py
import torch
from transformers import AutoTokenizer
import logging
from models.architectures import TechnicalTermPredictor
from scripts.helpers.data_loading import load_training_data
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        # Load model and tokenizer
        model = TechnicalTermPredictor.from_pretrained(
            'model/fine_tuned_enginbert')
        tokenizer = AutoTokenizer.from_pretrained('model/fine_tuned_enginbert')

        logger.info("Note: New layers were initialized " +
                    "for technical term prediction. This is expected.")

        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        # Load training data
        train_dataset = load_training_data(
            'data/cleaned_processed_papers.csv', tokenizer)
        train_dataloader = DataLoader(
            train_dataset, batch_size=16, shuffle=True)

        # Train the model
        logger.info("Starting technical term prediction training...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        model.train()
        model.to(device)

        for epoch in range(3):
            total_loss = 0
            for batch_idx, batch in enumerate(train_dataloader):
                # Only pass the expected arguments
                model_inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'term_labels': batch['term_labels'].to(device)
                }

                optimizer.zero_grad()
                outputs = model(**model_inputs)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if batch_idx % 10 == 0:  # Increased frequency of logging
                    logger.info(f"Epoch {epoch+1}, " +
                                f"Batch {batch_idx}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1}, Average loss: {avg_loss:.4f}")

        # Save the model
        output_dir = 'model/technical_term_model'
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved at '{output_dir}'")

    except Exception as e:
        logger.error(f"Error in technical term prediction training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
