import torch
from transformers import AutoTokenizer
import logging
from models.architectures import EquationUnderstanding
from scripts.helpers.data_loading import load_equation_data
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        # Load model and tokenizer
        model = EquationUnderstanding.from_pretrained(
            'model/technical_term_model')
        tokenizer = AutoTokenizer.from_pretrained(
            'model/technical_term_model')

        # Setup training
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load equation data
        train_dataset = load_equation_data(
            'data/cleaned_processed_papers.csv', tokenizer)
        train_dataloader = DataLoader(
            train_dataset, batch_size=16, shuffle=True)

        # Train
        logger.info("Starting equation understanding training...")
        model.train()
        model.to(device)

        for epoch in range(3):
            total_loss = 0
            for batch in train_dataloader:
                # Remove eq_labels from the batch if present since the model
                # does not accept it
                if 'eq_labels' in batch:
                    del batch['eq_labels']

                batch = {k: v.to(device) for k, v in batch.items()}

                optimizer.zero_grad()
                # Call the model without eq_labels
                outputs = model(**batch)

                # If your model doesn't return a loss by default,
                #  you'll need to compute it here.
                # For demonstration, let's assume the model
                #  returns 'loss' in outputs:
                loss = outputs.get('loss', None)
                if loss is not None:
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                else:
                    # Handle the case where no loss is provided by the model
                    # You might need a separate criterion
                    #  and label handling here.
                    pass

            if total_loss > 0:
                avg_loss = total_loss / len(train_dataloader)
                logger.info(f"Epoch {epoch+1}, Average loss: {avg_loss}")

        # Save
        model.save_pretrained('model/equation_model')
        tokenizer.save_pretrained('model/equation_model')
        logger.info("Equation understanding training completed")

    except Exception as e:
        logger.error(f"Error in equation understanding training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
