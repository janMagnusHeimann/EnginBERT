# scripts/train/component_relation.py
import torch
from transformers import AutoTokenizer, BertConfig
import logging
from models.architectures import ComponentRelation
from scripts.helpers.data_loading import load_component_data
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        # Load config and set required attributes
        config = BertConfig.from_pretrained('model/equation_model')
        config.num_component_types = 3
        # Set to the actual number of component types
        config.num_relation_types = 3
        # Set to the actual number of relation types

        # Load model and tokenizer with the updated config
        model = ComponentRelation.from_pretrained(
            'model/equation_model', config=config)
        tokenizer = AutoTokenizer.from_pretrained('model/equation_model')

        # Setup training
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load data
        train_dataset = load_component_data(
            'data/cleaned_processed_papers.csv', tokenizer)
        train_dataloader = DataLoader(
            train_dataset, batch_size=16, shuffle=True)

        # Train
        logger.info("Starting component relation training...")
        model.train()
        model.to(device)

        for epoch in range(3):
            total_loss = 0
            for batch in train_dataloader:
                # If the model expects 'labels'
                # instead of 'relation_labels', rename them
                if 'relation_labels' in batch:
                    batch['labels'] = batch.pop('relation_labels')

                batch = {k: v.to(device) for k, v in batch.items()}

                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1}, Average loss: {avg_loss}")

        # Save the model and tokenizer
        model.save_pretrained('model/component_model')
        tokenizer.save_pretrained('model/component_model')
        logger.info("Component relation training completed")

    except Exception as e:
        logger.error(f"Error in component relation training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
