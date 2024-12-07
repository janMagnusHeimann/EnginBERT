# scripts/train/component_relation.py
import torch
from transformers import AutoTokenizer
import logging
from models.architectures import ComponentRelation

from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        # Load model and tokenizer
        model = ComponentRelation.from_pretrained('model/equation_model')
        tokenizer = AutoTokenizer.from_pretrained('model/equation_model')

        # Setup training
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load data
        train_dataset = load_component_data('data/processed/components.csv', tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        # Train
        logger.info("Starting component relation training...")
        model.train()
        model.to(device)
        
        for epoch in range(3):
            total_loss = 0
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1}, Average loss: {avg_loss}")
        
        # Save
        model.save_pretrained('model/component_model')
        tokenizer.save_pretrained('model/component_model')
        logger.info("Component relation training completed")
        
    except Exception as e:
        logger.error(f"Error in component relation training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
