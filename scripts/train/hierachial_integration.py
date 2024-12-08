# scripts/train/hierarchical_integration.py
import torch
from transformers import AutoTokenizer
import logging
from models.architectures import HierarchicalIntegration
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        # Load model and tokenizer
        model = HierarchicalIntegration.from_pretrained(
            'model/component_model')
        tokenizer = AutoTokenizer.from_pretrained(
            'model/component_model')

        # Setup training
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load data
        train_dataset = load_hierarchical_data(
            'data/processed/hierarchical.csv', tokenizer)
        train_dataloader = DataLoader(
            train_dataset, batch_size=16, shuffle=True)

        # Load knowledge graph embeddings
        knowledge_embeddings = load_knowledge_graph(
            'data/knowledge_graph/engineering_kg.json')

        # Train
        logger.info("Starting hierarchical integration training...")
        model.train()
        model.to(device)
        knowledge_embeddings = knowledge_embeddings.to(device)

        for epoch in range(3):
            total_loss = 0
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                batch['knowledge_embeddings'] = knowledge_embeddings

                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs['loss'] if 'loss' in outputs else 0
                if loss != 0:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() if loss != 0 else 0

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1}, Average loss: {avg_loss}")

        # Save
        model.save_pretrained('model/hierarchical_model')
        tokenizer.save_pretrained('model/hierarchical_model')
        logger.info("Hierarchical integration training completed")

    except Exception as e:
        logger.error(f"Error in hierarchical integration training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
