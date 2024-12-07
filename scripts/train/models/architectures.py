# technical_term_training.py
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalTermPredictor(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.term_classifier = nn.Linear(config.hidden_size, 2)
        # Binary classification

        # Loss function for term prediction
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        term_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.term_classifier(sequence_output)

        loss = None
        if term_labels is not None:
            loss = self.loss_fct(logits.view(-1, 2), term_labels.view(-1))

        return {
            'loss': loss,
            'logits': logits
        }


def train_technical_term_model(
    model,
    train_dataloader,
    optimizer,
    num_epochs: int = 3,
    device: str = 'cuda'
):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()

            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)

            loss = outputs['loss']
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, " +
                    f"Average loss: {avg_loss:.4f}")


# equation_understanding.py
class EquationUnderstanding(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.equation_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        equation_spans: Optional[torch.Tensor] = None,
        equation_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]

        # Extract equation representations using spans
        if equation_spans is not None:
            equation_reprs = []
            for batch_idx, spans in enumerate(equation_spans):
                batch_reprs = []
                for start, end in spans:
                    span_repr = sequence_output[
                        batch_idx, start:end].mean(dim=0)
                    batch_reprs.append(span_repr)
                equation_reprs.append(torch.stack(batch_reprs))

            equation_reprs = torch.stack(equation_reprs)
            encoded_equations = self.equation_encoder(equation_reprs)

            # Compute contrastive loss if labels provided
            loss = None
            if equation_labels is not None:
                loss = self.compute_contrastive_loss(
                    encoded_equations, equation_labels)

            return {
                'loss': loss,
                'equation_embeddings': encoded_equations
            }

        return {'loss': None, 'equation_embeddings': None}

    def compute_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """Compute contrastive loss for equation similarity."""
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(
            embeddings, embeddings.transpose(0, 1)) / temperature

        # Create label mask (1 for positive pairs, 0 for negative)
        label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Compute loss
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = -(log_prob * label_mask).sum(dim=1) / label_mask.sum(dim=1)

        return loss.mean()


# component_relation.py
class ComponentRelation(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Component type classifier
        self.component_classifier = nn.Linear(
            config.hidden_size, config.num_component_types)

        # Relation classifier for component pairs
        self.relation_classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_relation_types)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        component_spans: Optional[torch.Tensor] = None,
        component_labels: Optional[torch.Tensor] = None,
        relation_pairs: Optional[torch.Tensor] = None,
        relation_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]

        # Extract component representations
        if component_spans is not None:
            component_reprs = self.extract_span_representations(
                sequence_output,
                component_spans
            )

            # Classify component types
            component_logits = self.component_classifier(component_reprs)

            # Process component pairs if provided
            relation_logits = None
            if relation_pairs is not None:
                pair_reprs = self.create_pair_representations(
                    component_reprs,
                    relation_pairs
                )
                relation_logits = self.relation_classifier(pair_reprs)

            # Compute losses if labels provided
            loss = 0
            if component_labels is not None:
                loss += nn.functional.cross_entropy(
                    component_logits.view(-1, self.config.num_component_types),
                    component_labels.view(-1)
                )

            if relation_labels is not None and relation_logits is not None:
                loss += nn.functional.cross_entropy(
                    relation_logits.view(-1, self.config.num_relation_types),
                    relation_labels.view(-1)
                )

            return {
                'loss': loss if loss != 0 else None,
                'component_logits': component_logits,
                'relation_logits': relation_logits
            }

        return {
            'loss': None, 'component_logits': None, 'relation_logits': None}

    def extract_span_representations(
        self,
        sequence_output: torch.Tensor,
        spans: torch.Tensor
    ) -> torch.Tensor:
        """Extract representations for text spans."""
        span_reprs = []
        for batch_idx, batch_spans in enumerate(spans):
            batch_reprs = []
            for start, end in batch_spans:
                span_repr = sequence_output[batch_idx, start:end].mean(dim=0)
                batch_reprs.append(span_repr)
            span_reprs.append(torch.stack(batch_reprs))
        return torch.stack(span_reprs)

    def create_pair_representations(
        self,
        component_reprs: torch.Tensor,
        pairs: torch.Tensor
    ) -> torch.Tensor:
        """Create representations for component pairs."""
        pair_reprs = []
        for batch_idx, batch_pairs in enumerate(pairs):
            batch_pair_reprs = []
            for idx1, idx2 in batch_pairs:
                pair_repr = torch.cat([
                    component_reprs[batch_idx, idx1],
                    component_reprs[batch_idx, idx2]
                ])
                batch_pair_reprs.append(pair_repr)
            pair_reprs.append(torch.stack(batch_pair_reprs))
        return torch.stack(pair_reprs)


# hierarchical_integration.py
class HierarchicalIntegration(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        # Hierarchical attention
        self.hierarchy_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob
        )

        # Knowledge integration
        self.knowledge_integration = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        knowledge_embeddings: Optional[torch.Tensor] = None,
        knowledge_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]

        if knowledge_embeddings is not None:
            # Apply hierarchical attention
            attended_output, attention_weights = self.hierarchy_attention(
                query=sequence_output.transpose(0, 1),
                key=knowledge_embeddings.transpose(0, 1),
                value=knowledge_embeddings.transpose(0, 1),
                key_padding_mask=knowledge_mask
            )

            # Combine with original representations
            combined = torch.cat([
                sequence_output,
                attended_output.transpose(0, 1)
            ], dim=-1)

            # Integrate knowledge
            enhanced_output = self.knowledge_integration(combined)

            return {
                'last_hidden_state': enhanced_output,
                'attention_weights': attention_weights,
                'pooler_output': outputs.pooler_output
            }

        return {
            'last_hidden_state': sequence_output,
            'pooler_output': outputs.pooler_output
        }
