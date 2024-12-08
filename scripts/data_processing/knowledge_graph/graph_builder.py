# scripts/data_processing/knowledge_graph/graph_builder.py
import networkx as nx
import json
import logging
from pathlib import Path
import spacy
import pandas as pd
import re
from typing import Dict, List, Set, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EngineeringKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        try:
            self.nlp = spacy.load("en_core_web_sm")
            # Using smaller model for now
            logger.info("Loaded language model")
        except OSError:
            logger.error("Language model not found")
            raise

        # Core engineering domains definition
        self.domains = {
            "mechanical_engineering": {
                "subfields": ["fluid_dynamics", "thermodynamics",
                              "mechanics", "materials"],
                "concepts": {
                    "fluid_dynamics": ["reynolds_number", "navier_stokes",
                                       "bernoulli", "flow"],
                    "thermodynamics": ["entropy", "heat_transfer",
                                       "thermal_conductivity"],
                    "mechanics": ["stress", "strain", "statics", "dynamics"],
                    "materials": ["composites", "alloys", "polymers"]
                }
            },
            "electrical_engineering": {
                "subfields": ["circuits", "electronics",
                              "power_systems", "control"],
                "concepts": {
                    "circuits": ["voltage", "current",
                                 "resistance", "capacitance"],
                    "electronics": ["semiconductor",
                                    "transistor", "amplifier"],
                    "power_systems": ["generation",
                                      "transmission", "distribution"],
                    "control": ["feedback", "pid", "stability"]
                }
            }
        }

        # Load domain patterns
        self.load_domain_patterns()

    def build_base_graph(self):
        """Build the base knowledge graph with engineering domains."""
        logger.info("Building base knowledge graph...")

        # Add domains
        for domain, content in self.domains.items():
            self.graph.add_node(domain, type="domain")
            logger.info(f"Added domain: {domain}")

            # Add subfields for each domain
            for subfield in content["subfields"]:
                self.graph.add_node(subfield, type="subfield")
                self.graph.add_edge(domain, subfield, relation="has_subfield")
                logger.info(f"Added subfield: {subfield} to {domain}")

                # Add concepts for each subfield
                if subfield in content["concepts"]:
                    for concept in content["concepts"][subfield]:
                        self.graph.add_node(concept, type="concept")
                        self.graph.add_edge(
                            subfield, concept, relation="has_concept")
                        logger.info(f"Added concept: {concept} to {subfield}")

        logger.info("Base graph built with " +
                    f"{self.graph.number_of_nodes()} nodes and "
                    f"{self.graph.number_of_edges()} edges")

        # Add cross-domain relationships
        self._add_cross_domain_relationships()

    def _add_cross_domain_relationships(self):
        """Add relationships between related concepts across domains."""
        related_concepts = {
            ("heat_transfer", "fluid_dynamics"): "affects",
            ("materials", "mechanics"): "influences",
            ("control", "electronics"): "uses",
            ("power_systems", "circuits"): "relies_on"
        }

        for (concept1, concept2), relation in related_concepts.items():
            if self.graph.has_node(concept1) and self.graph.has_node(concept2):
                self.graph.add_edge(concept1, concept2, relation=relation)
                logger.info("Added cross-domain relationship: " +
                            f"{concept1} {relation} {concept2}")

    def load_domain_patterns(self):
        """Load engineering-specific patterns and terms"""
        self.eng_patterns = {
            "equations": [
                r"(?<=[^A-Za-z])[A-Z][a-z]*\s*=\s*[\d.]+",  # Simple equations
                r"(?<=[^A-Za-z])[A-Z][a-z]*\s*=\s*[A-Za-z\d\s+\-*/()]+",
                # Complex equations
            ],
            "variables": [
                r"\b[A-Z][a-z]*\b(?=\s*[=<>])",  # Variables in equations
                r"\b[a-zA-Z]_[a-zA-Z0-9]+\b"  # Subscripted variables
            ],
            "units": [
                r"\b\d+\s*(?:m|kg|s|A|K|mol|cd|Hz|N|Pa|J|W|V)\b",  # SI units
                r"\b\d+\s*(?:mph|psi|ft|lb)\b"  # Imperial units
            ]
        }

        # Add relation types with weights
        self.relation_types = {
            "is_a": 1.0,
            "part_of": 0.8,
            "uses": 0.6,
            "related_to": 0.4,
            "measured_by": 0.7,
            "calculated_from": 0.7,
            "defined_by": 0.9
        }

    def extract_semantic_relations(self, doc) -> List[Tuple]:
        """Extract semantic relationships from text using dependency parsing"""
        relations = []
        for token in doc:
            if token.dep_ in ["nsubj", "dobj", "pobj"]:
                subject = token.text
                verb = token.head.text
                for child in token.head.children:
                    if child.dep_ in [
                         "dobj", "pobj"] and child.text != subject:
                        relations.append((subject, verb, child.text))
        return relations

    def extract_hierarchical_relations(self, terms: Set[str]) -> List[Tuple]:
        """Extract hierarchical relationships between terms"""
        hierarchical = []
        term_docs = {term: self.nlp(term) for term in terms}
        for term1 in terms:
            for term2 in terms:
                if term1 != term2:
                    # Check if one term is contained within another
                    if term1 in term2:
                        hierarchical.append((term2, "contains", term1))
                    # Check semantic similarity
                    similarity = term_docs[term1].similarity(term_docs[term2])
                    if similarity > 0.7:
                        hierarchical.append((term1, "related_to", term2))

        return hierarchical

    def add_equation_relationships(self):
        """Add relationships between equations, variables, and concepts"""
        equations = [n for n, d in self.graph.nodes(data=True)
                     if d.get("type") == "equation"]

        for eq in equations:
            # Extract variables from equation
            variables = set(re.findall(r'[A-Za-z_]+', eq))
            for var in variables:
                if self.graph.has_node(var):
                    self.graph.add_edge(eq, var,
                                        relation="uses",
                                        weight=self.relation_types["uses"])

    def calculate_centrality_metrics(self):
        """Calculate various centrality metrics for nodes"""
        metrics = {
            "degree": nx.degree_centrality(self.graph),
            "betweenness": nx.betweenness_centrality(self.graph),
            "eigenvector": nx.eigenvector_centrality(
                self.graph, max_iter=1000),
            "pagerank": nx.pagerank(self.graph)
        }

        # Add metrics as node attributes
        for metric_name, metric_values in metrics.items():
            nx.set_node_attributes(
                self.graph, metric_values, f"{metric_name}_centrality")

    def find_concept_communities(self):
        """Find communities of related concepts"""
        communities = nx.community.louvain_communities(
            self.graph.to_undirected())

        # Add community membership to nodes
        for i, community in enumerate(communities):
            for node in community:
                self.graph.nodes[node]["community"] = i

    def add_cross_references(self, papers_df: pd.DataFrame):
        """Add cross-references between papers and concepts"""
        # Create TF-IDF matrix
        texts = papers_df['text_sequences'].tolist()
        tfidf_matrix = self.tfidf.fit_transform(texts)

        # Calculate similarities between papers
        similarities = cosine_similarity(tfidf_matrix)

        # Add paper nodes and relationships
        for i, paper in papers_df.iterrows():
            paper_id = f"paper_{i}"
            self.graph.add_node(
                paper_id, type="paper", title=paper.get('title', ''))

            # Add relationships to similar papers
            similar_papers = np.argsort(similarities[i])[-5:]
            # Top 5 similar papers
            for similar_idx in similar_papers:
                if similar_idx != i:
                    self.graph.add_edge(paper_id, f"paper_{similar_idx}",
                                        relation="similar_to",
                                        similarity=similarities[i][
                                            similar_idx])

    def save_graph(self, output_path: Path):
        """Save the knowledge graph with all attributes"""
        # Calculate metrics before saving
        self.calculate_centrality_metrics()
        self.find_concept_communities()

        graph_data = {
            "nodes": [
                {
                    "id": node,
                    **{k: v for k, v in data.items()
                       if not isinstance(v, (nx.Graph, pd.DataFrame))}
                }
                for node, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **{k: v for k, v in data.items()
                       if not isinstance(v, (nx.Graph, pd.DataFrame))}
                }
                for u, v, data in self.graph.edges(data=True)
            ],
            "metadata": {
                "num_nodes": self.graph.number_of_nodes(),
                "num_edges": self.graph.number_of_edges(),
                "num_communities": len(set(nx.get_node_attributes(self.graph,
                                                                  'community'
                                                                  ).values())),
                "relation_types": list(self.relation_types.keys())
            }
        }

        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        logger.info(f"Knowledge graph saved to {output_path}")

    def analyze_graph(self) -> Dict:
        """Analyze the knowledge graph structure"""
        analysis = {
            "basic_stats": {
                "num_nodes": self.graph.number_of_nodes(),
                "num_edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph),
                "average_clustering": nx.average_clustering(self.graph),
            },
            "node_types": dict(pd.Series(
                [d.get('type', 'unknown') for _, d in self.graph.nodes(
                    data=True)]
            ).value_counts()),
            "relation_types": dict(pd.Series(
                [d.get('relation', 'unknown') for _, _, d in self.graph.edges(
                        data=True)]
            ).value_counts()),
            "top_central_nodes": {
                "degree": sorted(nx.degree_centrality(self.graph).items(),
                                 key=lambda x: x[1], reverse=True)[:10],
                "betweenness": sorted(nx.betweenness_centrality(
                    self.graph).items(),
                    key=lambda x: x[1], reverse=True)[:10]
            }
        }
        return analysis


def main():
    try:
        # Initialize the knowledge graph
        kg = EngineeringKnowledgeGraph()
        logger.info("Initialized Knowledge Graph")

        # Build base graph
        kg.build_base_graph()

        # Load papers data if file exists
        papers_path = Path('data/processed/cleaned_processed_papers.csv')
        if papers_path.exists():
            papers_df = pd.read_csv(papers_path)
            logger.info(f"Loaded {len(papers_df)} papers for processing")

            # Enrich graph from papers
            kg.enrich_from_papers(papers_df)
        else:
            logger.warning(
                "No processed papers found, skipping paper enrichment")

        # Save the graph
        output_path = Path('data/knowledge_graph/engineering_kg.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kg.save_graph(output_path)
        logger.info(f"Knowledge graph saved to {output_path}")

    except Exception as e:
        logger.error(f"Error in knowledge graph construction: {str(e)}")
        raise


if __name__ == "__main__":
    main()
