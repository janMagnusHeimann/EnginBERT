# scripts/data_processing/knowledge_graph/visualize.py
import networkx as nx
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def create_plotly_graph(
        G: nx.Graph, title: str = "Engineering Knowledge Graph"):
    """Create an interactive visualization of the knowledge graph"""
    # Get node positions using force-directed layout
    pos = nx.spring_layout(G, k=1/pow(len(G.nodes()), 0.3))

    # Create edge traces
    edge_x = []
    edge_y = []
    edge_text = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(edge[2].get('relation', 'related'))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []

    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        node_text.append(
            f"Node: {node[0]}<br>Type: {node[1].get('type', 'unknown')}")
        # Color based on community
        node_color.append(node[1].get('community', 0))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=10,
            color=node_color,
            line_width=2))

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                       title=title,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(
                           showgrid=False,
                           zeroline=False,
                           showticklabels=False),
                       yaxis=dict(
                           showgrid=False,
                           zeroline=False,
                           showticklabels=False))
                    )

    return fig


def create_metrics_visualization(graph_path: Path):
    """Create visualizations for graph metrics"""
    with open(graph_path) as f:
        data = json.load(f)

    # Convert to DataFrame for easier plotting
    nodes_df = pd.DataFrame(data['nodes'])
    edges_df = pd.DataFrame(data['edges'])

    # Create figures
    figs = []

    # Node type distribution
    type_counts = nodes_df['type'].value_counts()
    figs.append(px.pie(names=type_counts.index,
                       values=type_counts.values,
                       title="Distribution of Node Types"))

    # Centrality metrics
    centrality_cols = [col for col in nodes_df.columns if 'centrality' in col]
    for col in centrality_cols:
        fig = px.histogram(nodes_df, x=col,
                           title="Distribution of " +
                           f"{col.replace('_', ' ').title()}")
        figs.append(fig)

    # Relationship type distribution
    rel_counts = edges_df['relation'].value_counts()
    figs.append(px.bar(x=rel_counts.index,
                       y=rel_counts.values,
                       title="Distribution of Relationship Types"))

    return figs


def main():
    # Load the graph
    graph_path = Path('data/knowledge_graph/engineering_kg.json')
    with open(graph_path) as f:
        graph_data = json.load(f)

    # Create network graph
    G = nx.node_link_graph({
        'nodes': graph_data['nodes'],
        'links': graph_data['edges']
    })

    # Create visualizations
    network_fig = create_plotly_graph(G)
    metric_figs = create_metrics_visualization(graph_path)

    # Save visualizations
    output_dir = Path('data/knowledge_graph/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    network_fig.write_html(output_dir / 'network.html')
    for i, fig in enumerate(metric_figs):
        fig.write_html(output_dir / f'metric_{i}.html')


if __name__ == "__main__":
    main()
