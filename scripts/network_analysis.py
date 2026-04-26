import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def analyze_graph(file_path, name):
    print(f"\n{'='*25} {name} (Topological Summary) {'='*25}")

    # Load the sampled dataset
    df = pd.read_csv(file_path)

    # Load graph data as a NetworkX object
    G = nx.from_pandas_edgelist(df, source='head', target='tail', edge_attr='relation', create_using=nx.MultiDiGraph())

    # Convert to undirected graph to calculate Avg. Clustering Coefficient
    G_undirected = G.to_undirected()

    # Define common topological measures
    stats = {
        "Total Entities (Nodes)": G.number_of_nodes(),
        "Total Relations (Edges)": G.number_of_edges(),
        "Average Degree": round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 2),
        "Graph Density": f"{nx.density(G):.5f}",
        "Is Weakly Connected": nx.is_weakly_connected(G),
        "Number of Connected Components": nx.number_weakly_connected_components(G),
        "Avg Clustering Coefficient": round(nx.average_clustering(nx.Graph(G_undirected)), 4)
    }

    for key, val in stats.items():
        print(f" > {key}: {val}")

    # Generate network visualization
    print(f"Generating enhanced visualization for {name}...")
    plt.figure(figsize=(15, 15))

    pos = nx.spring_layout(G, k=0.12, iterations=50, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=60, node_color="#3498db", alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        width=1.5,
        edge_color="#5d6d7e",
        alpha=0.7,
        arrowsize=12,        # Visible arrows for directionality
        connectionstyle="arc3,rad=0.1" # Slight curve to help see overlapping edges
    )

    plt.title(f"Network Topology: {name} (Sampled)", fontsize=18)
    plt.axis('off')
    plt.tight_layout()

    output_img = f"results/{name}_numeric_topology.png"
    plt.savefig(output_img, dpi=300)
    print(f"Visualization saved: {output_img}")

if __name__ == "__main__":
    analyze_graph("data/FB15k-237_valid_numeric_500.csv", "FB15k-237")
    analyze_graph("data/WN18RR_valid_numeric_500.csv", "WN18RR")