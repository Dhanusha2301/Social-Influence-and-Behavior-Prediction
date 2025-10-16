
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_network(edges_list, top_influencers=None, node_labels=None, title="Social Network"):
    """
    Plots a network graph with highlighted top influencers.
    
    Parameters:
    - edges_list: list of tuples representing edges [(0,1), (0,2), ...]
    - top_influencers: list of node indices to highlight
    - node_labels: dict {node_index: label} for labeling nodes
    - title: string for plot title
    """
    G = nx.Graph()
    G.add_edges_from(edges_list)
    
    # Node colors: top influencers in red, others in lightblue
    colors = []
    for node in G.nodes():
        if top_influencers and node in top_influencers:
            colors.append('red')
        else:
            colors.append('lightblue')
    
    # Draw network
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # force-directed layout
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=400, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Draw labels if provided
    if node_labels:
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()


def plot_embeddings(embeddings, top_influencers=None, title="Node Embeddings"):
    """
    Plots 2D embeddings of nodes using first two dimensions.
    
    Parameters:
    - embeddings: numpy array or torch tensor of shape [num_nodes, embedding_dim]
    - top_influencers: list of node indices to highlight
    - title: plot title
    """
    if hasattr(embeddings, 'detach'):  # If tensor
        embeddings = embeddings.detach().numpy()
    
    plt.figure(figsize=(10, 8))
    
    for i, emb in enumerate(embeddings):
        if top_influencers and i in top_influencers:
            plt.scatter(emb[0], emb[1], color='red', s=100, label='Top Influencer' if i == top_influencers[0] else "")
        else:
            plt.scatter(emb[0], emb[1], color='lightblue', s=50)
    
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.title(title, fontsize=16)
    
    if top_influencers:
        plt.legend()
    plt.show()

if __name__ == "__main__":
  
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 0)]
  
    top_nodes = [0, 3]
   
    plot_network(edges_list=edges, top_influencers=top_nodes, title="Example Social Network")
   
    embeddings = np.random.rand(6, 2)
    plot_embeddings(embeddings, top_influencers=top_nodes, title="Example Node Embeddings")
