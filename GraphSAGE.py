import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, GATConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


G = nx.erdos_renyi_graph(100, 0.05)  # 100 nodes, 5% probability of edge
edges = np.array(list(G.edges)).T
x = np.random.rand(100, 5)  # Node features (5 features)

edge_index = torch.tensor(edges, dtype=torch.long)
x = torch.tensor(x, dtype=torch.float)


data = Data(x=x, edge_index=edge_index)


class SageNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SageNet, self).__init__()
        self.sage1 = GraphSAGE(in_channels, hidden_channels)
        self.sage2 = GraphSAGE(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = self.sage2(x, edge_index)
        return x


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super(GATNet, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels*heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x

# --- Instantiate models ---
sage_model = SageNet(in_channels=5, hidden_channels=16, out_channels=8)
gat_model = GATNet(in_channels=5, hidden_channels=8, out_channels=8)

# Forward pass to get embeddings
sage_embeddings = sage_model(data.x, data.edge_index)
gat_embeddings = gat_model(data.x, data.edge_index)


def causal_score(edge_index, embeddings):
    scores = []
    for i in range(embeddings.shape[0]):
        neighbors = edge_index[1][edge_index[0] == i]
        score = embeddings[neighbors].norm(dim=1).sum().item()
        scores.append(score)
    return np.array(scores)

sage_scores = causal_score(edge_index, sage_embeddings)
gat_scores = causal_score(edge_index, gat_embeddings)


top_influencers_sage = np.argsort(-sage_scores)[:10]
top_influencers_gat = np.argsort(-gat_scores)[:10]

print("Top influencers (GraphSAGE):", top_influencers_sage)
print("Top influencers (GAT):", top_influencers_gat)
