import pandas as pd
import torch
from torch_geometric.data import Data

def preprocess_network(nodes_file, edges_file):
    nodes = pd.read_csv(nodes_file)
    edges = pd.read_csv(edges_file)

    x = torch.tensor(nodes.iloc[:, 1:].values, dtype=torch.float)

    edge_index = torch.tensor(edges.values.T, dtype=torch.long)
    return Data(x=x, edge_index=edge_index)
