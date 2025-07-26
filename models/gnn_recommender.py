import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNRecommender(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(GNNRecommender, self).__init__()

        # Total number of nodes = users + items
        self.num_users = num_users
        self.num_items = num_items
        self.total_nodes = num_users + num_items

        # Embedding for all nodes (users + items)
        self.embedding = torch.nn.Embedding(self.total_nodes, embedding_dim)

        # Two-layer GCN
        self.gcn1 = GCNConv(embedding_dim, 64)
        self.gcn2 = GCNConv(64, 32)

    def forward(self, edge_index):
        x = self.embedding.weight

        # First GCN layer + ReLU
        x = F.relu(self.gcn1(x, edge_index))

        # Second GCN layer
        x = self.gcn2(x, edge_index)

        return x

    def predict(self, user_indices, item_indices):
        # Predict the score (e.g. dot product) between user and item embeddings
        user_embeddings = self.embedding(user_indices)
        item_embeddings = self.embedding(item_indices + self.num_users)

        return (user_embeddings * item_embeddings).sum(dim=1)