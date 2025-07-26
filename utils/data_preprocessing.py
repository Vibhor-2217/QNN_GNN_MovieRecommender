import pandas as pd
import torch
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import from_networkx
from sklearn.model_selection import train_test_split

def preprocess_movielens(path='data/reviews.csv', threshold=1e-8):
    df = pd.read_csv(path, sep='\t', names=['user', 'item', 'rating', 'timestamp'])

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df['user'] = user_encoder.fit_transform(df['user'])
    df['item'] = item_encoder.fit_transform(df['item'])

    num_users = df['user'].nunique()
    num_items = df['item'].nunique()

    # Build bipartite graph
    B = nx.Graph()
    B.add_nodes_from(df['user'], bipartite=0)
    B.add_nodes_from(df['item'] + num_users, bipartite=1)
    B.add_edges_from((row['user'], row['item'] + num_users) for _, row in df.iterrows())

    # PageRank computation
    pr = nx.pagerank(B)
    print(f"PageRank stats → Min: {min(pr.values()):.10f}, Max: {max(pr.values()):.10f}, Avg: {sum(pr.values()) / len(pr):.10f}")

    # Prune edges below threshold
    filtered_edges = [
        (u, v) for u, v in B.edges()
        if pr[u] > threshold and pr[v] > threshold
    ]

    G = nx.Graph()
    G.add_edges_from(filtered_edges)
    data = from_networkx(G)

    # Prepare encoder mappings
    user2idx = {original: idx for idx, original in enumerate(user_encoder.classes_)}
    item2idx = {original: idx for idx, original in enumerate(item_encoder.classes_)}

    return df, num_users, num_items, user2idx, item2idx


def split_data(df, test_size=0.2, random_state=42):
    """
    Split the dataframe into training and testing sets.
    Returns: train_tensor, test_tensor
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    def df_to_tensor(dataframe):
        users = torch.tensor(dataframe['user'].values, dtype=torch.long)
        items = torch.tensor(dataframe['item'].values, dtype=torch.long)
        ratings = torch.tensor(dataframe['rating'].values, dtype=torch.float)
        return torch.stack([users, items, ratings], dim=1)

    return df_to_tensor(train_df), df_to_tensor(test_df)