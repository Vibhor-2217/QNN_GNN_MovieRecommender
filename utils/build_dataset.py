import torch
import json
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# 🔧 Set paths using pathlib
ROOT_DIR = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT_DIR / "data" / "final"
TRAIN_PATH = FINAL_DIR / "train_data.pt"
META_PATH = FINAL_DIR / "meta.json"
GRAPH_PATH = FINAL_DIR / "graph_data.pt"

def build_dataset(df, edge_index, num_users, num_items):
    total_nodes = num_users + num_items
    x = torch.eye(total_nodes)  # Node features as one-hot

    edge_list = edge_index.t().tolist()

    # Corrected: (user, item + num_users) for rating_dict
    rating_dict = {(int(row['user']), int(row['item']) + num_users): float(row['rating'])
                   for _, row in df.iterrows()}

    y = torch.tensor(
        [rating_dict.get((u, v), rating_dict.get((v, u), 0)) for u, v in edge_list],
        dtype=torch.float32
    )

    indices = list(range(len(edge_list)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_idx = torch.tensor(train_idx)
    data.val_idx = torch.tensor(val_idx)
    data.test_idx = torch.tensor(test_idx)

    return data

def main():
    print("📦 Loading train data and metadata...")
    train_data = torch.load(TRAIN_PATH)

    with open(META_PATH, "r") as f:
        meta = json.load(f)
    num_users = meta["num_users"]
    num_items = meta["num_items"]

    print("🔗 Building edge index...")
    # ⚠️ Shift item index by +num_users to avoid ID collision
    edge_index = torch.tensor([
        [int(user), int(item) + num_users]
        for user, item, rating in train_data
    ], dtype=torch.long).t().contiguous()

    print("🧾 Creating dummy dataframe from train data...")
    df = pd.DataFrame(train_data.numpy(), columns=["user", "item", "rating"])

    print("🧠 Building PyG dataset...")
    data = build_dataset(df, edge_index, num_users, num_items)

    torch.save(data, GRAPH_PATH)
    print(f"✅ Graph data saved at: {GRAPH_PATH}")

if __name__ == "__main__":
    main()