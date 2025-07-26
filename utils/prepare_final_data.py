import torch
import json
from pathlib import Path
from utils.data_preprocessing import preprocess_movielens, split_data

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "reviews.csv"
SAVE_DIR = BASE_DIR / "data" / "final"

# Ensure save directory exists
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

def prepare_final_dataset():
    print("✅ Loading and preprocessing...")
    ratings, num_users, num_items, user2idx, item2idx = preprocess_movielens(DATA_FILE)

    print("✅ Splitting data...")
    train, test = split_data(ratings)

    # Save train/test as PyTorch tensors
    torch.save(train, SAVE_DIR / "train_data.pt")
    torch.save(test, SAVE_DIR / "test_data.pt")

    # Save meta info
    metadata = {
        "num_users": num_users,
        "num_items": num_items,
        "user2idx": {str(k): int(v) for k, v in user2idx.items()},
        "item2idx": {str(k): int(v) for k, v in item2idx.items()}
    }

    with open(SAVE_DIR / "meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Optionally generating dummy embeddings for QNN normalization...")
    # Dummy embeddings (for now) to simulate QNN input
    user_emb = torch.randn(num_users, 64)
    item_emb = torch.randn(num_items, 64)

    norm_user_emb = normalize(user_emb)
    norm_item_emb = normalize(item_emb)

    torch.save(norm_user_emb, SAVE_DIR / "normalized_user_emb.pt")
    torch.save(norm_item_emb, SAVE_DIR / "normalized_item_emb.pt")

    print(f"🎉 All files saved to: {SAVE_DIR}")

if __name__ == "__main__":
    prepare_final_dataset()