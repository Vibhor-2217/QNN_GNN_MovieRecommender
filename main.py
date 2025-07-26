import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from utils.data_preprocessing import preprocess_movielens
from models.gnn_recommender import GNNRecommender
from sklearn.metrics import mean_squared_error
import pandas as pd

# 🔄 Step 1: Preprocess data
df, num_users, num_items, user2idx, item2idx = preprocess_movielens()
print(f"✅ Data loaded: {len(df)} rows")
print(f"Users: {num_users} | Items: {num_items}")

# 🔗 Step 2: Build edge_index tensor from df
edge_index = torch.tensor([
    [int(row["user"]), int(row["item"])]
    for _, row in df.iterrows()
], dtype=torch.long).t().contiguous()
print(f"✅ Graph has {edge_index.shape[1]} edges.")

# 🎯 Step 3: Hyperparameters
embedding_dim = 64
epochs = 20
learning_rate = 0.01

# 🧠 Step 4: Initialize model and optimizer
model = GNNRecommender(num_users, num_items, embedding_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 🏗️ Step 5: Prepare user-item-rating tensors for training
user_tensor = torch.tensor(df['user'].values, dtype=torch.long)
item_tensor = torch.tensor(df['item'].values, dtype=torch.long)
rating_tensor = torch.tensor(df['rating'].values, dtype=torch.float)

# 🔁 Step 6: Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass to compute node embeddings
    model(edge_index)

    # Predict ratings
    preds = model.predict(user_tensor, item_tensor)

    # Compute MSE loss
    loss = F.mse_loss(preds, rating_tensor)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# 🔍 Step 7: Recommendation function
def recommend_top_k(user_id, k=5):
    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([user_id] * num_items)
        item_tensor = torch.tensor(list(range(num_items)))
        scores = model.predict(user_tensor, item_tensor)
        top_k_items = torch.topk(scores, k).indices
        return top_k_items.tolist()

# 🎯 Example: Recommend for user 42
user_id = 42
top_k = recommend_top_k(user_id, k=5)
print(f"\n🎯 Top 5 Recommendations for User {user_id}: {top_k}")

# 📈 Step 8: Final Evaluation
model.eval()
with torch.no_grad():
    preds = model.predict(user_tensor, item_tensor)
    mse = mean_squared_error(rating_tensor.numpy(), preds.numpy())
    print(f"\n✅ Final Mean Squared Error (MSE): {mse:.4f}")