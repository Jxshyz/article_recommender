import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import load_npz
from sklearn.preprocessing import MinMaxScaler
import pickle
from tqdm import tqdm


# =========================
# Create Interactions CSV if Needed
# =========================
def create_interaction_df(force=False):
    output_path = os.path.join("data", "preprocessed", "interactions_long.csv")
    if os.path.exists(output_path) and not force:
        print(f"[✓] interactions_long.csv exists – skipping creation.")
        return

    print("[!] Creating interactions_long.csv...")
    matrix_path = os.path.join("data", "preprocessed", "user_item_matrix.npz")
    sparse_matrix = load_npz(matrix_path)

    coo = sparse_matrix.tocoo()
    df_interactions = pd.DataFrame({"user_id": coo.row, "item_id": coo.col, "label": coo.data})
    print(f"Initial interactions: {len(df_interactions)}")
    print(f"Label values before filter: {df_interactions['label'].unique()}")

    # Filter out non-likes (should be mostly 1s)
    df_interactions = df_interactions[df_interactions["label"] == 1]
    print(f"After label filter: {len(df_interactions)}")

    # Load index-to-article mapping to recover article_id
    mapping_path = os.path.join("data", "preprocessed", "uim_mappings.pkl")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"[!] Mapping file not found: {mapping_path}")

    with open(mapping_path, "rb") as f:
        uim_u2i, uim_a2i, uim_i2u, uim_i2a = pickle.load(f)

    # Map internal item_id (column index) to real article_id
    idx2article = dict(enumerate(uim_i2a))
    df_interactions["article_id"] = df_interactions["item_id"].map(idx2article)

    # Load article timestamps
    articles_path = os.path.join("data", "ebnerd", "articles.parquet")
    articles_df = pd.read_parquet(articles_path)[["article_id", "published_time"]]

    print(f"Articles loaded: {len(articles_df)}")
    print(f"Missing published_time in articles: {articles_df['published_time'].isna().sum()}")

    # Merge on article_id
    df_interactions = df_interactions.merge(articles_df, on="article_id", how="left")
    print(f"After merge: {len(df_interactions)}")
    print(f"Missing published_time after merge: {df_interactions['published_time'].isna().sum()}")

    # Drop rows without timestamp and sort
    df_interactions = df_interactions.dropna(subset=["published_time"])
    df_interactions["published_time"] = pd.to_datetime(df_interactions["published_time"])
    df_interactions = df_interactions.sort_values("published_time")

    # Optional: restore item_id for model training
    df_interactions = df_interactions[["user_id", "item_id", "label", "published_time"]]

    if df_interactions.empty:
        print("[!] No valid interactions after filtering. Check sparse matrix or article data!")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_interactions.to_csv(output_path, index=False)
    print(f"[✓] Created: {output_path} with {len(df_interactions)} rows")


def create_balanced_interaction_df():
    output_path = os.path.join("data", "preprocessed", "interactions_long_balanced.csv")
    print("[!] Creating balanced interactions_long_balanced.csv...")

    matrix_path = os.path.join("data", "preprocessed", "user_item_matrix.npz")
    mapping_path = os.path.join("data", "preprocessed", "uim_mappings.pkl")
    articles_path = os.path.join("data", "ebnerd", "articles.parquet")

    # Load sparse matrix and mappings
    sparse_matrix = load_npz(matrix_path).tocoo()
    with open(mapping_path, "rb") as f:
        uim_u2i, uim_a2i, uim_i2u, uim_i2a = pickle.load(f)

    idx2article = dict(enumerate(uim_i2a))
    idx2user = dict(enumerate(uim_i2u))
    all_user_ids = list(uim_u2i.keys())
    all_item_ids = list(uim_a2i.keys())

    # Positive interactions
    df_positive = pd.DataFrame({"user_id": sparse_matrix.row, "item_id": sparse_matrix.col, "label": 1})
    df_positive["user_id"] = df_positive["user_id"].map(idx2user)
    df_positive["item_id"] = df_positive["item_id"].map(idx2article)

    # Negative sampling
    existing_pairs = set(zip(df_positive["user_id"], df_positive["item_id"]))
    negatives = []
    np.random.seed(42)

    print("[•] Sampling negative interactions...")
    with tqdm(total=len(df_positive)) as pbar:
        while len(negatives) < len(df_positive):
            u = np.random.choice(all_user_ids)
            i = np.random.choice(all_item_ids)
            if (u, i) not in existing_pairs:
                negatives.append((u, i))
                existing_pairs.add((u, i))
                pbar.update(1)

    df_negative = pd.DataFrame(negatives, columns=["user_id", "item_id"])
    df_negative["label"] = 0

    # Combine positive and negative
    df_all = pd.concat([df_positive, df_negative], ignore_index=True)

    # Add timestamps
    articles_df = pd.read_parquet(articles_path)[["article_id", "published_time"]]
    df_all = df_all.merge(articles_df.rename(columns={"article_id": "item_id"}), on="item_id", how="left")
    df_all = df_all.dropna(subset=["published_time"])
    df_all["published_time"] = pd.to_datetime(df_all["published_time"])
    df_all = df_all.sort_values("published_time")

    # Save CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_all.to_csv(output_path, index=False)
    print(f"[✓] Created: {output_path} with {len(df_all)} rows")


# =========================
# PyTorch Dataset
# =========================
class InteractionDataset(Dataset):
    def __init__(self, df, user2idx, item2idx):
        self.users = df["user_id"].map(user2idx).values
        self.items = df["item_id"].map(item2idx).values
        self.labels = df["label"].values.astype(np.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# =========================
# NCF Model
# =========================
class NCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, user_ids, item_ids):
        u = self.user_embedding(user_ids)
        i = self.item_embedding(item_ids)
        x = torch.cat([u, i], dim=1)
        return self.model(x).squeeze()


# =========================
# Training Function
# =========================
def train_ncf(
    csv_path="./data/preprocessed/interactions_long_balanced.csv",
    model_path="./models/ncf.pth",
    mapping_path="./models/ncf_mappings.pth",
    epochs=5,
    batch_size=1024,
    lr=0.001,
):
    if os.path.exists(model_path) and os.path.exists(mapping_path):
        print("[✓] NCF model and mappings already exist – skipping training.")
        return

    create_balanced_interaction_df()

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[!] interactions_long.csv is empty. Aborting training.")
        return

    df["published_time"] = pd.to_datetime(df["published_time"])
    df = df.sort_values("published_time")

    user2idx = {uid: idx for idx, uid in enumerate(df["user_id"].unique())}
    item2idx = {iid: idx for idx, iid in enumerate(df["item_id"].unique())}
    idx2item = {v: k for k, v in item2idx.items()}

    split_idx = max(1, int(len(df) * 0.9))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:] if split_idx < len(df) else pd.DataFrame(columns=df.columns)

    train_set = InteractionDataset(train_df, user2idx, item2idx)
    val_set = InteractionDataset(val_df, user2idx, item2idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = NCF(len(user2idx), len(item2idx)).to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    device = next(model.parameters()).device

    print(f"Training NCF on {len(train_df)} samples | Epochs: {epochs}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user, item, label in train_loader:
            user, item, label = user.to(device), item.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(user, item)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    torch.save({"user2idx": user2idx, "item2idx": item2idx, "idx2item": idx2item}, mapping_path)
    print(f"[✓] Model saved to {model_path}")
    print(f"[✓] Mappings saved to {mapping_path}")
