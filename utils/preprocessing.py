import os
import gdown
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import load_npz, save_npz
import utils.sparse_matrix as sparse
import utils.data_loader as data_loader

data_folder = "data"
preprocessed_folder = os.path.join(data_folder, "preprocessed")


def get_parquets():
    folder_id = "1kGBWTm-a1alJh_1pFu9K-3hYcYkNPO-g"
    download_folder = os.path.join(data_folder, "ebnerd")
    if not os.path.exists(download_folder):
        print(f"Downloading from Google Drive. Saving into '{download_folder}'.")
        gdown.download_folder(
            f"https://drive.google.com/drive/folders/{folder_id}", quiet=False, output=download_folder
        )
    return {
        "articles": os.path.join(download_folder, "articles.parquet"),
        "behaviour_train": os.path.join(download_folder, "train/behaviors.parquet"),
        "history_train": os.path.join(download_folder, "train/history.parquet"),
        "behaviour_validation": os.path.join(download_folder, "validation/behaviors.parquet"),
        "history_validation": os.path.join(download_folder, "validation/history.parquet"),
    }


def get_preprocessed_articles():
    articles_file = os.path.join(preprocessed_folder, "articles_cs.parquet")
    if not os.path.exists(articles_file):
        print(f"Preprocessing articles. Saving into '{articles_file}'.")
        files = get_parquets()
        articles = pd.read_parquet(files["articles"])
        articles["title"] = articles["title"].fillna("").astype(str)
        articles["subtitle"] = articles["subtitle"].fillna("").astype(str)
        articles["category_str"] = articles["category_str"].fillna("").astype(str)
        articles["body"] = articles["body"].fillna("").astype(str)
        articles["total_pageviews"] = articles["total_pageviews"].fillna(0)
        articles["published_time"] = pd.to_datetime(articles["published_time"])
        from utils.embedding import generate_embeddings

        articles["embedding"] = generate_embeddings(articles)
        os.makedirs(os.path.dirname(articles_file), exist_ok=True)
        articles.to_parquet(articles_file)
    return pd.read_parquet(articles_file)


def get_preprocessed_user_item_matrix():
    matrix_file = os.path.join(preprocessed_folder, "user_item_matrix.npz")
    mappings_file = os.path.join(preprocessed_folder, "uim_mappings.pkl")
    if not os.path.exists(matrix_file) or not os.path.exists(mappings_file):
        print(f"Preprocessing user-item-matrix. Saving into '{matrix_file}'.")
        Articles, behaviour_test, history_test, behaviour_value, history_value = data_loader.load()
        user_item_matrix, uim_u2i, uim_a2i, uim_i2u, uim_i2a = sparse.create_sparse(
            "data", Articles, behaviour_test, history_test, behaviour_value, history_value, matrix_file
        )
        os.makedirs(preprocessed_folder, exist_ok=True)
        save_npz(matrix_file, user_item_matrix)
        with open(mappings_file, "wb") as f:
            pickle.dump((uim_u2i, uim_a2i, uim_i2u, uim_i2a), f)
    user_item_matrix = load_npz(matrix_file)
    with open(mappings_file, "rb") as f:
        uim_u2i, uim_a2i, uim_i2u, uim_i2a = pickle.load(f)
    return user_item_matrix, uim_u2i, uim_a2i, uim_i2u, uim_i2a


def get_preprocessed_similarities(articles):
    matrix_file = os.path.join(preprocessed_folder, "similarity_matrix.pkl")
    if not os.path.exists(matrix_file):
        print(f"Preprocessing cosine similarity matrix. Saving into '{matrix_file}'.")
        embeddings = np.vstack(articles["embedding"].values)
        from sklearn.metrics.pairwise import cosine_similarity

        sm_i2a = articles["article_id"].values
        similarity_matrix = cosine_similarity(embeddings)
        sm_a2i = {article_id: idx for idx, article_id in enumerate(sm_i2a)}
        os.makedirs(os.path.dirname(matrix_file), exist_ok=True)
        with open(matrix_file, "wb") as f:
            pickle.dump((similarity_matrix, sm_a2i, sm_i2a), f)
    with open(matrix_file, "rb") as f:
        similarity_matrix, sm_a2i, sm_i2a = pickle.load(f)
    return similarity_matrix, sm_a2i, sm_i2a
