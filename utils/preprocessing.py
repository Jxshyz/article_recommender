# Import standard and third-party libraries
import os
import gdown
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import load_npz, save_npz

# Import custom utilities
import utils.sparse_matrix as sparse
import utils.data_loader as data_loader

# Define data folders
data_folder = "data"
preprocessed_folder = os.path.join(data_folder, "preprocessed")


def get_parquets():
    """
    Download the raw dataset from Google Drive if not already present.

    Returns:
        dict: A dictionary with paths to each parquet file (articles, behavior, and history for train and validation).
    """
    folder_id = "1kGBWTm-a1alJh_1pFu9K-3hYcYkNPO-g"
    download_folder = os.path.join(data_folder, "ebnerd")

    # Download data only if the folder doesn't exist
    if not os.path.exists(download_folder):
        print(f"Downloading from Google Drive. Saving into '{download_folder}'.")
        gdown.download_folder(
            f"https://drive.google.com/drive/folders/{folder_id}", quiet=False, output=download_folder
        )

    # Return file paths
    return {
        "articles": os.path.join(download_folder, "articles.parquet"),
        "behaviour_train": os.path.join(download_folder, "train/behaviors.parquet"),
        "history_train": os.path.join(download_folder, "train/history.parquet"),
        "behaviour_validation": os.path.join(download_folder, "validation/behaviors.parquet"),
        "history_validation": os.path.join(download_folder, "validation/history.parquet"),
    }


def get_preprocessed_articles():
    """
    Load or generate preprocessed article data with text embeddings.

    Returns:
        pd.DataFrame: A DataFrame with cleaned and enriched article data including text embeddings.
    """
    articles_file = os.path.join(preprocessed_folder, "articles_cs.parquet")

    # Generate and save article embeddings if not already processed
    if not os.path.exists(articles_file):
        print(f"Preprocessing articles. Saving into '{articles_file}'.")
        files = get_parquets()
        articles = pd.read_parquet(files["articles"])

        # Clean and convert text and numerical fields
        articles["title"] = articles["title"].fillna("").astype(str)
        articles["subtitle"] = articles["subtitle"].fillna("").astype(str)
        articles["category_str"] = articles["category_str"].fillna("").astype(str)
        articles["body"] = articles["body"].fillna("").astype(str)
        articles["total_pageviews"] = articles["total_pageviews"].fillna(0)
        articles["published_time"] = pd.to_datetime(articles["published_time"])

        # Generate sentence embeddings
        from utils.embedding import generate_embeddings

        articles["embedding"] = generate_embeddings(articles)

        # Save to disk
        os.makedirs(os.path.dirname(articles_file), exist_ok=True)
        articles.to_parquet(articles_file)

    # Load preprocessed file
    return pd.read_parquet(articles_file)


def get_preprocessed_user_item_matrix():
    """
    Load or generate the user-item interaction matrix and associated mappings.

    Returns:
        tuple:
            - user_item_matrix (scipy.sparse.csr_matrix)
            - uim_u2i (dict): user_id -> matrix row index
            - uim_a2i (dict): article_id -> matrix column index
            - uim_i2u (dict): matrix row index -> user_id
            - uim_i2a (dict): matrix column index -> article_id
    """
    matrix_file = os.path.join(preprocessed_folder, "user_item_matrix.npz")
    mappings_file = os.path.join(preprocessed_folder, "uim_mappings.pkl")

    # If matrix and mappings don't exist, generate and save them
    if not os.path.exists(matrix_file) or not os.path.exists(mappings_file):
        print(f"Preprocessing user-item-matrix. Saving into '{matrix_file}'.")
        Articles, behaviour_test, history_test, behaviour_value, history_value = data_loader.load()

        # Create sparse matrix and mappings
        user_item_matrix, uim_u2i, uim_a2i, uim_i2u, uim_i2a = sparse.create_sparse(
            "data", Articles, behaviour_test, history_test, behaviour_value, history_value, matrix_file
        )

        # Save matrix and mappings
        os.makedirs(preprocessed_folder, exist_ok=True)
        save_npz(matrix_file, user_item_matrix)
        with open(mappings_file, "wb") as f:
            pickle.dump((uim_u2i, uim_a2i, uim_i2u, uim_i2a), f)

    # Load matrix and mappings
    user_item_matrix = load_npz(matrix_file)
    with open(mappings_file, "rb") as f:
        uim_u2i, uim_a2i, uim_i2u, uim_i2a = pickle.load(f)

    return user_item_matrix, uim_u2i, uim_a2i, uim_i2u, uim_i2a


def get_preprocessed_similarities(articles):
    """
    Load or generate cosine similarity matrix for article embeddings.

    Parameters:
        articles (pd.DataFrame): DataFrame with an 'embedding' column.

    Returns:
        tuple:
            - similarity_matrix (np.ndarray): Cosine similarity matrix.
            - sm_a2i (dict): article_id -> matrix index
            - sm_i2a (np.ndarray): matrix index -> article_id
    """
    matrix_file = os.path.join(preprocessed_folder, "similarity_matrix.pkl")

    # If similarity matrix not already generated, create and save it
    if not os.path.exists(matrix_file):
        print(f"Preprocessing cosine similarity matrix. Saving into '{matrix_file}'.")
        embeddings = np.vstack(articles["embedding"].values)  # Stack embeddings into matrix

        from sklearn.metrics.pairwise import cosine_similarity

        similarity_matrix = cosine_similarity(embeddings)

        sm_i2a = articles["article_id"].values  # index to article_id
        sm_a2i = {article_id: idx for idx, article_id in enumerate(sm_i2a)}  # article_id to index

        # Save the similarity matrix and mappings
        os.makedirs(os.path.dirname(matrix_file), exist_ok=True)
        with open(matrix_file, "wb") as f:
            pickle.dump((similarity_matrix, sm_a2i, sm_i2a), f)

    # Load similarity matrix and mappings
    with open(matrix_file, "rb") as f:
        similarity_matrix, sm_a2i, sm_i2a = pickle.load(f)

    return similarity_matrix, sm_a2i, sm_i2a


def get_train_user_item_matrix():
    matrix = load_npz("data/preprocessed/user_item_matrix_train.npz")
    with open("data/preprocessed/uim_train_mappings.pkl", "rb") as f:
        return (matrix, *pickle.load(f))


def get_val_user_item_matrix():
    matrix = load_npz("data/preprocessed/user_item_matrix_val.npz")
    with open("data/preprocessed/uim_val_mappings.pkl", "rb") as f:
        return (matrix, *pickle.load(f))
