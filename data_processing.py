import os
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
from datetime import datetime, timedelta
import gdown

import time


def get_parquets():
    # Google Drive folder ID
    folder_id = "1kGBWTm-a1alJh_1pFu9K-3hYcYkNPO-g"

    # Define folder name for the download
    data_folder = "data"

    if not os.path.exists(data_folder):
        print(f"Downloading into '{data_folder}' from Google Drive...")
        gdown.download_folder(f"https://drive.google.com/drive/folders/{folder_id}", quiet=False, output=data_folder)

    else:
        print(f"Folder '{data_folder}' already exists. Skipping download.")

    return {
        "articles": data_folder + "\\articles.parquet",
        "behaviour_train": data_folder + "\\train\\behaviors.parquet",
        "history_train": data_folder + "\\train\\history.parquet",
        "behaviour_validation": data_folder + "\\validation\\behaviors.parquet",
        "history_validation": data_folder + "\\validation\\history.parquet",
    }


def get_preprocessed_articles():
    preprocessed_path = "data/preprocessed/articles_cs.parquet"
    if not os.path.exists(preprocessed_path):
        print(f"Preprocessing articles. This might take some time. Saving into '{preprocessed_path}'.")
        files = get_parquets()
        # See https://recsys.eb.dk/dataset/ for description of the dataset
        articles = pd.read_parquet(files["articles"])
        # behaviour_train = pd.read_parquet(files["behaviour_train"])
        # history_train = pd.read_parquet(files["history_train"])
        # behaviour_validation = pd.read_parquet(files["behaviour_validation"])
        # history_validation = pd.read_parquet(files["history_validation"])

        articles["title"] = articles["title"].fillna("").astype(str)
        articles["subtitle"] = articles["subtitle"].fillna("").astype(str)
        articles["category_str"] = articles["category_str"].fillna("").astype(str)
        articles["body"] = articles["body"].fillna("").astype(str)
        articles["total_pageviews"] = articles["total_pageviews"].fillna(0)
        articles["published_time"] = pd.to_datetime(articles["published_time"])

        articles["aggregated_text"] = articles[["title", "subtitle", "category_str", "body"]].apply(
            lambda x: f"TITLE: {x['title']}\n\n\nSUBTITLE: {x['subtitle']}\n\n\nCATEGORY: {x['category_str']}\n\n\nCONTENT: {x['body']}",
            axis=1,
        )

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        def batch_embed(texts, batch_size=1000):
            embeddings = []
            for i in range(0, len(texts), batch_size):
                print(f"Embedding in process - embedded {i}/{len(texts)}")
                embeddings.extend(model.encode(texts[i : i + batch_size]))
            return np.array(embeddings)

        articles["embedding"] = list(batch_embed(articles["aggregated_text"].tolist()))

        articles.drop(columns=["aggregated_text"], inplace=True)

        os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
        articles.to_parquet(preprocessed_path)

    return pd.read_parquet(preprocessed_path)


articles = get_preprocessed_articles()

import pickle
from scipy.sparse import load_npz, save_npz, csr_matrix
import sparse_matrix as sparse
import data_loader


def get_preprocessed_user_item_matrix():
    output_dir = "data/preprocessed"
    matrix_file = os.path.join(output_dir, "user_item_matrix.npz")
    mappings_file = os.path.join(output_dir, "mappings.pkl")

    if not os.path.exists(matrix_file) or not os.path.exists(mappings_file):
        print(f"Preprocessing user-item-matrix. This might take some time. Saving into '{matrix_file}'.")

        Articles, Bhv_test, Hstr_test, Bhv_val, Hstr_val = data_loader.load()
        user_item_matrix, user_to_idx, article_to_idx, all_users, all_articles = sparse.create_sparse(
            "data", Articles, Bhv_test, Hstr_test, Bhv_val, Hstr_val, matrix_file
        )

        os.makedirs(output_dir, exist_ok=True)

        save_npz(matrix_file, user_item_matrix)
        with open(mappings_file, "wb") as f:
            pickle.dump((user_to_idx, article_to_idx, all_users, all_articles), f)

    user_item_matrix = load_npz(matrix_file)
    with open(mappings_file, "rb") as f:
        user_to_idx, article_to_idx, all_users, all_articles = pickle.load(f)
    return user_item_matrix, user_to_idx, article_to_idx, all_users, all_articles


# setup
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

user_item_matrix, user_to_idx, article_to_idx, all_users, all_articles = get_preprocessed_user_item_matrix()


def get_liked_items(user_id):
    """Returns a list of article IDs liked by the given user."""
    if user_id not in user_to_idx:
        return []

    user_idx = user_to_idx[user_id]  # convert user_id to matrix index
    liked_articles = user_item_matrix[user_idx].nonzero()[1]  # Get article matrix indices
    return [all_articles[idx] for idx in liked_articles]  # Convert back to article_id


def get_users_who_liked(article_id):
    """Returns a list of user IDs who liked the given article."""
    if article_id not in article_to_idx:
        return []

    article_idx = article_to_idx[article_id]  # convert article_id to matrix index
    liked_users = user_item_matrix[:, article_idx].nonzero()[0]  # Get user matrix indices
    return [all_users[idx] for idx in liked_users]  # Convert back to user_id


from sklearn.metrics.pairwise import cosine_similarity


def get_preprocessed_similarities():
    matrix_file = "data/preprocessed/similarity_matrix.pkl"

    if not os.path.exists(matrix_file):
        print(f"Preprocessing cosine similarity matrix. This might take some time. Saving into '{matrix_file}'.")

        embeddings = np.vstack(articles["embedding"].values)
        all_articleids = articles["article_id"].values

        similarity_matrix = cosine_similarity(embeddings)
        articleid_to_index = {article_id: idx for idx, article_id in enumerate(all_articleids)}

        os.makedirs(os.path.dirname(matrix_file), exist_ok=True)

        with open(matrix_file, "wb") as f:
            pickle.dump((similarity_matrix, articleid_to_index, all_articleids), f)

    with open(matrix_file, "rb") as f:
        similarity_matrix, articleid_to_index, all_articleids = pickle.load(f)

    return similarity_matrix, articleid_to_index, all_articleids


similarity_matrix, articleid_to_index, all_articleids = get_preprocessed_similarities()


def get_similarity(id1, id2):
    if id1 not in articleid_to_index or id2 not in articleid_to_index:
        return None
    return similarity_matrix[articleid_to_index[id1], articleid_to_index[id2]]


def baseline_filtering(articles, date=None, max_age=timedelta(days=7)):
    max_pageviews = articles["total_pageviews"].max()
    articles["baseline_score"] = articles["total_pageviews"] / max_pageviews

    if date is not None:
        date = pd.to_datetime(date)
        end_date = date
        start_date = date - max_age

        articles["baseline_score"] = np.where(
            (articles["published_time"] >= start_date) & (articles["published_time"] <= end_date),
            articles["baseline_score"],
            0.0,
        )

    return articles.sort_values(by="baseline_score", ascending=False)


def content_based_filtering(articles, user_id):
    liked_articles = articles[articles["article_id"].isin(get_liked_items(user_id))]
    liked_idxs = np.array([articleid_to_index[article_id] for article_id in liked_articles["article_id"]])

    # final score of an item is the mean of all cosine similarities between this item and all liked items
    similarity_scores = similarity_matrix[:, liked_idxs]  # Shape: (num_articles, num_liked_articles)
    articles["contentbased_score"] = similarity_scores.mean(axis=1)  # Shape: (num_articles,)

    return articles.sort_values(by="contentbased_score", ascending=False)


def collaborative_filtering(articles, user_id):
    liked_article_ids = get_liked_items(user_id)
    liked_indices = np.array([articleid_to_index[article_id] for article_id in liked_article_ids])

    # find all items that are similar to the liked items
    similarity_scores = similarity_matrix[liked_indices]  # Shape: (num_liked_articles, num_articles)
    top_similar_indices = np.argsort(similarity_scores, axis=1)[:, -5:][:, ::-1]  # Shape: (num_liked_articles, 5)
    similar_liked_articles = set(all_articleids[top_similar_indices].flatten())  # Shape: (<= num_liked_articles * 5,)
    # find all users who liked the similar items
    similar_articles_idx = [
        article_to_idx[article_id] for article_id in similar_liked_articles if article_id in article_to_idx
    ]
    same_likes = [all_users[idx] for idx in user_item_matrix[:, similar_articles_idx].nonzero()[0]]

    # count how many times each user liked a similar item
    user_rank = Counter(same_likes)
    user_ids = np.array(list(user_rank.keys()))
    user_idxs = [user_to_idx.get(user_id) for user_id in user_ids]
    scores = np.array(list(user_rank.values()))
    scores = scores / scores.max() if scores.size > 0 else scores  # normalize scores

    # final score of an item is the maximum normalized score of all users who liked it
    article_scores = defaultdict(float)
    for uidx, score in zip(user_idxs, scores):
        # below is the most time-consuming LOC. However, it cannot be batched due to the necessary call to nonzero
        liked_articles = all_articles[user_item_matrix[uidx].nonzero()[1]]
        for article_id in liked_articles:
            article_scores[article_id] = max(article_scores.get(article_id, 0), score)
    articles["collaborative_score"] = articles["article_id"].map(article_scores).fillna(0)

    return articles.sort_values(by="collaborative_score", ascending=False)


def hybrid_filtering(articles):
    weights = {"baseline_score": 0.5, "contentbased_score": 0.25, "collaborative_score": 0.25}

    # final score is the weighted sum of the other scores
    articles["hybrid_score"] = (
        articles["baseline_score"] * weights["baseline_score"]
        + articles["contentbased_score"] * weights["contentbased_score"]
        + articles["collaborative_score"] * weights["collaborative_score"]
    )

    return articles.sort_values(by="hybrid_score", ascending=False)


total_start = time.time()
start = time.time()
articles = baseline_filtering(articles=articles)
print(f"Baseline:      {time.time() - start:4f}")
start = time.time()
articles = content_based_filtering(articles=articles, user_id=151570)  # randomly picked user id
print(f"Content-Based: {time.time() - start:4f}")
start = time.time()
articles = collaborative_filtering(articles=articles, user_id=151570)  # randomly picked user id
print(f"Collaborative: {time.time() - start:4f}")
start = time.time()
articles = hybrid_filtering(articles=articles)
print(f"Hybrid:        {time.time() - start:4f}")
print()
print(f"Total:         {time.time() - total_start:4f}")

for _, row in articles.head(10).iterrows():
    print(
        f"ID: {row['article_id']}, BL {row['baseline_score']:.4f} / CBF {row['contentbased_score']:.4f} / CF {row['collaborative_score']:.4f} / HF {row['hybrid_score']:.4f}, Title: {row['title']}"
    )

print(f"\n\n\n\n\nLAST 10:\n")

for _, row in articles.tail(10).iterrows():
    print(
        f"ID: {row['article_id']}, BL {row['baseline_score']:.4f} / CBF {row['contentbased_score']:.4f} / CF {row['collaborative_score']:.4f} / HF {row['hybrid_score']:.4f}, Title: {row['title']}"
    )
