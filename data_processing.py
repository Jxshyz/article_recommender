import os
import time
from datetime import datetime, timedelta
import gdown
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.sparse import load_npz, save_npz, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

import sparse_matrix as sparse
import data_loader

data_folder = "data"
preprocessed_folder = os.path.join(data_folder, "preprocessed")


def get_parquets():
    # Google Drive folder ID
    folder_id = "1kGBWTm-a1alJh_1pFu9K-3hYcYkNPO-g"
    download_folder = os.path.join(data_folder, "ebnerd")

    if not os.path.exists(download_folder):
        # See https://recsys.eb.dk/dataset/ for description of the dataset
        print(f"Downloading from Google Drive. This might take some time. Saving into '{download_folder}'.")
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
        print(f"Preprocessing articles. This might take some time. Saving into '{articles_file}'.")

        # load articles
        files = get_parquets()
        articles = pd.read_parquet(files["articles"])

        # clean up dataset
        articles["title"] = articles["title"].fillna("").astype(str)
        articles["subtitle"] = articles["subtitle"].fillna("").astype(str)
        articles["category_str"] = articles["category_str"].fillna("").astype(str)
        articles["body"] = articles["body"].fillna("").astype(str)
        articles["total_pageviews"] = articles["total_pageviews"].fillna(0)
        articles["published_time"] = pd.to_datetime(articles["published_time"])

        # compute embedding for every article
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

        # save preprocessed articles
        os.makedirs(os.path.dirname(articles_file), exist_ok=True)
        articles.to_parquet(articles_file)

    return pd.read_parquet(articles_file)


articles = get_preprocessed_articles()


def get_preprocessed_user_item_matrix():
    matrix_file = os.path.join(preprocessed_folder, "user_item_matrix.npz")
    mappings_file = os.path.join(preprocessed_folder, "uim_mappings.pkl")

    if not os.path.exists(matrix_file) or not os.path.exists(mappings_file):
        print(f"Preprocessing user-item-matrix. This might take some time. Saving into '{matrix_file}'.")

        Articles, behaviour_test, history_test, behaviour_value, history_value = data_loader.load()

        # compute user-item-matrix and save mappings from / to matrix index
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


# setup
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

user_item_matrix, uim_u2i, uim_a2i, uim_i2u, uim_i2a = get_preprocessed_user_item_matrix()


def get_liked_items(user_id):
    """Returns a list of article IDs liked by the given user."""
    if user_id not in uim_u2i:
        return []

    user_idx = uim_u2i[user_id]  # convert user_id to matrix index
    liked_article_idxs = user_item_matrix[user_idx].nonzero()[1]  # Get indices of articles liked by the user
    return [uim_i2a[idx] for idx in liked_article_idxs]  # Convert indices back to article_id


def get_users_who_liked(article_id):
    """Returns a list of user IDs who liked the given article."""
    if article_id not in uim_a2i:
        return []

    article_idx = uim_a2i[article_id]  # convert article_id to matrix index
    user_idxs_who_liked = user_item_matrix[:, article_idx].nonzero()[0]  # Get indices of users who liked the article
    return [uim_i2u[idx] for idx in user_idxs_who_liked]  # Convert indices back to user_id


def get_preprocessed_similarities():
    matrix_file = os.path.join(preprocessed_folder, "similarity_matrix.pkl")

    if not os.path.exists(matrix_file):
        print(f"Preprocessing cosine similarity matrix. This might take some time. Saving into '{matrix_file}'.")

        embeddings = np.vstack(articles["embedding"].values)

        # compute pairwise cosine similarities and save mappings from / to matrix index
        sm_i2a = articles["article_id"].values
        similarity_matrix = cosine_similarity(embeddings)
        sm_a2i = {article_id: idx for idx, article_id in enumerate(sm_i2a)}

        os.makedirs(os.path.dirname(matrix_file), exist_ok=True)

        with open(matrix_file, "wb") as f:
            pickle.dump((similarity_matrix, sm_a2i, sm_i2a), f)

    with open(matrix_file, "rb") as f:
        similarity_matrix, sm_a2i, sm_i2a = pickle.load(f)

    return similarity_matrix, sm_a2i, sm_i2a


similarity_matrix, sm_a2i, sm_i2a = get_preprocessed_similarities()


def get_similarity(article_id1, article_id2):
    if article_id1 not in sm_a2i or article_id2 not in sm_a2i:
        return None
    return similarity_matrix[sm_a2i[article_id1], sm_a2i[article_id2]]


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
    liked_idxs = np.array([sm_a2i[article_id] for article_id in get_liked_items(user_id)])

    # final score of an item is the mean of all cosine similarities between this item and all liked items
    similarity_scores = similarity_matrix[:, liked_idxs]  # Shape: (num_articles, num_liked_articles)
    articles["contentbased_score"] = similarity_scores.mean(axis=1)  # Shape: (num_articles,)

    return articles.sort_values(by="contentbased_score", ascending=False)


def collaborative_filtering(articles, user_id):
    liked_idxs = np.array([sm_a2i[article_id] for article_id in get_liked_items(user_id)])

    # find all items that are similar to the liked items
    similarity_scores = similarity_matrix[liked_idxs]  # Shape: (num_liked_articles, num_articles)
    top_similar_idxs = np.argsort(similarity_scores, axis=1)[:, -5:][:, ::-1]  # Shape: (num_liked_articles, 5)
    similar_article_ids = set(sm_i2a[top_similar_idxs].flatten())  # Shape: (<= num_liked_articles * 5,)
    # find all users who liked the similar items
    similar_article_idxs = [uim_a2i[article_id] for article_id in similar_article_ids if article_id in uim_a2i]
    similar_user_ids = [uim_i2u[idx] for idx in user_item_matrix[:, similar_article_idxs].nonzero()[0]]

    # count how many times each user liked a similar item
    user_rank = Counter(similar_user_ids)
    user_ids = np.array(list(user_rank.keys()))
    user_idxs = [uim_u2i.get(user_id) for user_id in user_ids]
    scores = np.array(list(user_rank.values()))
    scores = scores / scores.max() if scores.size > 0 else scores  # normalize scores

    # final score of an item is the maximum normalized score of all users who liked it
    article_scores = defaultdict(float)
    for user_idx, score in zip(user_idxs, scores):
        # below is the most time-consuming LOC. However, it cannot be batched due to the necessary call to nonzero
        liked_article_ids = uim_i2a[user_item_matrix[user_idx].nonzero()[1]]
        for article_id in liked_article_ids:
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
articles = content_based_filtering(articles=articles, user_id=151570)  # randomly picked user_id
print(f"Content-Based: {time.time() - start:4f}")
start = time.time()
articles = collaborative_filtering(articles=articles, user_id=151570)  # randomly picked user_id
print(f"Collaborative: {time.time() - start:4f}")
start = time.time()
articles = hybrid_filtering(articles=articles)
print(f"Hybrid:        {time.time() - start:4f}")
print()
print(f"Total:         {time.time() - total_start:4f}\n")

print(f"\nFIRST 10:")
for _, row in articles.head(10).iterrows():
    print(
        f"ID: {row['article_id']}, BL {row['baseline_score']:.4f} / CBF {row['contentbased_score']:.4f} / CF {row['collaborative_score']:.4f} / HF {row['hybrid_score']:.4f}, Title: {row['title']}"
    )

print(f"\nLAST 10:")
for _, row in articles.tail(10).iterrows():
    print(
        f"ID: {row['article_id']}, BL {row['baseline_score']:.4f} / CBF {row['contentbased_score']:.4f} / CF {row['collaborative_score']:.4f} / HF {row['hybrid_score']:.4f}, Title: {row['title']}"
    )
