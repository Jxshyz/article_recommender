import os
import numpy as np
from collections import Counter
import pandas as pd
from datetime import datetime, timedelta
import gdown


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


# setup
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


def get_liked_items(user):
    # Returns a list of ids representing the items the given user has liked
    # TODO stub - implement correctly using user-item-matrix M
    return baseline_filtering().head(10)["article_id"].tolist()


def get_users_who_liked(item):
    # Returns a list of ids representing the users that have liked the given item
    # TODO stub - implement correctly using user-item-matrix M
    return [1, 2, 3]


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def baseline_filtering(date=None, max_age=timedelta(days=7)):
    articles = get_preprocessed_articles()

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


def content_based_filtering(user_id="DUMMY_USER_ID"):
    articles = get_preprocessed_articles()

    liked_articles = articles[articles["article_id"].isin(get_liked_items(user_id))]

    # compute mean cosine similarity between all articles and liked_articles
    articles["contentbased_score"] = articles["embedding"].apply(
        lambda x: np.mean([cosine_similarity(x, y) for y in liked_articles["embedding"]])
    )

    return articles.sort_values(by="contentbased_score", ascending=False)


def collaborative_filtering(user_id="DUMMY_USER_ID"):
    articles = get_preprocessed_articles()

    liked_articles = articles[articles["article_id"].isin(get_liked_items(user_id))]
    similar_liked_articles = set()

    for _, liked_article in liked_articles.iterrows():
        articles["similarity"] = articles["embedding"].apply(
            lambda article: cosine_similarity(liked_article["embedding"], article)
        )
        similar_liked_articles.update(articles.nlargest(5, "similarity")["article_id"].tolist())

    same_likes = []
    for article_id in similar_liked_articles:
        same_likes.extend(get_users_who_liked(article_id))

    user_rank = Counter(same_likes)

    user_data = []

    for user_id, score in user_rank.items():
        user_data.append({"user_id": user_id, "score": score, "article_ids": get_liked_items(user_id)})

    user_data = sorted(user_data, key=lambda x: x["score"], reverse=True)

    max_score = max(entry["score"] for entry in user_data)
    for user in user_data:
        user["normalized_score"] = user["score"] / max_score

    article_scores = {article_id: 0.0 for article_id in articles["article_id"]}
    for user in user_data:
        for article_id in user["article_ids"]:
            article_scores[article_id] = max(article_scores[article_id], user["normalized_score"])

    articles["collaborative_score"] = articles["article_id"].map(article_scores)

    return articles.sort_values(by="collaborative_score", ascending=False)


# for _, row in baseline_filtering().head(100).iterrows():
#     print(f"ID: {row['article_id']}, Title: {row['title']}, baseline: {row['baseline_score']:.4f}")

# for _, row in content_based_filtering(user_id="DUMMY").head(100).iterrows():
#     print(f"ID: {row['article_id']}, Title: {row['title']}, content-based: {row['contentbased_score']:.4f}")

# for _, row in collaborative_filtering(user_id="DUMMY").head(100).iterrows():
#     print(f"ID: {row['article_id']}, Title: {row['title']}, collaborative: {row['collaborative_score']:.4f}")
