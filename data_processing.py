import os
import numpy as np
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


def print_dataframe(df, cols=["article_id", "title", "published_time", "total_pageviews", "total_read_time"]):
    print(df[cols])


def most_popular(n=5, date=None, max_age=timedelta(days=7)):
    """
    Baseline recommender. Suggests the most popular articles, based on 'total_pageviews'.

    Args:
        n (int, optional): Amount of returned articles. Defaults to 5.
        date (str, optional): Date of recommendation. Format is YYYY-MM-DD. Defaults to None.
        max_age (timedelta, optional): Maximum age of recommended articles. Defaults to timedelta(days=7).

    Returns:
        pd.DataFrame: A DataFrame containing most popular articles
    """

    filtered_articles = get_preprocessed_articles()

    if date:
        date = pd.to_datetime(date)
        filtered_articles = filtered_articles[
            (filtered_articles["published_time"] >= date - max_age) & (filtered_articles["published_time"] <= date)
        ]

    return filtered_articles.sort_values(by="total_pageviews", ascending=False).head(n)


# print_dataframe(most_popular(5, "2023-04-17"))
# print_dataframe(most_popular(1))


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_liked_items(user):
    # Returns a list of ids representing the items the given user has liked
    # TODO stub - implement correctly using user-item-matrix M
    return most_popular(10)["article_id"].tolist()


# => Loading articles time (len 20738): 1177.0574 seconds (without embeddings pre-computed)
# => Loading articles time (len 20738): 0.5313 seconds (with precomputed embeddings, just loading from disk)


def content_based_filtering(user_id="DUMMY_USER_ID", n=5):
    articles = get_preprocessed_articles()

    liked_articles = articles[articles["article_id"].isin(get_liked_items(user_id))]

    # compute mean cosine similarity between all articles and liked_articles
    articles["mean_similarity"] = articles["embedding"].apply(
        lambda x: np.mean([cosine_similarity(x, y) for y in liked_articles["embedding"]])
    )

    return articles.sort_values(by="mean_similarity", ascending=False).head(n)


for _, row in content_based_filtering(user_id="DUMMY", n=100).iterrows():
    print(f"ID: {row['article_id']}, Title: {row['title']}, Similarity: {row['mean_similarity']:.4f}")
