import os
import pandas as pd
from datetime import datetime, timedelta
import gdown


def ensure_download():
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


# setup
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

files = ensure_download()

# See https://recsys.eb.dk/dataset/ for description of the dataset
articles = pd.read_parquet(files["articles"])
# behaviour_train = pd.read_parquet(files["behaviour_train"])
# history_train = pd.read_parquet(files["history_train"])
# behaviour_validation = pd.read_parquet(files["behaviour_validation"])
# history_validation = pd.read_parquet(files["history_validation"])

# preprocessing
articles["total_pageviews"] = articles["total_pageviews"].fillna(0)
articles["published_time"] = pd.to_datetime(articles["published_time"])


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

    filtered_articles = articles.copy()

    if date:
        date = pd.to_datetime(date)
        filtered_articles = filtered_articles[
            (filtered_articles["published_time"] >= date - max_age) & (filtered_articles["published_time"] <= date)
        ]

    return filtered_articles.sort_values(by="total_pageviews", ascending=False).head(n)


print(most_popular(5, "2023-04-17"))
print(most_popular(5))
