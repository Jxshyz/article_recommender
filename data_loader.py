import os
import pandas as pd


def load():
    # Define folder and file paths
    download_folder = "data/ebnerd"
    files = {
        "articles": os.path.join(download_folder, "articles.parquet"),
        "behaviour_train": os.path.join(download_folder, "train/behaviors.parquet"),
        "history_train": os.path.join(download_folder, "train/history.parquet"),
        "behaviour_validation": os.path.join(download_folder, "validation/behaviors.parquet"),
        "history_validation": os.path.join(download_folder, "validation/history.parquet"),
    }

    # Load the raw datasets: Articles, Behaviors (test/validation), and History (test/validation).

    # Articles
    Articles = pd.read_parquet(files["articles"])

    # Test set
    Bhv_test = pd.read_parquet(files["behaviour_train"])
    Hstr_test = pd.read_parquet(files["history_train"])

    # Validation set
    Bhv_val = pd.read_parquet(files["behaviour_validation"])
    Hstr_val = pd.read_parquet(files["history_validation"])

    return Articles, Bhv_test, Hstr_test, Bhv_val, Hstr_val
