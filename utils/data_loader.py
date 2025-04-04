# Import necessary libraries
import os
import pandas as pd


# Define a function to load multiple parquet files from a specified folder structure
def load():
    """
    Load article metadata and user behavior/history data from local parquet files.

    Returns:
        tuple: A tuple containing five pandas DataFrames:
            - Articles (pd.DataFrame): Metadata/content of the articles.
            - Bhv_test (pd.DataFrame): Training set - user behavior logs (e.g. impressions, clicks).
            - Hstr_test (pd.DataFrame): Training set - user reading history.
            - Bhv_val (pd.DataFrame): Validation set - user behavior logs.
            - Hstr_val (pd.DataFrame): Validation set - user reading history.
    """

    # Set the root folder where the data is stored
    download_folder = "data/ebnerd"

    # Define the paths to the relevant parquet files using dictionary keys
    files = {
        "articles": os.path.join(download_folder, "articles.parquet"),
        "behaviour_train": os.path.join(download_folder, "train/behaviors.parquet"),
        "history_train": os.path.join(download_folder, "train/history.parquet"),
        "behaviour_validation": os.path.join(download_folder, "validation/behaviors.parquet"),
        "history_validation": os.path.join(download_folder, "validation/history.parquet"),
    }

    # Load the parquet files into Pandas DataFrames
    Articles = pd.read_parquet(files["articles"])  # Metadata or content about articles
    Bhv_test = pd.read_parquet(files["behaviour_train"])  # Training set: user behavior logs (clicks etc.)
    Hstr_test = pd.read_parquet(files["history_train"])  # Training set: user reading history
    Bhv_val = pd.read_parquet(files["behaviour_validation"])  # Validation set: user behavior logs
    Hstr_val = pd.read_parquet(files["history_validation"])  # Validation set: user reading history

    # Return all loaded DataFrames as a tuple
    return Articles, Bhv_test, Hstr_test, Bhv_val, Hstr_val
