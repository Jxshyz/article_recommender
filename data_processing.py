import os
import pandas as pd
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


files = ensure_download()

#   -->      Datasets     <--  #

# Articles
Articles = pd.read_parquet(files["articles"])

# Test set
Bhv_test = pd.read_parquet(files["behaviour_train"])
Hstr_test = pd.read_parquet(files["history_train"])

# Validation set
Bhv_val = pd.read_parquet(files["behaviour_validation"])
Hstr_val = pd.read_parquet(files["history_validation"])
