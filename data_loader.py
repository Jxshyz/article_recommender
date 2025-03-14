import pandas as pd

def load():
    # Define folder and file paths
    folder_name = "data"
    file1 = folder_name + "\\articles.parquet"
    file2 = folder_name + "\\train\\behaviors.parquet"
    file3 = folder_name + "\\train\\history.parquet"
    file4 = folder_name + "\\validation\\behaviors.parquet"
    file5 = folder_name + "\\validation\\history.parquet"

    # Load the raw datasets: Articles, Behaviors (test/validation), and History (test/validation).

    # Articles
    Articles = pd.read_parquet(file1)

    # Test set
    Bhv_test = pd.read_parquet(file2)
    Hstr_test = pd.read_parquet(file3)

    # Validation set
    Bhv_val = pd.read_parquet(file4)
    Hstr_val = pd.read_parquet(file5)

    return Articles, Bhv_test, Hstr_test, Bhv_val, Hstr_val