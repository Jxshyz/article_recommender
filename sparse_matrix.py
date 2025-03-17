import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from tqdm import tqdm


def data_exploration(data_folder):
    Bhv_test = pd.read_parquet(f"{data_folder}\\train\\behaviors.parquet")
    Hstr_test = pd.read_parquet(f"{data_folder}\\train\\history.parquet")
    Bhv_val = pd.read_parquet(f"{data_folder}\\validation\\behaviors.parquet")
    Hstr_val = pd.read_parquet(f"{data_folder}\\validation\\history.parquet")

    def safe_eval(x):
        try:
            return eval(x) if isinstance(x, str) else x
        except:
            return []

    print("=== Bhv_test Stats ===")
    print("\nScroll Percentage Stats:")
    print(Bhv_test["scroll_percentage"].describe())
    print(f"NaN in scroll_percentage: {Bhv_test['scroll_percentage'].isna().sum()}")

    print("\nRead Time Stats:")
    print(Bhv_test["read_time"].describe())
    print(f"NaN in read_time: {Bhv_test['read_time'].isna().sum()}")

    Bhv_test["clicked_match"] = Bhv_test.apply(
        lambda row: (
            row["article_id"] in safe_eval(row["article_ids_clicked"]) if pd.notna(row["article_id"]) else False
        ),
        axis=1,
    )
    print(f"\nRows where article_id matches clicked: {Bhv_test['clicked_match'].sum()} / {len(Bhv_test)}")

    print("\n=== Hstr_test Stats ===")
    Hstr_test["article_ids"] = Hstr_test["article_id_fixed"].apply(safe_eval)
    Hstr_test["read_times"] = Hstr_test["read_time_fixed"].apply(safe_eval)
    Hstr_test["scroll_percents"] = Hstr_test["scroll_percentage_fixed"].apply(safe_eval)

    Hstr_test_exploded = Hstr_test.explode("article_ids")
    Hstr_test_exploded["read_time"] = pd.to_numeric(Hstr_test["read_times"].explode(), errors="coerce")
    Hstr_test_exploded["scroll_percentage"] = pd.to_numeric(Hstr_test["scroll_percents"].explode(), errors="coerce")

    print("\nExploded Scroll Percentage Stats:")
    print(Hstr_test_exploded["scroll_percentage"].describe())
    print(f"NaN in scroll_percentage: {Hstr_test_exploded['scroll_percentage'].isna().sum()}")

    print("\nExploded Read Time Stats:")
    print(Hstr_test_exploded["read_time"].describe())
    print(f"NaN in read_time: {Hstr_test_exploded['read_time'].isna().sum()}")

    print("\n=== Bhv_val Stats ===")
    print("\nScroll Percentage Stats:")
    print(Bhv_val["scroll_percentage"].describe())
    print(f"NaN in scroll_percentage: {Bhv_val['scroll_percentage'].isna().sum()}")

    print("\nRead Time Stats:")
    print(Bhv_val["read_time"].describe())
    print(f"NaN in read_time: {Bhv_val['read_time'].isna().sum()}")

    Bhv_val["clicked_match"] = Bhv_val.apply(
        lambda row: (
            row["article_id"] in safe_eval(row["article_ids_clicked"]) if pd.notna(row["article_id"]) else False
        ),
        axis=1,
    )
    print(f"\nRows where article_id matches clicked: {Bhv_val['clicked_match'].sum()} / {len(Bhv_val)}")

    print("\n=== Hstr_val Stats ===")
    Hstr_val["article_ids"] = Hstr_val["article_id_fixed"].apply(safe_eval)
    Hstr_val["read_times"] = Hstr_val["read_time_fixed"].apply(safe_eval)
    Hstr_val["scroll_percents"] = Hstr_val["scroll_percentage_fixed"].apply(safe_eval)

    Hstr_val_exploded = Hstr_val.explode("article_ids")
    Hstr_val_exploded["read_time"] = pd.to_numeric(Hstr_val["read_times"].explode(), errors="coerce")
    Hstr_val_exploded["scroll_percentage"] = pd.to_numeric(Hstr_val["scroll_percents"].explode(), errors="coerce")

    print("\nExploded Scroll Percentage Stats:")
    print(Hstr_val_exploded["scroll_percentage"].describe())
    print(f"NaN in scroll_percentage: {Hstr_val_exploded['scroll_percentage'].isna().sum()}")

    print("\nExploded Read Time Stats:")
    print(Hstr_val_exploded["read_time"].describe())
    print(f"NaN in read_time: {Hstr_val_exploded['read_time'].isna().sum()}")


def create_sparse(
    data_folder, Articles, Bhv_test, Hstr_test, Bhv_val, Hstr_val, output_file="user_item_likes_matrix_all.npz"
):
    def safe_eval(x):
        try:
            return eval(x) if isinstance(x, str) else x
        except:
            return []

    Bhv_test = Bhv_test.dropna(subset=["article_id"])
    Bhv_test["article_id"] = Bhv_test["article_id"].astype(int)
    merged_test = Bhv_test.merge(Articles[["article_id", "body", "article_type"]], on="article_id", how="left")

    Hstr_test["article_id_fixed"] = Hstr_test["article_id_fixed"].apply(safe_eval)
    Hstr_test["read_time_fixed"] = Hstr_test["read_time_fixed"].apply(safe_eval)
    Hstr_test["scroll_percentage_fixed"] = Hstr_test["scroll_percentage_fixed"].apply(safe_eval)
    Hstr_test_exploded = Hstr_test.explode("article_id_fixed")
    Hstr_test_exploded["read_time"] = pd.to_numeric(Hstr_test["read_time_fixed"].explode(), errors="coerce")
    Hstr_test_exploded["scroll_percentage"] = pd.to_numeric(
        Hstr_test["scroll_percentage_fixed"].explode(), errors="coerce"
    )
    Hstr_test_exploded["article_id"] = pd.to_numeric(Hstr_test_exploded["article_id_fixed"], errors="coerce")
    Hstr_test_exploded = Hstr_test_exploded.dropna(subset=["article_id"])
    Hstr_test_exploded["article_id"] = Hstr_test_exploded["article_id"].astype(int)
    Hstr_test_merged = Hstr_test_exploded.merge(
        Articles[["article_id", "body", "article_type"]], on="article_id", how="left"
    )

    Bhv_val = Bhv_val.dropna(subset=["article_id"])
    Bhv_val["article_id"] = Bhv_val["article_id"].astype(int)
    merged_val = Bhv_val.merge(Articles[["article_id", "body", "article_type"]], on="article_id", how="left")

    Hstr_val["article_id_fixed"] = Hstr_val["article_id_fixed"].apply(safe_eval)
    Hstr_val["read_time_fixed"] = Hstr_val["read_time_fixed"].apply(safe_eval)
    Hstr_val["scroll_percentage_fixed"] = Hstr_val["scroll_percentage_fixed"].apply(safe_eval)
    Hstr_val_exploded = Hstr_val.explode("article_id_fixed")
    Hstr_val_exploded["read_time"] = pd.to_numeric(Hstr_val["read_time_fixed"].explode(), errors="coerce")
    Hstr_val_exploded["scroll_percentage"] = pd.to_numeric(
        Hstr_val["scroll_percentage_fixed"].explode(), errors="coerce"
    )
    Hstr_val_exploded["article_id"] = pd.to_numeric(Hstr_val_exploded["article_id_fixed"], errors="coerce")
    Hstr_val_exploded = Hstr_val_exploded.dropna(subset=["article_id"])
    Hstr_val_exploded["article_id"] = Hstr_val_exploded["article_id"].astype(int)
    Hstr_val_merged = Hstr_val_exploded.merge(
        Articles[["article_id", "body", "article_type"]], on="article_id", how="left"
    )

    combined_df = pd.concat([merged_test, Hstr_test_merged, merged_val, Hstr_val_merged], ignore_index=True)
    combined_df["body_length"] = combined_df["body"].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)

    def is_low_text_article(article_type, body_length):
        low_text_types = ["video"]
        return (article_type in low_text_types) or (body_length < 500)

    combined_df["is_low_text"] = combined_df.apply(
        lambda row: is_low_text_article(row["article_type"], row["body_length"]), axis=1
    )
    combined_df["scroll_percentage"] = pd.to_numeric(combined_df["scroll_percentage"], errors="coerce").fillna(0)
    combined_df["read_time"] = pd.to_numeric(combined_df["read_time"], errors="coerce").fillna(0)

    epsilon = 1e-6
    combined_df["adjusted_scroll"] = np.where(
        combined_df["body_length"] > epsilon,
        combined_df["scroll_percentage"] / ((combined_df["body_length"] + epsilon) / 1000),
        combined_df["scroll_percentage"],
    )

    user_stats = (
        combined_df.groupby("user_id")
        .agg(
            {
                "adjusted_scroll": lambda x: np.percentile(x.dropna(), 60),
                "read_time": lambda x: np.percentile(x.dropna(), 60),
            }
        )
        .rename(columns={"adjusted_scroll": "scroll_threshold", "read_time": "read_threshold"})
    )

    user_stats["scroll_threshold"] = user_stats["scroll_threshold"].fillna(0.05)
    user_stats["read_threshold"] = user_stats["read_threshold"].fillna(5)

    combined_df = combined_df.merge(user_stats, on="user_id", how="left")

    def infer_like(row):
        clicked = False
        if "article_ids_inview" in row and "article_ids_clicked" in row:
            inview = (
                eval(row["article_ids_inview"])
                if isinstance(row["article_ids_inview"], str)
                else row["article_ids_inview"]
            )
            clicked = (
                row["article_id"] in eval(row["article_ids_clicked"])
                if isinstance(row["article_ids_clicked"], str)
                else False
            )

        scroll_ok = row["adjusted_scroll"] > row["scroll_threshold"]
        read_ok = row["read_time"] > row["read_threshold"]

        if row["is_low_text"]:
            return read_ok or clicked
        elif row["body_length"] > 2000:
            return read_ok and (scroll_ok or clicked)
        else:
            return read_ok or (scroll_ok and clicked)

    combined_df["liked"] = combined_df.apply(infer_like, axis=1)

    print(f"Total interactions: {len(combined_df)}")
    print(f"Inferred likes: {combined_df['liked'].sum()}")
    print(f"Users with likes: {combined_df[combined_df['liked']]['user_id'].nunique()}")
    print(f"Articles with likes: {combined_df[combined_df['liked']]['article_id'].nunique()}")

    all_users = combined_df["user_id"].unique()
    all_articles = combined_df["article_id"].unique()
    n_users = len(all_users)
    n_articles = len(all_articles)

    user_to_idx = {uid: i for i, uid in enumerate(all_users)}
    article_to_idx = {aid: j for j, aid in enumerate(all_articles)}

    liked_interactions = combined_df[combined_df["liked"]][["user_id", "article_id"]].drop_duplicates()
    rows = [user_to_idx[uid] for uid in liked_interactions["user_id"]]
    cols = [article_to_idx[aid] for aid in liked_interactions["article_id"]]
    data = np.ones(len(liked_interactions), dtype=np.uint8)

    user_item_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_articles))

    print(f"Matrix shape: {user_item_matrix.shape} (users: {n_users}, articles: {n_articles})")
    print(f"Number of non-zero entries: {user_item_matrix.nnz}")
    print(f"Sparsity: {user_item_matrix.nnz / (n_users * n_articles):.6f}")

    save_npz(output_file, user_item_matrix)

    user_idx = user_to_idx[all_users[0]]
    liked_articles = user_item_matrix[user_idx].nonzero()[1]
    print(f"Articles liked by user {all_users[0]}: {[all_articles[idx] for idx in liked_articles]}")

    return user_item_matrix, user_to_idx, article_to_idx


def Sparse_exploration(npz_file_path, data_folder):
    loaded_matrix = load_npz(npz_file_path)
    n_users, n_articles = loaded_matrix.shape

    likes_per_user = loaded_matrix.sum(axis=1).A.ravel()
    users_without_likes = np.sum(likes_per_user == 0)
    print(f"Number of users without a liked article: {users_without_likes}")

    likes_per_article = loaded_matrix.sum(axis=0).A.ravel()
    articles_without_likes = np.sum(likes_per_article == 0)
    print(f"Number of articles without a like: {articles_without_likes}")

    Bhv_test = pd.read_parquet(f"{data_folder}\\train\\behaviors.parquet")
    Hstr_test = pd.read_parquet(f"{data_folder}\\train\\history.parquet")
    Bhv_val = pd.read_parquet(f"{data_folder}\\validation\\behaviors.parquet")
    Hstr_val = pd.read_parquet(f"{data_folder}\\validation\\history.parquet")

    combined_df = pd.concat(
        [
            Bhv_test[["user_id", "article_id"]],
            Hstr_test.explode("article_id_fixed")[["user_id", "article_id_fixed"]].rename(
                columns={"article_id_fixed": "article_id"}
            ),
            Bhv_val[["user_id", "article_id"]],
            Hstr_val.explode("article_id_fixed")[["user_id", "article_id_fixed"]].rename(
                columns={"article_id_fixed": "article_id"}
            ),
        ]
    ).dropna(subset=["article_id"])

    combined_df["article_id"] = combined_df["article_id"].astype(int)
    all_users = combined_df["user_id"].unique()
    user_to_idx = {uid: i for i, uid in enumerate(all_users)}
    user_interactions = combined_df.groupby("user_id")["article_id"].nunique()

    top_10_indices = np.argsort(likes_per_user)[::-1][:10]
    top_10_user_ids = [all_users[idx] for idx in top_10_indices]
    top_10_likes = [likes_per_user[idx] for idx in top_10_indices]
    top_10_total_interactions = [user_interactions.loc[uid] for uid in top_10_user_ids]

    print("\nTop 10 users with most liked articles and total interactions:")
    for user_id, like_count, total_count in zip(top_10_user_ids, top_10_likes, top_10_total_interactions):
        print(f"User {user_id}: {like_count} liked articles, {total_count} total articles interacted with")

    bottom_10_indices = np.argsort(likes_per_user)[:10]
    bottom_10_user_ids = [all_users[idx] for idx in bottom_10_indices]
    bottom_10_likes = [likes_per_user[idx] for idx in bottom_10_indices]
    bottom_10_total_interactions = [user_interactions.loc[uid] for uid in bottom_10_user_ids]

    print("\n10 users with smallest number of liked articles and total interactions:")
    for user_id, like_count, total_count in zip(bottom_10_user_ids, bottom_10_likes, bottom_10_total_interactions):
        print(f"User {user_id}: {like_count} liked articles, {total_count} total articles interacted with")

    random_indices = np.random.choice(n_users, 10, replace=False)
    random_user_ids = [all_users[idx] for idx in random_indices]
    random_likes = [likes_per_user[idx] for idx in random_indices]
    random_total_interactions = [user_interactions.loc[uid] for uid in random_user_ids]

    print("\n10 random users with liked articles and total interactions:")
    for user_id, like_count, total_count in zip(random_user_ids, random_likes, random_total_interactions):
        print(f"User {user_id}: {like_count} liked articles, {total_count} total articles interacted with")

    print(f"\nTotal users: {n_users}")
    print(f"Total articles: {n_articles}")
