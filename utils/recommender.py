# Import necessary libraries
import time
import numpy as np
import pandas as pd
from datetime import timedelta
from collections import defaultdict, Counter
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from utils.ncf_model import NCF


def get_liked_items(user_id, uim_u2i, user_item_matrix, uim_i2a):
    """
    Get a list of article IDs liked by a specific user.

    Parameters:
        user_id (str): The user's ID.
        uim_u2i (dict): Mapping from user ID to matrix index.
        user_item_matrix (np.ndarray): User-item interaction matrix.
        uim_i2a (dict): Mapping from matrix column index to article ID.

    Returns:
        list: List of article IDs the user liked.
    """
    if user_id not in uim_u2i:
        return []
    user_idx = uim_u2i[user_id]
    liked_article_idxs = user_item_matrix[user_idx].nonzero()[1]
    return [uim_i2a[idx] for idx in liked_article_idxs]


def get_users_who_liked(article_id, uim_a2i, user_item_matrix, uim_i2u):
    """
    Get all users who interacted with a given article.

    Parameters:
        article_id (str): The article's ID.
        uim_a2i (dict): Mapping from article ID to matrix index.
        user_item_matrix (np.ndarray): User-item interaction matrix.
        uim_i2u (dict): Mapping from matrix row index to user ID.

    Returns:
        list: List of user IDs who liked the article.
    """
    if article_id not in uim_a2i:
        return []
    article_idx = uim_a2i[article_id]
    user_idxs_who_liked = user_item_matrix[:, article_idx].nonzero()[0]
    return [uim_i2u[idx] for idx in user_idxs_who_liked]


def get_similarity(article_id1, article_id2, sm_a2i, similarity_matrix):
    """
    Get similarity score between two articles based on a precomputed similarity matrix.

    Parameters:
        article_id1 (str): First article ID.
        article_id2 (str): Second article ID.
        sm_a2i (dict): Mapping from article ID to similarity matrix index.
        similarity_matrix (np.ndarray): Precomputed article similarity matrix.

    Returns:
        float or None: Similarity score or None if any ID is not found.
    """
    if article_id1 not in sm_a2i or article_id2 not in sm_a2i:
        return None
    return similarity_matrix[sm_a2i[article_id1], sm_a2i[article_id2]]


def baseline_filtering(articles, date=None, max_age=timedelta(days=7)):
    """
    Rank articles based on recent popularity (pageviews) and filter by publication date.

    Parameters:
        articles (pd.DataFrame): Articles with 'total_pageviews' and 'published_time'.
        date (datetime): Reference date for filtering.
        max_age (timedelta): Max age for articles to be considered.

    Returns:
        pd.DataFrame: Articles with normalized 'baseline_score'.
    """
    articles["baseline_score"] = articles["total_pageviews"]

    if date is not None:
        date = pd.to_datetime(date)
        end_date = date
        start_date = date - max_age

        # Zero out scores for articles outside the desired date range
        articles["baseline_score"] = np.where(
            (articles["published_time"] >= start_date) & (articles["published_time"] <= end_date),
            articles["baseline_score"],
            0.0,
        )

    # Log transform and normalize
    articles["baseline_score"] = np.log1p(articles["baseline_score"])
    scaler = MinMaxScaler()
    articles[["baseline_score"]] = scaler.fit_transform(articles[["baseline_score"]])
    return articles.sort_values(by="baseline_score", ascending=False)


def content_based_filtering(articles, user_id, uim_u2i, user_item_matrix, uim_i2a, sm_a2i, similarity_matrix):
    """
    Recommend articles similar to those a user has liked using content similarity.

    Returns:
        pd.DataFrame: Articles with 'contentbased_score' column.
    """
    liked_idxs = np.array(
        [sm_a2i[article_id] for article_id in get_liked_items(user_id, uim_u2i, user_item_matrix, uim_i2a)]
    )
    if len(liked_idxs) == 0:
        articles["contentbased_score"] = 0
        return articles

    # Average similarity of each article to all liked articles
    similarity_scores = similarity_matrix[:, liked_idxs]
    articles["contentbased_score"] = similarity_scores.mean(axis=1)
    articles["contentbased_score"] = np.power(articles["contentbased_score"], 1 / 2)  # Dampen high scores

    # Normalize scores
    scaler = MinMaxScaler()
    articles[["contentbased_score"]] = scaler.fit_transform(articles[["contentbased_score"]])
    return articles.sort_values(by="contentbased_score", ascending=False)


def collaborative_filtering(
    articles, user_id, uim_u2i, user_item_matrix, uim_i2u, uim_i2a, uim_a2i, sm_a2i, similarity_matrix, sm_i2a
):
    """
    Recommend articles based on users who liked similar articles (collaborative filtering).

    Returns:
        pd.DataFrame: Articles with 'collaborative_score' column.
    """
    liked_idxs = np.array(
        [sm_a2i[article_id] for article_id in get_liked_items(user_id, uim_u2i, user_item_matrix, uim_i2a)]
    )
    if len(liked_idxs) == 0:
        articles["collaborative_score"] = 0
        return articles

    # Get top similar articles for each liked article
    similarity_scores = similarity_matrix[liked_idxs]
    top_similar_idxs = np.argsort(similarity_scores, axis=1)[:, -5:][:, ::-1]
    similar_article_ids = set(sm_i2a[top_similar_idxs].flatten())

    # Find users who liked these similar articles
    similar_article_idxs = [uim_a2i[aid] for aid in similar_article_ids if aid in uim_a2i]
    similar_user_ids = [uim_i2u[idx] for idx in user_item_matrix[:, similar_article_idxs].nonzero()[0]]
    user_rank = Counter(similar_user_ids)

    # Collect scores based on how many similar articles a user liked
    user_ids = np.array(list(user_rank.keys()))
    user_idxs = [uim_u2i.get(uid) for uid in user_ids]
    scores = np.array(list(user_rank.values()))

    article_scores = defaultdict(float)
    for user_idx, score in zip(user_idxs, scores):
        liked_article_ids = uim_i2a[user_item_matrix[user_idx].nonzero()[1]]
        for article_id in liked_article_ids:
            article_scores[article_id] = max(article_scores.get(article_id, 0), score)

    # Map scores back to articles
    articles["collaborative_score"] = articles["article_id"].map(article_scores).fillna(0)
    articles["collaborative_score"] = np.log1p(articles["collaborative_score"])  # Smooth large values
    scaler = MinMaxScaler()
    articles[["collaborative_score"]] = scaler.fit_transform(articles[["collaborative_score"]])
    return articles.sort_values(by="collaborative_score", ascending=False)


def ncf_filtering(articles, user_id):

    # Load mappings first
    mappings = torch.load("./models/ncf_mappings.pth", weights_only=False)

    user2idx = mappings["user2idx"]
    item2idx = mappings["item2idx"]

    # Build model with correct shape
    model = NCF(n_users=len(user2idx), n_items=len(item2idx))
    model.load_state_dict(torch.load("./models/ncf.pth", map_location="cpu"))
    model.eval()

    if user_id not in user2idx:
        articles["ncf_score"] = 0
        return articles

    user_idx = user2idx[user_id]
    article_ids = articles["article_id"].values
    known_ids = [aid for aid in article_ids if aid in item2idx]

    item_indices = [item2idx[aid] for aid in known_ids]
    user_tensor = torch.tensor([user_idx] * len(item_indices))
    item_tensor = torch.tensor(item_indices)

    with torch.no_grad():
        scores = model(user_tensor, item_tensor).numpy()

    score_map = dict(zip(known_ids, scores))
    articles["ncf_score"] = articles["article_id"].map(score_map).fillna(0)

    scaler = MinMaxScaler()
    articles[["ncf_score"]] = scaler.fit_transform(articles[["ncf_score"]])
    return articles


def hybrid_filtering(articles):
    weights = {
        "baseline_score": 0.1,
        "contentbased_score": 0.2,
        "collaborative_score": 0.2,
        "ncf_score": 0.5,
    }
    for score in weights:
        if score not in articles.columns:
            articles[score] = 0

    articles["hybrid_score"] = sum(articles[score] * weight for score, weight in weights.items())
    return articles.sort_values(by="hybrid_score", ascending=False)


def recommend(
    articles,
    user_id,
    uim_u2i,
    user_item_matrix,
    uim_i2u,
    uim_i2a,
    uim_a2i,
    sm_a2i,
    similarity_matrix,
    sm_i2a,
    date=None,
):
    """
    Run the full recommendation pipeline (baseline, content-based, collaborative, hybrid).

    Returns:
        pd.DataFrame: Articles scored and sorted by different strategies.
    """
    total_start = time.time()

    # Baseline
    start = time.time()
    articles = baseline_filtering(articles=articles, date=date)
    print(f"\nBaseline:      {time.time() - start:.4f}")

    # Content-Based
    start = time.time()
    articles = content_based_filtering(
        articles=articles,
        user_id=user_id,
        uim_u2i=uim_u2i,
        user_item_matrix=user_item_matrix,
        uim_i2a=uim_i2a,
        sm_a2i=sm_a2i,
        similarity_matrix=similarity_matrix,
    )
    print(f"Content-Based: {time.time() - start:.4f}")

    # Collaborative
    start = time.time()
    articles = collaborative_filtering(
        articles=articles,
        user_id=user_id,
        uim_u2i=uim_u2i,
        user_item_matrix=user_item_matrix,
        uim_i2u=uim_i2u,
        uim_i2a=uim_i2a,
        uim_a2i=uim_a2i,
        sm_a2i=sm_a2i,
        similarity_matrix=similarity_matrix,
        sm_i2a=sm_i2a,
    )
    print(f"Collaborative: {time.time() - start:.4f}")

    # NCF
    start = time.time()
    articles = ncf_filtering(articles=articles, user_id=user_id)
    print(f"NCF:           {time.time() - start:.4f}")

    # Hybrid
    start = time.time()
    articles = hybrid_filtering(articles=articles)
    print(f"Hybrid:        {time.time() - start:.4f}")
    print(f"Total:         {time.time() - total_start:.4f} for user_id '{user_id}'\n")

    return articles
