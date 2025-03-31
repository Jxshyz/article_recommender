import time
import numpy as np
from datetime import timedelta
from collections import defaultdict, Counter
from sklearn.preprocessing import MinMaxScaler


def get_liked_items(user_id, uim_u2i, user_item_matrix, uim_i2a):
    if user_id not in uim_u2i:
        return []
    user_idx = uim_u2i[user_id]
    liked_article_idxs = user_item_matrix[user_idx].nonzero()[1]
    return [uim_i2a[idx] for idx in liked_article_idxs]


def get_users_who_liked(article_id, uim_a2i, user_item_matrix, uim_i2u):
    if article_id not in uim_a2i:
        return []
    article_idx = uim_a2i[article_id]
    user_idxs_who_liked = user_item_matrix[:, article_idx].nonzero()[0]
    return [uim_i2u[idx] for idx in user_idxs_who_liked]


def get_similarity(article_id1, article_id2, sm_a2i, similarity_matrix):
    if article_id1 not in sm_a2i or article_id2 not in sm_a2i:
        return None
    return similarity_matrix[sm_a2i[article_id1], sm_a2i[article_id2]]


def baseline_filtering(articles, date=None, max_age=timedelta(days=7)):
    articles["baseline_score"] = articles["total_pageviews"]
    if date is not None:
        date = pd.to_datetime(date)
        end_date = date
        start_date = date - max_age
        articles["baseline_score"] = np.where(
            (articles["published_time"] >= start_date) & (articles["published_time"] <= end_date),
            articles["baseline_score"],
            0.0,
        )
    articles["baseline_score"] = np.log1p(articles["baseline_score"])
    scaler = MinMaxScaler()
    articles[["baseline_score"]] = scaler.fit_transform(articles[["baseline_score"]])
    return articles.sort_values(by="baseline_score", ascending=False)


def content_based_filtering(articles, user_id, uim_u2i, user_item_matrix, uim_i2a, sm_a2i, similarity_matrix):
    liked_idxs = np.array(
        [sm_a2i[article_id] for article_id in get_liked_items(user_id, uim_u2i, user_item_matrix, uim_i2a)]
    )
    if len(liked_idxs) == 0:
        articles["contentbased_score"] = 0
        return articles
    similarity_scores = similarity_matrix[:, liked_idxs]
    articles["contentbased_score"] = similarity_scores.mean(axis=1)
    articles["contentbased_score"] = np.power(articles["contentbased_score"], 1 / 2)
    scaler = MinMaxScaler()
    articles[["contentbased_score"]] = scaler.fit_transform(articles[["contentbased_score"]])
    return articles.sort_values(by="contentbased_score", ascending=False)


def collaborative_filtering(
    articles, user_id, uim_u2i, user_item_matrix, uim_i2u, uim_i2a, uim_a2i, sm_a2i, similarity_matrix, sm_i2a
):
    liked_idxs = np.array(
        [sm_a2i[article_id] for article_id in get_liked_items(user_id, uim_u2i, user_item_matrix, uim_i2a)]
    )
    if len(liked_idxs) == 0:
        articles["collaborative_score"] = 0
        return articles
    similarity_scores = similarity_matrix[liked_idxs]
    top_similar_idxs = np.argsort(similarity_scores, axis=1)[:, -5:][:, ::-1]
    similar_article_ids = set(sm_i2a[top_similar_idxs].flatten())
    similar_article_idxs = [uim_a2i[article_id] for article_id in similar_article_ids if article_id in uim_a2i]
    similar_user_ids = [uim_i2u[idx] for idx in user_item_matrix[:, similar_article_idxs].nonzero()[0]]
    user_rank = Counter(similar_user_ids)
    user_ids = np.array(list(user_rank.keys()))
    user_idxs = [uim_u2i.get(user_id) for user_id in user_ids]
    scores = np.array(list(user_rank.values()))
    article_scores = defaultdict(float)
    for user_idx, score in zip(user_idxs, scores):
        liked_article_ids = uim_i2a[user_item_matrix[user_idx].nonzero()[1]]
        for article_id in liked_article_ids:
            article_scores[article_id] = max(article_scores.get(article_id, 0), score)
    articles["collaborative_score"] = articles["article_id"].map(article_scores).fillna(0)
    articles["collaborative_score"] = np.log1p(articles["collaborative_score"])
    scaler = MinMaxScaler()
    articles[["collaborative_score"]] = scaler.fit_transform(articles[["collaborative_score"]])
    return articles.sort_values(by="collaborative_score", ascending=False)


def hybrid_filtering(articles):
    weights = {"baseline_score": 0.5, "contentbased_score": 0.25, "collaborative_score": 0.25}
    articles["hybrid_score"] = (
        articles["baseline_score"] * weights["baseline_score"]
        + articles["contentbased_score"] * weights["contentbased_score"]
        + articles["collaborative_score"] * weights["collaborative_score"]
    )
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
    total_start = time.time()
    start = time.time()
    articles = baseline_filtering(articles=articles, date=date)
    print(f"\nBaseline:      {time.time() - start:.4f}")
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
    start = time.time()
    articles = hybrid_filtering(articles=articles)
    print(f"Hybrid:        {time.time() - start:.4f}")
    print(f"Total:         {time.time() - total_start:.4f} for user_id '{user_id}'")
    print()
    return articles
