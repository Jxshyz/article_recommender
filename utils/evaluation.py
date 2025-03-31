import random
import numpy as np
import pandas as pd
from utils.recommender import recommend


def temporarily_remove_liked_articles(
    user_id, liked_articles, user_item_matrix, uim_u2i, uim_a2i, removal_fraction=0.2
):
    original_matrix = user_item_matrix.copy()
    num_to_remove = max(1, int(len(liked_articles) * removal_fraction))
    removed_articles = random.sample(list(liked_articles), num_to_remove)
    user_idx = uim_u2i[user_id]
    for article in removed_articles:
        article_idx = uim_a2i[article]
        user_item_matrix[user_idx, article_idx] = 0
    return original_matrix, set(removed_articles)


def evaluate_recommendations(
    recommend,
    articles,
    user_item_matrix,
    uim_u2i,
    uim_i2u,
    uim_i2a,
    uim_a2i,
    sm_a2i,
    similarity_matrix,
    sm_i2a,
    k=10,
    n=20,
    removal_fraction=0.2,
):
    scores = ["baseline_score", "contentbased_score", "collaborative_score", "hybrid_score"]
    total_hits = {score: 0 for score in scores}
    total_precision = {score: 0 for score in scores}
    total_recall = {score: 0 for score in scores}
    total_ap = {score: 0 for score in scores}
    total_ndcg = {score: 0 for score in scores}
    users = random.sample(list(uim_u2i.keys()), n)

    for user_id in users:
        liked = set([uim_i2a[idx] for idx in user_item_matrix[uim_u2i[user_id]].nonzero()[1]])
        if not liked:
            continue
        original_matrix, removed = temporarily_remove_liked_articles(
            user_id, liked, user_item_matrix, uim_u2i, uim_a2i, removal_fraction
        )
        recommendations = recommend(
            articles, user_id, uim_u2i, user_item_matrix, uim_i2u, uim_i2a, uim_a2i, sm_a2i, similarity_matrix, sm_i2a
        )
        recommendations = recommendations[~recommendations["article_id"].isin(liked - removed)]

        for score in scores:
            recommended = recommendations.nlargest(k, score)["article_id"].tolist()
            hits = any(article in removed for article in recommended)
            precision = len(set(recommended) & removed) / k
            recall = len(set(recommended) & removed) / len(removed)
            ap = 0
            correct = 0
            for i, article in enumerate(recommended, start=1):
                if article in removed:
                    correct += 1
                    ap += correct / i
            ap /= min(len(removed), k)
            dcg = sum(1 / np.log2(i + 1) for i, article in enumerate(recommended, start=1) if article in removed)
            ideal_dcg = sum(1 / np.log2(i + 1) for i in range(1, min(len(removed), k) + 1))
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0

            total_hits[score] += hits
            total_precision[score] += precision
            total_recall[score] += recall
            total_ap[score] += ap
            total_ndcg[score] += ndcg

        user_item_matrix[:] = original_matrix

    num_users = len(users)
    results = {
        score: {
            "Hit Rate": float(total_hits[score] / num_users),
            "Precision@K": float(total_precision[score] / num_users),
            "Recall@K": float(total_recall[score] / num_users),
            "MAP@K": float(total_ap[score] / num_users),
            "NDCG@K": float(total_ndcg[score] / num_users),
        }
        for score in scores
    }

    data = {score.replace("_score", ""): [] for score in results.keys()}
    metrics = list(next(iter(results.values())).keys())
    for score in results:
        for metric in metrics:
            data[score.replace("_score", "")].append(results[score][metric])

    return pd.DataFrame(data, index=metrics)
