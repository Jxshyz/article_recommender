# Import necessary libraries
import random
import numpy as np
import pandas as pd
from utils.recommender import recommend  # Custom recommendation function


def temporarily_remove_liked_articles(
    user_id, liked_articles, user_item_matrix, uim_u2i, uim_a2i, removal_fraction=0.2
):
    """
    Temporarily removes a fraction of liked articles from the user-item interaction matrix
    for evaluation purposes.

    Parameters:
        user_id (str or int): ID of the user.
        liked_articles (set): Set of article IDs that the user liked.
        user_item_matrix (np.ndarray): Matrix of user-item interactions.
        uim_u2i (dict): Mapping from user ID to matrix index.
        uim_a2i (dict): Mapping from article ID to matrix index.
        removal_fraction (float): Fraction of liked articles to remove (default: 0.2).

    Returns:
        tuple: A tuple containing:
            - A copy of the original user-item matrix before modification.
            - A set of removed article IDs.
    """
    original_matrix = user_item_matrix.copy()  # Backup original state
    num_to_remove = max(1, int(len(liked_articles) * removal_fraction))  # Ensure at least one is removed
    removed_articles = random.sample(list(liked_articles), num_to_remove)  # Randomly select articles to remove
    user_idx = uim_u2i[user_id]  # Get user's index in the matrix

    # Zero out the selected articles for the user
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
    """
    Evaluate the performance of recommendation strategies using offline metrics.

    Parameters:
        recommend (function): The recommendation function to evaluate.
        articles (pd.DataFrame): DataFrame containing article metadata.
        user_item_matrix (np.ndarray): User-item interaction matrix.
        uim_u2i (dict): User ID to matrix index mapping.
        uim_i2u (dict): Matrix index to user ID mapping.
        uim_i2a (dict): Matrix index to article ID mapping.
        uim_a2i (dict): Article ID to matrix index mapping.
        sm_a2i (dict): Similarity matrix article ID to index mapping.
        similarity_matrix (np.ndarray): Precomputed article similarity matrix.
        sm_i2a (dict): Index to article ID mapping for similarity matrix.
        k (int): Number of top recommendations to evaluate (default: 10).
        n (int): Number of users to evaluate (default: 20).
        removal_fraction (float): Fraction of liked articles to hide for evaluation (default: 0.2).

    Returns:
        pd.DataFrame: A DataFrame summarizing evaluation metrics for each scoring method.
    """
    # Scoring strategies used in the recommendation output
    scores = ["baseline_score", "contentbased_score", "collaborative_score", "hybrid_score"]

    # Initialize accumulators for evaluation metrics
    total_hits = {score: 0 for score in scores}
    total_precision = {score: 0 for score in scores}
    total_recall = {score: 0 for score in scores}
    total_ap = {score: 0 for score in scores}
    total_ndcg = {score: 0 for score in scores}

    # Randomly sample n users for evaluation
    users = random.sample(list(uim_u2i.keys()), n)

    for user_id in users:
        # Get all articles liked by the user
        liked = set([uim_i2a[idx] for idx in user_item_matrix[uim_u2i[user_id]].nonzero()[1]])
        if not liked:
            continue  # Skip users with no likes

        # Temporarily remove a fraction of liked articles
        original_matrix, removed = temporarily_remove_liked_articles(
            user_id, liked, user_item_matrix, uim_u2i, uim_a2i, removal_fraction
        )

        # Get recommendations from the system
        recommendations = recommend(
            articles, user_id, uim_u2i, user_item_matrix, uim_i2u, uim_i2a, uim_a2i, sm_a2i, similarity_matrix, sm_i2a
        )

        # Filter out articles already liked but not removed (keep only unseen or removed ones)
        recommendations = recommendations[~recommendations["article_id"].isin(liked - removed)]

        for score in scores:
            # Get top-k recommendations by current scoring method
            recommended = recommendations.nlargest(k, score)["article_id"].tolist()

            # Compute evaluation metrics
            hits = any(article in removed for article in recommended)
            precision = len(set(recommended) & removed) / k
            recall = len(set(recommended) & removed) / len(removed)

            # Mean Average Precision (MAP@K)
            ap = 0
            correct = 0
            for i, article in enumerate(recommended, start=1):
                if article in removed:
                    correct += 1
                    ap += correct / i
            ap /= min(len(removed), k)

            # Normalized Discounted Cumulative Gain (NDCG@K)
            dcg = sum(1 / np.log2(i + 1) for i, article in enumerate(recommended, start=1) if article in removed)
            ideal_dcg = sum(1 / np.log2(i + 1) for i in range(1, min(len(removed), k) + 1))
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0

            # Accumulate metrics
            total_hits[score] += hits
            total_precision[score] += precision
            total_recall[score] += recall
            total_ap[score] += ap
            total_ndcg[score] += ndcg

        # Restore the original matrix after evaluation
        user_item_matrix[:] = original_matrix

    # Average the results over all evaluated users
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

    # Format the results as a DataFrame
    data = {score.replace("_score", ""): [] for score in results.keys()}
    metrics = list(next(iter(results.values())).keys())
    for score in results:
        for metric in metrics:
            data[score.replace("_score", "")].append(results[score][metric])

    return pd.DataFrame(data, index=metrics)
