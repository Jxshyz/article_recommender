import numpy as np
#from sklearn.metrics import recall_score

def evaluate_recommendation(predictions, ground_truth):
    """
    Evaluate recall for a recommendation system.
    
    Parameters:
    - predictions: List of sets of recommended items (e.g., from different filterers)
    - ground_truth: List of sets of actual clicked items
    
    Returns:
    - recall_scores: List of recall scores for each recommender
    """
    recall_scores = []
    for pred in predictions:
        true_positives = sum([1 for user in range(len(ground_truth)) if len(pred[user] & ground_truth[user]) > 0])
        false_negatives = sum([1 for user in range(len(ground_truth)) if len(pred[user] & ground_truth[user]) == 0])
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        recall_scores.append(recall)
    
    return recall_scores


def split_data_sliding_window(interactions, window_size, step_size):
    """
    Split the data using a sliding window approach.
    
    Parameters:
    - interactions: List of tuples (user_id, item_id, timestamp)
    - window_size: Size of sliding window in time units (e.g., days)
    - step_size: Step size for the sliding window
    
    Returns:
    - train_test_splits: List of (train_set, test_set) pairs
    """
    from datetime import timedelta
    
    interactions.sort(key=lambda x: x[2])  # Sort by timestamp
    start_time = interactions[0][2]
    end_time = interactions[-1][2]
    
    splits = []
    current_time = start_time
    
    while current_time + window_size <= end_time:
        train_window = [(u, i) for u, i, t in interactions if current_time <= t < current_time + window_size]
        test_window = [(u, i) for u, i, t in interactions if current_time + window_size <= t < current_time + 2 * window_size]
        
        train_set = {u: set() for u, i in train_window}
        test_set = {u: set() for u, i in test_window}
        
        for u, i in train_window:
            train_set[u].add(i)
        for u, i in test_window:
            test_set[u].add(i)
        
        splits.append((train_set, test_set))
        current_time += step_size
    
    return splits


# Example Input
# Predictions from the three recommenders
most_popular_preds = [{1, 3, 5}, {2, 4, 6}, {1, 2, 3}]
content_based_preds = [{1, 2}, {2, 5, 6}, {4, 5}]
collaborative_preds = [{3, 4, 5}, {1, 2, 3}, {5, 6}]

# Ground truth from user interactions
ground_truth = [{1, 5}, {2, 6}, {4, 5}]

# Evaluate Recall
recalls = evaluate_recommendation(
    [most_popular_preds, content_based_preds, collaborative_preds],
    ground_truth
)

print("Recall Scores:")
print(f"Most Popular: {recalls[0]:.4f}")
print(f"Content-Based: {recalls[1]:.4f}")
print(f"Collaborative Filtering: {recalls[2]:.4f}")
