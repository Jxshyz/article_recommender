import os
import utils.preprocessing  # Module for data preprocessing
import utils.recommender  # Module for generating recommendations
import utils.evaluation  # Module for evaluating recommendations
import utils.track  # Module for resource usage tracking
from utils.ncf_model import train_ncf  # NCF training function


def main():
    """Main function to execute the recommendation system pipeline."""
    print("Starting recommendation system execution...")

    # Stage 1: Data Loading and Preprocessing
    with utils.track.resource_tracker("Data Loading"):
        print("Loading and preprocessing data...")
        articles = utils.preprocessing.get_preprocessed_articles()
        user_item_matrix, uim_u2i, uim_a2i, uim_i2u, uim_i2a = utils.preprocessing.get_preprocessed_user_item_matrix()
        print(f"Matrix non-zero entries: {user_item_matrix.nnz}")
        print(f"Sample values: {user_item_matrix.data[:10]}")
        similarity_matrix, sm_a2i, sm_i2a = utils.preprocessing.get_preprocessed_similarities(articles)
        print("Data loading completed.")

    # Stage 2: Train NCF (only if not already trained)
    with utils.track.resource_tracker("Train NCF"):
        train_ncf()

    # Safeguard: Check if NCF model and mappings exist
    if not os.path.exists("./models/ncf.pth") or not os.path.exists("./models/ncf_mappings.pth"):
        print("[!] NCF model or mapping not found. Skipping recommendation step.")
        return

    # Select first user for demonstration
    user_id = list(uim_u2i.keys())[0]

    # Stage 3: Recommendation Generation
    with utils.track.resource_tracker("Recommendation"):
        print(f"Generating recommendations for user {user_id}...")
        recommendations = utils.recommender.recommend(
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
        )
        print("Top 5 recommendations:")
        print(recommendations[["article_id", "title", "hybrid_score"]].head())
        print("Recommendation completed.")

    # Stage 4: Evaluation
    with utils.track.resource_tracker("Evaluation"):
        print("Evaluating recommendation system...")
        eval_results = utils.evaluation.evaluate_recommendations(
            utils.recommender.recommend,
            articles,
            user_item_matrix,
            uim_u2i,
            uim_i2u,
            uim_i2a,
            uim_a2i,
            sm_a2i,
            similarity_matrix,
            sm_i2a,
            k=500,
            n=100,
        )
        print("Evaluation Results:")
        print(eval_results)
        print("Evaluation completed.")

    utils.track.save_resources(os.path.join("resource_usage.csv"))
    print("Execution completed.")


if __name__ == "__main__":
    main()
