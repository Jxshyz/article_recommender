import os
import utils.preprocessing
import utils.recommender
import utils.evaluation
import utils.track


def main():
    print("Starting recommendation system execution...")

    with utils.track.resource_tracker("Data Loading"):
        print("Loading and preprocessing data...")
        articles = utils.preprocessing.get_preprocessed_articles()
        user_item_matrix, uim_u2i, uim_a2i, uim_i2u, uim_i2a = utils.preprocessing.get_preprocessed_user_item_matrix()
        similarity_matrix, sm_a2i, sm_i2a = utils.preprocessing.get_preprocessed_similarities(articles)
        print("Data loading completed.")

    user_id = list(uim_u2i.keys())[0]
    with utils.track.resource_tracker("Recommendation"):
        print(f"Generating recommendations for user {user_id}...")
        recommendations = utils.recommender.recommend(
            articles, user_id, uim_u2i, user_item_matrix, uim_i2u, uim_i2a, uim_a2i, sm_a2i, similarity_matrix, sm_i2a
        )
        print("Top 5 recommendations:")
        print(recommendations[["article_id", "title", "hybrid_score"]].head())
        print("Recommendation completed.")

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
            k=40,
            n=5,
        )
        print("Evaluation Results:")
        print(eval_results)
        print("Evaluation completed.")

    utils.track.save_resources(os.path.join("resource_usage.csv"))
    print("Execution completed.")


if __name__ == "__main__":
    main()
