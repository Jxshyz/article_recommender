# Import necessary libraries
import numpy as np
from sentence_transformers import SentenceTransformer


def generate_embeddings(articles):
    """
    Generate sentence embeddings for articles using a pre-trained multilingual SentenceTransformer.

    Parameters:
        articles (pd.DataFrame): A pandas DataFrame containing the following columns:
            - 'title': The title of the article
            - 'subtitle': The subtitle of the article
            - 'category_str': The category or tag of the article
            - 'body': The main content of the article

    Returns:
        np.ndarray: A NumPy array of sentence embeddings for each article.
    """

    # Combine relevant text fields into a single string per article for embedding
    articles["aggregated_text"] = articles[["title", "subtitle", "category_str", "body"]].apply(
        lambda x: f"TITLE: {x['title']}\n\n\nSUBTITLE: {x['subtitle']}\n\n\nCATEGORY: {x['category_str']}\n\n\nCONTENT: {x['body']}",
        axis=1,
    )

    # Load a pre-trained multilingual sentence embedding model
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Define a helper function to generate embeddings in batches
    def batch_embed(texts, batch_size=1000):
        """
        Compute embeddings for a list of texts in batches.

        Parameters:
            texts (List[str]): List of text strings to embed.
            batch_size (int): Number of texts to process per batch.

        Returns:
            np.ndarray: Array of embeddings.
        """
        embeddings = []  # List to store the results

        # Loop through the texts in batches
        for i in range(0, len(texts), batch_size):
            print(f"Embedding in process - embedded {i}/{len(texts)}")
            embeddings.extend(model.encode(texts[i : i + batch_size]))  # Encode current batch

        return np.array(embeddings)

    # Generate embeddings for the aggregated text
    embeddings = list(batch_embed(articles["aggregated_text"].tolist()))

    # Remove the temporary aggregated text column from the DataFrame
    articles.drop(columns=["aggregated_text"], inplace=True)

    # Return the final array of embeddings
    return embeddings


def generate_embeddings_test(articles):
    """
    Generate sentence embeddings for articles using the BAAI/bge-m3 model.

    Parameters:
        articles (pd.DataFrame): A pandas DataFrame containing the following columns:
            - 'title': The title of the article
            - 'subtitle': The subtitle of the article
            - 'category_str': The category or tag of the article
            - 'body': The main content of the article

    Returns:
        np.ndarray: A NumPy array of sentence embeddings for each article (1024 dimensions).
    """
    # Combine relevant text fields into a single string per article for embedding
    articles["aggregated_text"] = articles[["title", "subtitle", "category_str", "body"]].apply(
        lambda x: f"TITLE: {x['title']}\nSUBTITLE: {x['subtitle']}\nCATEGORY: {x['category_str']}\nCONTENT: {x['body']}",
        axis=1,
    )

    # Load the BAAI/bge-m3 model
    model = SentenceTransformer("BAAI/bge-m3")

    # Generate embeddings with progress bar
    embeddings = model.encode(
        articles["aggregated_text"].tolist(),
        batch_size=32,  # Smaller batch size due to larger model
        show_progress_bar=True,  # Display progress for better monitoring
        convert_to_numpy=True,  # Ensure output is a NumPy array
    )

    # Remove the temporary aggregated text column from the DataFrame
    articles.drop(columns=["aggregated_text"], inplace=True)

    # Return the final array of embeddings
    return embeddings


# Example usage (uncomment to test)
# import pandas as pd
# articles = pd.DataFrame({
#     'title': ['Sample Title'],
#     'subtitle': ['Sample Subtitle'],
#     'category_str': ['News'],
#     'body': ['This is a sample article body.']
# })
# embeddings = generate_embeddings_bge_m3(articles)
# print(f"Embeddings shape: {embeddings.shape}")
