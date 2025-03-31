import numpy as np
from sentence_transformers import SentenceTransformer


def generate_embeddings(articles):
    articles["aggregated_text"] = articles[["title", "subtitle", "category_str", "body"]].apply(
        lambda x: f"TITLE: {x['title']}\n\n\nSUBTITLE: {x['subtitle']}\n\n\nCATEGORY: {x['category_str']}\n\n\nCONTENT: {x['body']}",
        axis=1,
    )
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def batch_embed(texts, batch_size=1000):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            print(f"Embedding in process - embedded {i}/{len(texts)}")
            embeddings.extend(model.encode(texts[i : i + batch_size]))
        return np.array(embeddings)

    embeddings = list(batch_embed(articles["aggregated_text"].tolist()))
    articles.drop(columns=["aggregated_text"], inplace=True)
    return embeddings
