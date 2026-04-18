import pickle
from sklearn.metrics.pairwise import cosine_similarity

class SimpleRetriever:
    def __init__(self, vector_store_path):
        with open(vector_store_path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"]

    def retrieve(self, query, model, top_k=3):
        query_emb = model.encode([query])
        sims = cosine_similarity(query_emb, self.embeddings)[0]
        # sort indices by similarity (descending)
        sorted_indices = sims.argsort()[::-1][:top_k]
        # Return (chunk, score) tuples
        return [(self.chunks[i], sims[i]) for i in sorted_indices]
