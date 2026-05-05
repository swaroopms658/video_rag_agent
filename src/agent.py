import hashlib
import pickle
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(text):
    return " ".join(text.split()).strip().lower()


def make_chunk_id(chunk):
    normalized = normalize_text(chunk)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


class SimpleRetriever:
    def __init__(self, vector_store_path):
        with open(vector_store_path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"]
        self.chunk_ids = [make_chunk_id(chunk) for chunk in self.chunks]

    def retrieve(self, query, model, top_k=3, boost_ids=None, boost_weight=0.15):
        query_emb = model.encode([query])
        sims = cosine_similarity(query_emb, self.embeddings)[0]

        if boost_ids:
            boost_lookup = set(boost_ids)
            for index, chunk_id in enumerate(self.chunk_ids):
                if chunk_id in boost_lookup:
                    sims[index] += boost_weight

        # sort indices by similarity (descending)
        sorted_indices = sims.argsort()[::-1][:top_k]
        # Return (chunk, score) tuples
        return [(self.chunks[i], sims[i]) for i in sorted_indices]
