import hashlib
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    import faiss as _faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


def normalize_text(text):
    return " ".join(text.split()).strip().lower()


def make_chunk_id(chunk):
    normalized = normalize_text(chunk)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


class SimpleRetriever:
    """Dense retriever that supports both legacy pickle stores and FAISS directories.

    Legacy path  (file ending in .pkl): loads embeddings + sklearn cosine.
    FAISS path   (directory with index.faiss + meta.pkl): uses FAISS IndexFlatIP.

    Auto-detected from the path passed to __init__.
    """

    def __init__(self, vector_store_path):
        if os.path.isdir(vector_store_path):
            self._load_faiss(vector_store_path)
        else:
            self._load_legacy(vector_store_path)

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_faiss(self, store_dir):
        if not _HAS_FAISS:
            raise ImportError("pip install faiss-cpu  to load FAISS vector stores")
        self._index = _faiss.read_index(os.path.join(store_dir, "index.faiss"))
        with open(os.path.join(store_dir, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        self.chunks = meta["chunks"]
        self.chunk_ids = [make_chunk_id(c) for c in self.chunks]
        self._use_faiss = True

    def _load_legacy(self, pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self._embeddings = data["embeddings"]
        self.chunk_ids = [make_chunk_id(c) for c in self.chunks]
        self._use_faiss = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query, model, top_k=3, boost_ids=None, boost_weight=0.15):
        """Return list of (chunk_text, score) — backward-compatible API."""
        results = self._retrieve_internal(query, model, top_k, boost_ids, boost_weight)
        return [(chunk, score) for chunk, score, _ in results]

    def retrieve_with_ids(self, query, model, top_k=10, boost_ids=None, boost_weight=0.15):
        """Return list of (chunk_text, score, chunk_id) — used by the eval harness."""
        return self._retrieve_internal(query, model, top_k, boost_ids, boost_weight)

    # ------------------------------------------------------------------
    # Internal retrieval
    # ------------------------------------------------------------------

    def _retrieve_internal(self, query, model, top_k, boost_ids, boost_weight):
        if self._use_faiss:
            return self._retrieve_faiss(query, model, top_k, boost_ids, boost_weight)
        return self._retrieve_legacy(query, model, top_k, boost_ids, boost_weight)

    def _retrieve_faiss(self, query, model, top_k, boost_ids, boost_weight):
        query_emb = model.encode([query]).astype("float32")
        norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        query_emb /= np.where(norm == 0, 1.0, norm)

        # Over-fetch when boosting so we can re-rank before trimming to top_k
        fetch_k = min(top_k * 3, len(self.chunks)) if boost_ids else top_k
        scores, indices = self._index.search(query_emb, fetch_k)
        scores = scores[0].copy()
        indices = indices[0]

        if boost_ids:
            boost_set = set(boost_ids)
            for j, idx in enumerate(indices):
                if self.chunk_ids[idx] in boost_set:
                    scores[j] += boost_weight
            order = scores.argsort()[::-1][:top_k]
            indices = indices[order]
            scores = scores[order]

        return [(self.chunks[i], float(scores[j]), self.chunk_ids[i])
                for j, i in enumerate(indices)]

    def _retrieve_legacy(self, query, model, top_k, boost_ids, boost_weight):
        query_emb = model.encode([query])
        sims = cosine_similarity(query_emb, self._embeddings)[0].copy()

        if boost_ids:
            boost_lookup = set(boost_ids)
            for idx, chunk_id in enumerate(self.chunk_ids):
                if chunk_id in boost_lookup:
                    sims[idx] += boost_weight

        top_indices = sims.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(sims[i]), self.chunk_ids[i])
                for i in top_indices]
