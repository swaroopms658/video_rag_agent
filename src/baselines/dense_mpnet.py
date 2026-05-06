"""Dense retrieval baseline using all-mpnet-base-v2.

Loads the FAISS index (or legacy pkl), encodes the query with mpnet, and
retrieves via cosine similarity. The embed_model argument passed to
retrieve() is ignored — this retriever maintains its own mpnet instance.
"""

from __future__ import annotations

import os
import pickle

import numpy as np

from src.baselines import BaseRetriever
from src.agent import make_chunk_id

MPNET_MODEL = "all-mpnet-base-v2"


class DenseMpnetRetriever(BaseRetriever):
    """Dense FAISS retriever backed by all-mpnet-base-v2 (768-d)."""

    def __init__(self, store_path: str):
        from sentence_transformers import SentenceTransformer
        import faiss

        self._model = SentenceTransformer(MPNET_MODEL, device="cpu")
        self._store_path = store_path

        meta_path = os.path.join(store_path, "meta.pkl")
        if os.path.isdir(store_path) and os.path.exists(meta_path):
            self._load_faiss(store_path, meta_path)
        else:
            self._load_legacy(store_path)

    # ------------------------------------------------------------------

    def retrieve(self, query: str, model, top_k: int = 3, boost_ids=None) -> list:
        return [(c, s) for c, s, _ in self._search(query, top_k)]

    def retrieve_with_ids(self, query: str, model, top_k: int = 10, boost_ids=None) -> list:
        return self._search(query, top_k)

    # ------------------------------------------------------------------

    def _search(self, query: str, top_k: int) -> list:
        q_emb = self._model.encode([query], normalize_embeddings=True).astype("float32")
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q_emb, min(top_k, len(self._chunks)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                results.append((self._chunks[idx], float(score), self._chunk_ids[idx]))
            return results

        # Legacy: sklearn cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(q_emb, self._legacy_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self._chunks[i], float(sims[i]), self._chunk_ids[i]) for i in top_idx]

    def _load_faiss(self, store_path: str, meta_path: str):
        import faiss
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self._chunks = meta["chunks"]
        self._chunk_ids = [make_chunk_id(c) for c in self._chunks]

        # Rebuild mpnet FAISS index since stored embeddings used MiniLM
        embeddings = self._model.encode(
            self._chunks, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        dim = embeddings.shape[1]
        index = __import__("faiss").IndexFlatIP(dim)
        index.add(embeddings)
        self._faiss_index = index
        self._legacy_embeddings = None

    def _load_legacy(self, store_path: str):
        with open(store_path, "rb") as f:
            store = pickle.load(f)
        self._chunks = store.get("chunks", [])
        self._chunk_ids = [make_chunk_id(c) for c in self._chunks]
        # Re-encode with mpnet
        embeddings = self._model.encode(
            self._chunks, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        self._legacy_embeddings = embeddings
        self._faiss_index = None
