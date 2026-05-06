"""BM25 retrieval baseline using rank_bm25.

Loads chunk text from a FAISS vector store directory (meta.pkl) or a
legacy sklearn pkl store. Builds a BM25Okapi index over the chunks at
construction time. Retrieval is query-time tokenization → BM25 ranking.
No embedding model is used; the `model` argument is accepted but ignored.
"""

from __future__ import annotations

import pickle
import re

import numpy as np

from src.baselines import BaseRetriever
from src.agent import make_chunk_id


def _simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever(BaseRetriever):
    """BM25Okapi retriever over a pre-built chunk list."""

    def __init__(self, store_path: str):
        from rank_bm25 import BM25Okapi
        import os

        chunks, timestamps = _load_chunks(store_path)
        self._chunks = chunks
        self._timestamps = timestamps
        self._chunk_ids = [make_chunk_id(c) for c in chunks]

        tokenized = [_simple_tokenize(c) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    # ------------------------------------------------------------------

    def retrieve(self, query: str, model, top_k: int = 3, boost_ids=None) -> list:
        results = self._rank(query, top_k)
        return [(c, s) for c, s, _ in results]

    def retrieve_with_ids(self, query: str, model, top_k: int = 10, boost_ids=None) -> list:
        return self._rank(query, top_k)

    # ------------------------------------------------------------------

    def _rank(self, query: str, top_k: int) -> list:
        tokens = _simple_tokenize(query)
        scores = self._bm25.get_scores(tokens)
        # Normalise to [0, 1] so scores are comparable across systems
        max_s = float(scores.max()) if scores.max() > 0 else 1.0
        scores = scores / max_s
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self._chunks[i], float(scores[i]), self._chunk_ids[i])
            for i in top_indices
        ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_chunks(store_path: str):
    import os
    meta_path = os.path.join(store_path, "meta.pkl")
    if os.path.isdir(store_path) and os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        chunks = meta["chunks"]
        timestamps = meta.get("timestamps", [None] * len(chunks))
        return chunks, timestamps

    # Legacy sklearn .pkl format
    with open(store_path, "rb") as f:
        store = pickle.load(f)
    chunks = store.get("chunks", [])
    timestamps = store.get("timestamps", [None] * len(chunks))
    return chunks, timestamps
