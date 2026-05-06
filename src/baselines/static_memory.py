"""Static-memory baseline — exact reproduction of the conference paper variant.

Parameters match src/retrieval_memory.py exactly:
  threshold τ = 0.85   (cosine similarity gate)
  boost weight β = 0.15

When past queries exceed the similarity threshold, their successful context
IDs get a β=0.15 additive score boost before final ranking. This is the
handcrafted rule that ITMA replaces with a learned, counterfactually-updated
memory bank.
"""

from __future__ import annotations

import json
import os

from src.baselines import BaseRetriever
from src.agent import SimpleRetriever, make_chunk_id

FEEDBACK_FILE = "data/rl_feedback.json"
THRESHOLD = 0.85
BOOST = 0.15


class StaticMemoryRetriever(BaseRetriever):
    """Conference-paper τ+β memory boost on top of dense MiniLM retrieval."""

    def __init__(self, store_path: str, feedback_file: str = FEEDBACK_FILE,
                 threshold: float = THRESHOLD, boost: float = BOOST):
        from sentence_transformers import SentenceTransformer, util
        self._dense = SimpleRetriever(store_path)
        self._encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self._threshold = threshold
        self._boost = boost
        self._memory: list[dict] = []
        self._load_memory(feedback_file)

    # ------------------------------------------------------------------

    def retrieve(self, query: str, model, top_k: int = 3, boost_ids=None) -> list:
        return [(c, s) for c, s, _ in self._retrieve_impl(query, model, top_k)]

    def retrieve_with_ids(self, query: str, model, top_k: int = 10, boost_ids=None) -> list:
        return self._retrieve_impl(query, model, top_k)

    # ------------------------------------------------------------------

    def _retrieve_impl(self, query: str, model, top_k: int) -> list:
        boost_ids = self._get_boost_ids(query)
        candidates = self._dense.retrieve_with_ids(
            query, model, top_k=top_k, boost_ids=boost_ids
        )
        if not boost_ids:
            return candidates

        boosted = []
        bid_set = set(boost_ids)
        for chunk, score, cid in candidates:
            new_score = score + self._boost if cid in bid_set else score
            boosted.append((chunk, min(new_score, 1.0), cid))
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted[:top_k]

    def _get_boost_ids(self, query: str) -> list[str]:
        if not self._memory:
            return []
        from sentence_transformers import util
        q_emb = self._encoder.encode(query, convert_to_tensor=True)
        ids: list[str] = []
        for mem in self._memory:
            sim = util.pytorch_cos_sim(q_emb, mem["_embedding"]).item()
            if sim > self._threshold:
                ids.extend(mem["context_ids"])
        return list(set(ids))

    def _load_memory(self, feedback_file: str):
        if not os.path.exists(feedback_file):
            return
        entries = []
        with open(feedback_file) as f:
            for line in f:
                try:
                    e = json.loads(line.strip())
                    if e.get("reward") == 1:
                        entries.append(e)
                except Exception:
                    continue
        if not entries:
            return
        from sentence_transformers import SentenceTransformer
        queries = [e["query"] for e in entries]
        embs = self._encoder.encode(queries, convert_to_tensor=True)
        for e, emb in zip(entries, embs):
            e["_embedding"] = emb
        self._memory = entries
