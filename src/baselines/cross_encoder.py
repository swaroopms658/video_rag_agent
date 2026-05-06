"""Dense + Cross-Encoder reranking baseline.

Stage 1: Dense-MiniLM retrieves top-20 candidates.
Stage 2: ms-marco-MiniLM-L-6-v2 cross-encoder scores each (query, chunk)
         pair and re-ranks to top-k.

The cross-encoder score is a logit (not bounded to [0,1]), so we apply
sigmoid normalisation for comparability.
"""

from __future__ import annotations

import math
import pickle

from src.baselines import BaseRetriever
from src.agent import SimpleRetriever, make_chunk_id

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FIRST_STAGE_K = 20


class CrossEncoderRetriever(BaseRetriever):
    """Two-stage retriever: MiniLM dense recall → cross-encoder reranking."""

    def __init__(self, store_path: str):
        from sentence_transformers import CrossEncoder
        self._dense = SimpleRetriever(store_path)
        self._ce = CrossEncoder(CROSS_ENCODER_MODEL)

    # ------------------------------------------------------------------

    def retrieve(self, query: str, model, top_k: int = 3, boost_ids=None) -> list:
        return [(c, s) for c, s, _ in self._rerank(query, model, top_k)]

    def retrieve_with_ids(self, query: str, model, top_k: int = 10, boost_ids=None) -> list:
        return self._rerank(query, model, top_k)

    # ------------------------------------------------------------------

    def _rerank(self, query: str, model, top_k: int) -> list:
        candidates = self._dense.retrieve_with_ids(
            query, model, top_k=FIRST_STAGE_K
        )
        if not candidates:
            return []

        pairs = [(query, c[0]) for c in candidates]
        logits = self._ce.predict(pairs)

        # Sigmoid normalisation: σ(logit) → [0, 1]
        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        scored = sorted(
            zip(candidates, logits),
            key=lambda x: x[1],
            reverse=True,
        )
        results = []
        for (chunk, _dense_score, cid), logit in scored[:top_k]:
            results.append((chunk, _sigmoid(float(logit)), cid))
        return results
