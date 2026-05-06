"""ITMA Memory Bank — bounded FIFO with freshness decay and counterfactual reweighting.

The memory bank stores past retrieval interactions. No gradient updates happen here;
all adaptation is through the memory bank state (online, at deployment time).

Entry structure:
  {
    "query_emb":   np.ndarray(384,)    # encoded query
    "context_emb": np.ndarray(384,)    # encoded context that was helpful
    "chunk_id":    str                 # for deduplication
    "age":         int                 # number of queries since this entry was added
    "w_cf":        float               # counterfactual weight ∈ [0.1, 5.0]
  }

Effective weight at query time: w(j) = exp(-λ·age_j) · w_cf_j
"""

from __future__ import annotations

import json
import os
import pickle
from collections import deque
from typing import Optional

import numpy as np

DEFAULT_CAPACITY = 128
DEFAULT_LAMBDA = 0.05    # freshness decay rate
DEFAULT_ETA = 0.05       # counterfactual learning rate
W_CF_MIN = 0.1
W_CF_MAX = 5.0


class MemoryBank:
    """Online memory bank. No PyTorch, no gradients — pure numpy + Python state."""

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        lam: float = DEFAULT_LAMBDA,
        eta: float = DEFAULT_ETA,
        persist_path: Optional[str] = None,
    ):
        self.capacity = capacity
        self.lam = lam
        self.eta = eta
        self.persist_path = persist_path
        self._entries: deque[dict] = deque()
        self._query_count = 0  # total queries seen (used for age tracking)

        if persist_path and os.path.exists(persist_path):
            self._load(persist_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def size(self) -> int:
        return len(self._entries)

    def add(self, query_emb: np.ndarray, context_emb: np.ndarray, chunk_id: str):
        """Add a new memory entry. Evicts oldest when at capacity."""
        # Skip duplicate chunk IDs to avoid redundancy
        if any(e["chunk_id"] == chunk_id for e in self._entries):
            return

        entry = {
            "query_emb":   query_emb.astype(np.float32),
            "context_emb": context_emb.astype(np.float32),
            "chunk_id":    chunk_id,
            "age":         0,
            "w_cf":        1.0,
        }
        if len(self._entries) >= self.capacity:
            self._entries.popleft()  # FIFO eviction
        self._entries.append(entry)

        # Increment age of all existing entries
        for e in self._entries:
            e["age"] += 1

    def effective_weights(self) -> np.ndarray:
        """Compute effective weight w(j) = exp(-λ·age_j) · w_cf_j for all entries."""
        if not self._entries:
            return np.array([], dtype=np.float32)
        ws = np.array(
            [np.exp(-self.lam * e["age"]) * e["w_cf"] for e in self._entries],
            dtype=np.float32,
        )
        return ws

    def query_embeddings(self) -> np.ndarray:
        """Stack query embeddings, shape (N, D)."""
        if not self._entries:
            return np.zeros((0, 384), dtype=np.float32)
        return np.stack([e["query_emb"] for e in self._entries])

    def context_embeddings(self) -> np.ndarray:
        """Stack context embeddings, shape (N, D)."""
        if not self._entries:
            return np.zeros((0, 384), dtype=np.float32)
        return np.stack([e["context_emb"] for e in self._entries])

    def attend(self, query_emb: np.ndarray, temperature: float = 0.1) -> np.ndarray:
        """Compute attention-weighted memory summary vector.

        α_j = softmax( w_j · cos(q, q_j) / temperature )
        m   = Σ_j  α_j · context_emb_j
        Returns:
          m (D,) — the memory context vector, zeros if memory is empty.
        """
        if not self._entries:
            return np.zeros(query_emb.shape, dtype=np.float32)

        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        q_embs = self.query_embeddings()
        norms = np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-8
        q_embs_normed = q_embs / norms

        cos_sims = q_embs_normed @ q                     # (N,)
        weights = self.effective_weights()               # (N,)
        logits = weights * cos_sims / temperature        # (N,)

        # Stable softmax
        logits -= logits.max()
        alpha = np.exp(logits)
        alpha /= (alpha.sum() + 1e-8)

        c_embs = self.context_embeddings()               # (N, D)
        m = alpha @ c_embs                               # (D,)
        return m.astype(np.float32)

    def update_counterfactual(
        self,
        attended_chunk_ids: list[str],
        attended_alphas: list[float],
        reward: float,
    ):
        """Update w_cf for entries that were attended to in the last query.

        reward = +1 if the answer was correct/helpful, -1 if not.
        Entries with chunk_id in attended_chunk_ids get w_cf adjusted by
        ±η · α_j, clipped to [W_CF_MIN, W_CF_MAX].
        """
        id_to_alpha = dict(zip(attended_chunk_ids, attended_alphas))
        for entry in self._entries:
            alpha = id_to_alpha.get(entry["chunk_id"])
            if alpha is None:
                continue
            delta = self.eta * alpha * reward
            entry["w_cf"] = float(
                np.clip(entry["w_cf"] + delta, W_CF_MIN, W_CF_MAX)
            )

    def increment_query_count(self):
        self._query_count += 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None):
        target = path or self.persist_path
        if not target:
            return
        os.makedirs(os.path.dirname(target) if os.path.dirname(target) else ".", exist_ok=True)
        state = {
            "capacity": self.capacity,
            "lam": self.lam,
            "eta": self.eta,
            "query_count": self._query_count,
            "entries": list(self._entries),
        }
        with open(target, "wb") as f:
            pickle.dump(state, f)

    def _load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.capacity = state.get("capacity", self.capacity)
        self.lam = state.get("lam", self.lam)
        self.eta = state.get("eta", self.eta)
        self._query_count = state.get("query_count", 0)
        self._entries = deque(state.get("entries", []))

    def reset(self):
        self._entries.clear()
        self._query_count = 0
        if self.persist_path and os.path.exists(self.persist_path):
            os.remove(self.persist_path)
