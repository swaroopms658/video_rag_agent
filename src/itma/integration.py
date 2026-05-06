"""ITMA integration — retriever that uses the frozen scoring head + memory bank.

Retrieval pipeline:
  1. Dense-MiniLM FAISS over-fetch top-N candidates (N=20).
  2. Compute memory summary m = MemoryBank.attend(q_emb).
  3. Score each candidate: final_score = FrozenScoringHead.score(q, c, m).
  4. Re-rank by final_score, return top-k.
  5. After answer generation and feedback:
       - MemoryBank.add(q_emb, best_context_emb, chunk_id)
       - MemoryBank.update_counterfactual(...)

The retriever can run without a checkpoint (scoring head uses random weights but
gated scores still give valid rankings — just not optimised). This allows
evaluation before pretraining is complete, which can serve as a "no-pretrain"
ablation.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.baselines import BaseRetriever
from src.agent import SimpleRetriever, make_chunk_id
from src.itma.memory_bank import MemoryBank
from src.itma.scoring_head import FrozenScoringHead

FIRST_STAGE_K = 20
DEFAULT_MEMORY_PATH = "data/itma_memory.pkl"
DEFAULT_CHECKPOINT = "checkpoints/itma_head.pt"


class ITMARetriever(BaseRetriever):
    """Inference-Time Memory Adaptation retriever.

    - Frozen scoring head (pretrained once, never retrained at deploy).
    - Online memory bank (no gradients, updated after each query).

    Ablation flags:
      use_scoring_head=False  → skip head, use only cosine sim (+ optional ID-boost)
      use_id_boost=False      → skip ID-boost, rely only on scoring head
    """

    def __init__(
        self,
        store_path: str,
        checkpoint: Optional[str] = None,
        memory_path: Optional[str] = DEFAULT_MEMORY_PATH,
        memory_capacity: int = 128,
        lam: float = 0.05,
        eta: float = 0.05,
        device: str = "cpu",
        use_scoring_head: bool = True,
        use_id_boost: bool = True,
    ):
        self._use_scoring_head = use_scoring_head
        self._use_id_boost = use_id_boost
        self._dense = SimpleRetriever(store_path)
        ckpt_path = checkpoint if checkpoint else DEFAULT_CHECKPOINT
        ckpt = ckpt_path if ckpt_path and os.path.exists(ckpt_path) else None
        self._head = FrozenScoringHead(checkpoint_path=ckpt, device=device) if use_scoring_head else None
        self._memory = MemoryBank(
            capacity=memory_capacity,
            lam=lam,
            eta=eta,
            persist_path=memory_path,
        )
        self._encoder: Optional[SentenceTransformer] = None

        # Cache of chunk embeddings keyed by chunk_id for memory updates
        self._chunk_emb_cache: dict[str, np.ndarray] = {}

        # State for counterfactual update (set during retrieve, used after feedback)
        self._last_q_emb: Optional[np.ndarray] = None
        self._last_attended_ids: list[str] = []
        self._last_attended_alphas: list[float] = []

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str, model, top_k: int = 3, boost_ids=None) -> list:
        return [(c, s) for c, s, _ in self._rank(query, model, top_k)]

    def retrieve_with_ids(self, query: str, model, top_k: int = 10, boost_ids=None) -> list:
        return self._rank(query, model, top_k)

    # ------------------------------------------------------------------
    # Feedback loop: call after you know whether the answer was good
    # ------------------------------------------------------------------

    def record_feedback(
        self,
        helpful_chunk_ids: list[str],
        reward: float,
    ):
        """Update memory bank after answer quality is known.

        helpful_chunk_ids: chunk IDs of contexts that were actually useful.
        reward: +1.0 = correct/helpful, -1.0 = wrong/unhelpful.
        """
        if self._last_q_emb is None:
            return

        # Add helpful contexts to memory bank
        if reward > 0:
            for cid in helpful_chunk_ids:
                c_emb = self._chunk_emb_cache.get(cid)
                if c_emb is not None:
                    self._memory.add(self._last_q_emb, c_emb, cid)

        # Counterfactual update on attended entries
        if self._last_attended_ids:
            self._memory.update_counterfactual(
                self._last_attended_ids,
                self._last_attended_alphas,
                reward,
            )

        self._memory.increment_query_count()

    def save_memory(self):
        self._memory.save()

    def reset_memory(self):
        self._memory.reset()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rank(self, query: str, model, top_k: int) -> list:
        # Get encoder (lazily inherit from model arg or fall back to MiniLM)
        encoder = model if model is not None else self._get_encoder()

        # Encode query
        q_emb = encoder.encode(query, normalize_embeddings=True).astype(np.float32)
        self._last_q_emb = q_emb

        # Dense first-stage retrieval
        candidates = self._dense.retrieve_with_ids(query, encoder, top_k=FIRST_STAGE_K)
        if not candidates:
            return []

        # Load candidate embeddings (from FAISS meta or recompute)
        c_texts = [c[0] for c in candidates]
        c_ids = [c[2] for c in candidates]
        c_embs = self._get_chunk_embeddings(c_texts, c_ids, encoder)

        # Memory summary (same for all candidates in this query)
        m = self._memory.attend(q_emb)     # (D,)
        m_batch = np.tile(m[None], (len(c_texts), 1))   # (N, D)

        # Compute attention weights (for counterfactual tracking)
        if self._memory.size() > 0:
            weights = self._memory.effective_weights()
            q_normed = q_emb / (np.linalg.norm(q_emb) + 1e-8)
            q_embs_mem = self._memory.query_embeddings()
            norms = np.linalg.norm(q_embs_mem, axis=1, keepdims=True) + 1e-8
            cos_sims = (q_embs_mem / norms) @ q_normed
            logits = weights * cos_sims / 0.1
            logits -= logits.max()
            alpha = np.exp(logits)
            alpha /= (alpha.sum() + 1e-8)

            # Map entry chunk_ids to their alpha weights
            mem_entries = list(self._memory._entries)
            self._last_attended_ids = [e["chunk_id"] for e in mem_entries]
            self._last_attended_alphas = alpha.tolist()
        else:
            self._last_attended_ids = []
            self._last_attended_alphas = []

        # Score all candidates
        if self._use_scoring_head:
            scores = self._head.score(q_emb, c_embs, m_batch)   # (N,)
        else:
            # Ablation: no scoring head — use cosine similarity directly
            from src.itma.scoring_head import BETA_DENSE
            q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
            c_norms = c_embs / (np.linalg.norm(c_embs, axis=1, keepdims=True) + 1e-8)
            cos_sim = (c_norms @ q_norm).astype(np.float32)
            scores = (cos_sim + 1.0) / 2.0   # scale to [0,1]

        # Memory ID-boost: if a candidate chunk was explicitly marked helpful for a
        # past query that is similar to the current query, boost its score.
        # This bypasses the under-trained gate and directly rewards known-good chunks.
        if self._use_id_boost and self._memory.size() > 0:
            q_norm_vec = q_emb / (np.linalg.norm(q_emb) + 1e-8)
            c_ids_set = set(c_ids)
            bonus = np.zeros(len(c_ids), dtype=np.float32)
            for entry in self._memory._entries:
                cid = entry["chunk_id"]
                if cid not in c_ids_set:
                    continue
                q_mem = entry["query_emb"]
                q_mem_norm = q_mem / (np.linalg.norm(q_mem) + 1e-8)
                q_sim = max(0.0, float(q_norm_vec @ q_mem_norm))
                eff_w = (np.exp(-self._memory.lam * entry["age"])
                         * entry["w_cf"] * q_sim)
                bonus[c_ids.index(cid)] += eff_w
            if bonus.max() > 1e-6:
                bonus /= bonus.max() + 1e-8
                scores = scores + 0.4 * bonus

        # Sort by score descending
        order = np.argsort(scores)[::-1][:top_k]
        results = [(c_texts[i], float(scores[i]), c_ids[i]) for i in order]
        return results

    def _get_chunk_embeddings(
        self,
        texts: list[str],
        ids: list[str],
        encoder,
    ) -> np.ndarray:
        """Return stacked embeddings (N, D); cache for reuse in feedback loop."""
        result = []
        to_encode_idx, to_encode_texts = [], []
        for i, (text, cid) in enumerate(zip(texts, ids)):
            if cid in self._chunk_emb_cache:
                result.append((i, self._chunk_emb_cache[cid]))
            else:
                to_encode_idx.append(i)
                to_encode_texts.append(text)

        if to_encode_texts:
            new_embs = encoder.encode(
                to_encode_texts, normalize_embeddings=True
            ).astype(np.float32)
            for idx, emb in zip(to_encode_idx, new_embs):
                cid = ids[idx]
                self._chunk_emb_cache[cid] = emb
                result.append((idx, emb))

        result.sort(key=lambda x: x[0])
        return np.stack([emb for _, emb in result])

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        return self._encoder
