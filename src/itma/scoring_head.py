"""ITMA Scoring Head — pretrained once, frozen forever at deployment.

Architecture:
  Input features: [q(D), c_i(D), m(D), q⊙c_i(D), q_proj(P)⊙m_proj(P)]
    where D=384 (MiniLM embedding dim), P=128 (projection dim)
  Total input dim = D+D+D+D + P+P = 4·384 + 2·128 = 1792

  Layers:
    proj_q:  Linear(D, P)        # query projection for memory interaction
    proj_m:  Linear(D, P)        # memory vector projection
    mlp:     Linear(1792, 256) → GELU → Linear(256, 128) → GELU → Linear(128, 1)
    gate:    Linear(D + P, 1)    # cold-start gate — σ(gate) → 0 when memory empty

  Output:
    final_score = β_dense · cos(q, c_i) + σ(gate) · s(q, c_i, M)
    where s(q, c_i, M) = sigmoid(mlp([q, c_i, m, q⊙c_i, q_proj⊙m_proj]))

Cold-start guarantee:
  When M is empty, m = 0 ⇒ gate logit ≈ 0 ⇒ σ(gate) ≈ 0.5 at random init,
  but we regularise the gate bias to -3 so σ(−3) ≈ 0.047 at init.
  The pretrain loss also encourages gate → 0 when M is empty (via gate regularisation).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 384   # MiniLM embedding dimension
P = 128   # projection dimension
# v5: added c⊙m so the MLP has a direct candidate–memory alignment feature.
# Without it the gate can open all it wants but the gated score has no
# dependence on whether the candidate matches what's in memory.
MLP_INPUT = 5 * D + P   # 2048: [q, c, m, q⊙c, c⊙m] (each D) + [q_proj⊙m_proj] (P)

BETA_DENSE = 0.7   # weight on cos-sim component
BETA_MEM = 0.3     # weight on learned-head component


class ITMAHead(nn.Module):
    """Pretrained scoring head. Load from checkpoint and call .eval() at deployment."""

    def __init__(self, emb_dim: int = D, proj_dim: int = P, legacy: bool = False):
        """
        legacy=False (default, v5+): MLP input includes c*m feature → 5D+P=2048.
        legacy=True  (v3/v4 ckpts):  MLP input excludes c*m         → 4D+P=1664.
        FrozenScoringHead auto-detects from checkpoint shape.
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.legacy = legacy
        mlp_in = (4 if legacy else 5) * emb_dim + proj_dim

        self.proj_q = nn.Linear(emb_dim, proj_dim, bias=False)
        self.proj_m = nn.Linear(emb_dim, proj_dim, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Gate: takes [q(D), m(D)] → scalar logit
        # Bias initialised to -3 so gate is near-closed at init (cold-start safety)
        self.gate = nn.Linear(emb_dim + emb_dim, 1)
        nn.init.constant_(self.gate.bias, -3.0)

    def forward(
        self,
        q: torch.Tensor,        # (B, D) query embeddings
        c: torch.Tensor,        # (B, D) candidate context embeddings
        m: torch.Tensor,        # (B, D) memory summary vectors (zeros if M empty)
        cos_sim: torch.Tensor,  # (B,) precomputed cos(q, c)
    ) -> torch.Tensor:
        """Returns final_score (B,), values roughly in [0, 1]."""
        q_proj = self.proj_q(q)          # (B, P)
        m_proj = self.proj_m(m)          # (B, P)

        parts = [q, c, m, q * c]
        if not self.legacy:
            parts.append(c * m)          # v5: candidate-memory alignment
        parts.append(q_proj * m_proj)
        feats = torch.cat(parts, dim=-1)

        s = torch.sigmoid(self.mlp(feats).squeeze(-1))          # (B,)
        gate_logit = self.gate(torch.cat([q, m], dim=-1))       # (B, 1)
        g = torch.sigmoid(gate_logit).squeeze(-1)               # (B,)

        # Scale cos_sim from [-1,1] to [0,1]
        cos_01 = (cos_sim + 1.0) / 2.0

        final = BETA_DENSE * cos_01 + BETA_MEM * g * s
        return final

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Inference-time helpers (no-grad, numpy ↔ torch conversion)
# ---------------------------------------------------------------------------

class FrozenScoringHead:
    """Wraps ITMAHead for inference — no gradients, numpy I/O."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        legacy = False
        state = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            # Auto-detect arch from MLP input dim:
            #   1664 = 4D + P  → legacy (v3/v4, no c*m)
            #   2048 = 5D + P  → v5+   (with c*m)
            w = state.get("mlp.0.weight")
            if w is not None and w.shape[1] == 4 * D + P:
                legacy = True
        self.model = ITMAHead(legacy=legacy).to(self.device)
        if state is not None:
            self.model.load_state_dict(state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def score(
        self,
        query_emb: np.ndarray,        # (D,) or (B, D)
        context_embs: np.ndarray,     # (B, D)
        memory_summaries: np.ndarray, # (B, D) — same for all if precomputed once per query
    ) -> np.ndarray:
        """Score each (query, context) pair. Returns (B,) float32 scores."""
        if query_emb.ndim == 1:
            q = np.tile(query_emb[None], (len(context_embs), 1))
        else:
            q = query_emb

        q_t = torch.from_numpy(q.astype(np.float32)).to(self.device)
        c_t = torch.from_numpy(context_embs.astype(np.float32)).to(self.device)
        m_t = torch.from_numpy(memory_summaries.astype(np.float32)).to(self.device)

        # Cosine similarity (both should already be L2-normalised from FAISS store)
        q_norm = F.normalize(q_t, dim=-1)
        c_norm = F.normalize(c_t, dim=-1)
        cos_sim = (q_norm * c_norm).sum(dim=-1)   # (B,)

        with torch.no_grad():
            scores = self.model(q_t, c_t, m_t, cos_sim)

        return scores.cpu().numpy().astype(np.float32)
