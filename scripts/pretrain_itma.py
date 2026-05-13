"""Pretrain the ITMAHead with a 3-stage memory-aware curriculum.

Stage A — Backbone warmup (gate frozen, m=0)
    Loss = pairwise margin on score(q, pos) > score(q, neg).
    Teaches the MLP to score (q, c) using cosine + dense features.
    Gate parameters are frozen (bias stays at -3 → σ ≈ 0.047), preserving
    cold-start safety as a *learned* property, not just an init artefact.

Stage B — Gate opening on helpful memory
    For each triple, sample K (q_j, c_j) pairs from the SAME source as
    the triple. Encode them, build memory summary `m` via the same
    softmax attention as MemoryBank.attend(). Train with:
        margin loss + λ_gate · BCE(σ(gate), target=1)
    so the gate learns to open when memory is informative.

Stage C — Robustness on noisy memory
    For half the batch, sample memory from a DIFFERENT source
    (unhelpful / off-topic). BCE target=0 on those examples.
    Teaches the gate to close on noise.

Empty-memory examples (m=0) are mixed throughout B and C so the gate
keeps the cold-start guarantee even after unfreezing.

Usage:
    python scripts/pretrain_itma.py
    python scripts/pretrain_itma.py --triples data/itma_pretrain/triples.jsonl \
        --out checkpoints/itma_head_v5.pt \
        --epochs-a 5 --epochs-b 5 --epochs-c 5 --batch-size 64

Outputs:
    checkpoints/itma_head_v5.pt              (final weights)
    checkpoints/itma_head_v5.train_log.json  (loss / gate stats per epoch)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Local imports
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.itma.scoring_head import ITMAHead, BETA_DENSE, BETA_MEM, D, P  # noqa: E402

ENCODER_NAME = "all-MiniLM-L6-v2"
DEFAULT_TRIPLES = "data/itma_pretrain/triples.jsonl"
DEFAULT_OUT = "checkpoints/itma_head_v5.pt"
ATTEND_TEMPERATURE = 0.1            # matches MemoryBank.attend()
MEM_K = 4                           # neighbours per memory variant
MARGIN = 0.1
LAMBDA_GATE = 3.0                   # weight on gate BCE loss
GATE_LR_MULT = 10.0                 # gate params train at LR * this multiplier


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_triples(path: str) -> list[dict]:
    triples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not all(k in t for k in ("query", "positive", "negative")):
                continue
            t.setdefault("source", "_unknown")
            triples.append(t)
    return triples


class TripleDataset(Dataset):
    """Holds (q, pos, neg, source) and indexes per-source for memory sampling."""

    def __init__(self, triples: list[dict]):
        self.triples = triples
        self.by_source: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(triples):
            self.by_source[t["source"]].append(i)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return idx  # we resolve in collate_fn so we can sample memory


# ---------------------------------------------------------------------------
# Embedding cache — encode every distinct text once
# ---------------------------------------------------------------------------

class EmbCache:
    """Encodes every unique string in the corpus exactly once with MiniLM."""

    def __init__(self, encoder, device: str):
        self.encoder = encoder
        self.device = device
        self._cache: dict[str, np.ndarray] = {}

    def encode_many(self, texts: list[str], batch_size: int = 128):
        new = [t for t in texts if t not in self._cache]
        new = list(dict.fromkeys(new))  # dedupe, preserve order
        if not new:
            return
        print(f"  encoding {len(new)} new texts...")
        embs = self.encoder.encode(
            new,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        for t, e in zip(new, embs):
            self._cache[t] = e

    def get(self, text: str) -> np.ndarray:
        return self._cache[text]

    def get_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._cache[t] for t in texts])


# ---------------------------------------------------------------------------
# Memory construction (mirrors MemoryBank.attend(), uniform w_cf=1, age=0)
# ---------------------------------------------------------------------------

def build_memory_summary(
    q_emb: np.ndarray,                  # (D,)
    mem_query_embs: np.ndarray,         # (K, D)
    mem_context_embs: np.ndarray,       # (K, D)
    temperature: float = ATTEND_TEMPERATURE,
) -> np.ndarray:
    """Replicate MemoryBank.attend() exactly. Returns (D,)."""
    if mem_query_embs.shape[0] == 0:
        return np.zeros_like(q_emb)
    q = q_emb / (np.linalg.norm(q_emb) + 1e-8)
    norms = np.linalg.norm(mem_query_embs, axis=1, keepdims=True) + 1e-8
    q_normed = mem_query_embs / norms
    cos_sims = q_normed @ q                              # (K,)
    weights = np.ones_like(cos_sims, dtype=np.float32)   # w_cf=1, age=0
    logits = weights * cos_sims / temperature
    logits -= logits.max()
    alpha = np.exp(logits)
    alpha /= (alpha.sum() + 1e-8)
    return (alpha @ mem_context_embs).astype(np.float32)


def sample_memory_pool(
    triple_idx: int,
    triples: list[dict],
    by_source: dict[str, list[int]],
    rng: random.Random,
    mode: str,                          # "empty" | "helpful" | "unhelpful"
    k: int = MEM_K,
) -> list[int]:
    """Return a list of triple indices to use as memory entries. Empty list = m=0."""
    if mode == "empty":
        return []
    src = triples[triple_idx]["source"]
    if mode == "helpful":
        candidates = [i for i in by_source[src] if i != triple_idx]
    else:  # unhelpful
        candidates = [
            i for s, idxs in by_source.items() if s != src for i in idxs
        ]
    if not candidates:
        return []
    return rng.sample(candidates, min(k, len(candidates)))


# ---------------------------------------------------------------------------
# Batch building
# ---------------------------------------------------------------------------

def build_batch(
    indices: list[int],
    triples: list[dict],
    by_source: dict[str, list[int]],
    cache: EmbCache,
    rng: random.Random,
    mode_mix: dict[str, float],         # e.g. {"empty": 0.3, "helpful": 0.7}
    device: torch.device,
) -> dict:
    """Build a batch dict with q, pos, neg, m embeddings + gate targets."""
    modes_list, gate_targets_list = [], []
    for _ in indices:
        r = rng.random()
        cum = 0.0
        for mode, p in mode_mix.items():
            cum += p
            if r <= cum:
                modes_list.append(mode)
                gate_targets_list.append(1.0 if mode == "helpful" else 0.0)
                break
        else:
            modes_list.append("empty")
            gate_targets_list.append(0.0)

    q_embs, pos_embs, neg_embs, m_pos_embs, m_neg_embs = [], [], [], [], []
    for idx, mode in zip(indices, modes_list):
        t = triples[idx]
        q_emb = cache.get(t["query"])
        pos_emb = cache.get(t["positive"])
        neg_emb = cache.get(t["negative"])

        mem_idxs = sample_memory_pool(idx, triples, by_source, rng, mode)
        if mem_idxs:
            mem_q = np.stack([cache.get(triples[i]["query"]) for i in mem_idxs])
            mem_c = np.stack([cache.get(triples[i]["positive"]) for i in mem_idxs])
            m_pos = build_memory_summary(q_emb, mem_q, mem_c)
            # For the negative candidate, the same memory summary applies
            # (memory is per-query, not per-candidate).
            m_neg = m_pos
        else:
            m_pos = np.zeros_like(q_emb)
            m_neg = np.zeros_like(q_emb)

        q_embs.append(q_emb)
        pos_embs.append(pos_emb)
        neg_embs.append(neg_emb)
        m_pos_embs.append(m_pos)
        m_neg_embs.append(m_neg)

    def to_t(arr):
        return torch.from_numpy(np.stack(arr)).to(device)

    return {
        "q":    to_t(q_embs),
        "pos":  to_t(pos_embs),
        "neg":  to_t(neg_embs),
        "m_pos": to_t(m_pos_embs),
        "m_neg": to_t(m_neg_embs),
        "gate_target": torch.tensor(gate_targets_list, dtype=torch.float32, device=device),
        "modes": modes_list,
    }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def head_score_with_gate(
    head: ITMAHead,
    q: torch.Tensor,
    c: torch.Tensor,
    m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward with explicit return of gate value + raw mlp score for diagnostics."""
    q_proj = head.proj_q(q)
    m_proj = head.proj_m(m)
    feats = torch.cat([q, c, m, q * c, c * m, q_proj * m_proj], dim=-1)
    s = torch.sigmoid(head.mlp(feats).squeeze(-1))
    gate_logit = head.gate(torch.cat([q, m], dim=-1)).squeeze(-1)
    g = torch.sigmoid(gate_logit)
    cos_sim = F.cosine_similarity(q, c, dim=-1)
    cos_01 = (cos_sim + 1.0) / 2.0
    final = BETA_DENSE * cos_01 + BETA_MEM * g * s
    return final, g, s


def compute_loss(
    head: ITMAHead,
    batch: dict,
    use_gate_loss: bool,
) -> tuple[torch.Tensor, dict]:
    score_pos, g_pos, _ = head_score_with_gate(head, batch["q"], batch["pos"], batch["m_pos"])
    score_neg, _, _ = head_score_with_gate(head, batch["q"], batch["neg"], batch["m_neg"])

    margin_loss = F.relu(MARGIN - (score_pos - score_neg)).mean()
    total = margin_loss

    metrics = {
        "margin_loss": float(margin_loss.detach()),
        "gate_mean": float(g_pos.mean().detach()),
        "score_pos_mean": float(score_pos.mean().detach()),
        "score_neg_mean": float(score_neg.mean().detach()),
    }

    if use_gate_loss:
        # BCE between σ(gate) and target. Use g_pos (gate seen by the positive
        # candidate, which is the one whose score we want memory to lift).
        gate_loss = F.binary_cross_entropy(
            g_pos.clamp(1e-6, 1 - 1e-6),
            batch["gate_target"],
        )
        total = total + LAMBDA_GATE * gate_loss
        metrics["gate_loss"] = float(gate_loss.detach())

    metrics["total_loss"] = float(total.detach())
    return total, metrics


# ---------------------------------------------------------------------------
# Training stages
# ---------------------------------------------------------------------------

STAGE_CONFIGS = {
    "A": {
        "mode_mix":      {"empty": 1.0},
        "freeze_gate":   True,
        "use_gate_loss": False,
        "lr":            3e-4,
    },
    "B": {
        "mode_mix":      {"empty": 0.3, "helpful": 0.7},
        "freeze_gate":   False,
        "use_gate_loss": True,
        "lr":            1e-3,
    },
    "C": {
        "mode_mix":      {"empty": 0.2, "helpful": 0.4, "unhelpful": 0.4},
        "freeze_gate":   False,
        "use_gate_loss": True,
        "lr":            5e-4,
    },
}


def set_gate_trainable(head: ITMAHead, trainable: bool):
    for p in head.gate.parameters():
        p.requires_grad_(trainable)


def run_stage(
    stage: str,
    head: ITMAHead,
    triples: list[dict],
    by_source: dict[str, list[int]],
    cache: EmbCache,
    epochs: int,
    batch_size: int,
    rng: random.Random,
    device: torch.device,
    log: list,
):
    cfg = STAGE_CONFIGS[stage]
    set_gate_trainable(head, not cfg["freeze_gate"])

    # Two param groups: gate params get a higher LR so they can escape the
    # bias=-3 init quickly. Stage A freezes the gate so this is a no-op there.
    gate_param_ids = {id(p) for p in head.gate.parameters()}
    body_params = [p for p in head.parameters()
                   if p.requires_grad and id(p) not in gate_param_ids]
    gate_params = [p for p in head.gate.parameters() if p.requires_grad]
    param_groups = [{"params": body_params, "lr": cfg["lr"]}]
    if gate_params:
        param_groups.append({"params": gate_params, "lr": cfg["lr"] * GATE_LR_MULT})
    opt = torch.optim.Adam(param_groups)
    params = body_params + gate_params

    n = len(triples)
    indices = list(range(n))
    print(f"\n=== Stage {stage}  epochs={epochs}  lr={cfg['lr']}  "
          f"freeze_gate={cfg['freeze_gate']}  mode_mix={cfg['mode_mix']} ===")

    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        head.train()
        epoch_metrics = defaultdict(list)
        n_batches = 0
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            if len(batch_idx) < 2:
                continue
            batch = build_batch(
                batch_idx, triples, by_source, cache,
                rng, cfg["mode_mix"], device,
            )
            opt.zero_grad()
            loss, m = compute_loss(head, batch, cfg["use_gate_loss"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            for k, v in m.items():
                epoch_metrics[k].append(v)
            n_batches += 1

        avg = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}
        avg.update({"stage": stage, "epoch": epoch, "n_batches": n_batches})
        log.append(avg)
        print(
            f"  epoch {epoch}/{epochs}  "
            f"loss={avg.get('total_loss', 0):.4f}  "
            f"margin={avg.get('margin_loss', 0):.4f}  "
            f"gate_mean={avg.get('gate_mean', 0):.4f}  "
            f"pos={avg.get('score_pos_mean', 0):.4f}  "
            f"neg={avg.get('score_neg_mean', 0):.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", default=DEFAULT_TRIPLES)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--epochs-a", type=int, default=5)
    parser.add_argument("--epochs-b", type=int, default=5)
    parser.add_argument("--epochs-c", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None,
                        help="cuda | cpu (auto-detect if omitted)")
    parser.add_argument("--encoder", default=ENCODER_NAME)
    parser.add_argument("--no-stage-a", action="store_true")
    parser.add_argument("--no-stage-b", action="store_true")
    parser.add_argument("--no-stage-c", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"device: {device}")

    triples = load_triples(args.triples)
    if not triples:
        print(f"ERROR: no triples in {args.triples}")
        return
    print(f"loaded {len(triples)} triples from {args.triples}")
    by_source = defaultdict(list)
    for i, t in enumerate(triples):
        by_source[t["source"]].append(i)
    n_sources = len(by_source)
    src_sizes = sorted([len(v) for v in by_source.values()], reverse=True)
    print(f"  {n_sources} unique source(s)  triple counts (top): {src_sizes[:5]}")
    if n_sources < 2:
        print("  WARNING: <2 sources — Stage C (unhelpful memory) will degenerate.")

    # Encode all texts once
    from sentence_transformers import SentenceTransformer
    print(f"loading encoder: {args.encoder}")
    encoder = SentenceTransformer(args.encoder, device=device_str)
    cache = EmbCache(encoder, device_str)
    all_texts = []
    for t in triples:
        all_texts.extend([t["query"], t["positive"], t["negative"]])
    t0 = time.time()
    cache.encode_many(all_texts)
    print(f"  encoded {len(cache._cache)} unique texts in {time.time() - t0:.1f}s")

    head = ITMAHead().to(device)
    print(f"head params: {head.n_params():,}")

    log: list = []
    rng = random.Random(args.seed)

    if not args.no_stage_a:
        run_stage("A", head, triples, by_source, cache,
                  args.epochs_a, args.batch_size, rng, device, log)
    if not args.no_stage_b:
        run_stage("B", head, triples, by_source, cache,
                  args.epochs_b, args.batch_size, rng, device, log)
    if not args.no_stage_c:
        run_stage("C", head, triples, by_source, cache,
                  args.epochs_c, args.batch_size, rng, device, log)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(head.state_dict(), args.out)
    print(f"\nsaved checkpoint -> {args.out}")

    log_path = args.out.replace(".pt", ".train_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"saved log -> {log_path}")

    # Quick gate diagnostic
    head.eval()
    with torch.no_grad():
        sample = build_batch(
            list(range(min(64, len(triples)))),
            triples, by_source, cache, random.Random(0),
            {"empty": 1.0}, device,
        )
        _, g_empty, _ = head_score_with_gate(head, sample["q"], sample["pos"], sample["m_pos"])
        sample = build_batch(
            list(range(min(64, len(triples)))),
            triples, by_source, cache, random.Random(0),
            {"helpful": 1.0}, device,
        )
        _, g_help, _ = head_score_with_gate(head, sample["q"], sample["pos"], sample["m_pos"])
    print(
        f"\nfinal gate diagnostic: "
        f"sigmoid(gate) on empty memory  = {float(g_empty.mean()):.4f}\n"
        f"                       helpful = {float(g_help.mean()):.4f}"
    )
    print("   (want: empty ~ 0 for cold-start safety; helpful >> empty)")


if __name__ == "__main__":
    main()
