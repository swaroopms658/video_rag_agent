"""One-time ITMA scoring-head pretraining script.

Trains the ITMAHead on synthetic (query, positive_context, negative_context) triples
generated from held-out lecture transcripts. After training, the head is frozen
forever — no retraining at deployment time.

Usage:
    python train_itma_pretrain.py --data  data/itma_pretrain/triples.jsonl \
                                  --out   checkpoints/itma_head.pt \
                                  --epochs 10 --batch 32 --lr 2e-4

    # Colab-friendly (smaller batch, mixed precision):
    python train_itma_pretrain.py --data data/itma_pretrain/triples.jsonl \
                                  --out checkpoints/itma_head.pt \
                                  --epochs 10 --batch 16 --fp16

Dataset format (triples.jsonl):
    Each line is a JSON object:
    {
      "query":    "...",
      "positive": "...",    <- relevant context passage
      "negative": "..."     <- hard negative (BM25/top-but-non-relevant)
    }
    Optionally include synthetic memory context via:
    {
      "query": "...", "positive": "...", "negative": "...",
      "memory_queries":   ["...", ...],    <- up to 8 past queries
      "memory_contexts":  ["...", ...],    <- corresponding helpful contexts
      "memory_rewards":   [1.0, -1.0, ...]  <- whether each was helpful (+1/-1)
    }

Run scripts/build_itma_pretrain_data.py first to generate this JSONL.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.itma.scoring_head import ITMAHead, D, P
from src.itma.memory_bank import MemoryBank, DEFAULT_LAMBDA, DEFAULT_ETA

MARGIN = 0.2          # triplet loss margin
GATE_REG = 0.01       # regularisation weight pushing gate → 0 when M empty
DEFAULT_LR = 2e-4
DEFAULT_EPOCHS = 10
DEFAULT_BATCH = 32
SYNTH_MEMORY_SIZE = 8  # max entries in synthetic memory during pretrain


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TripletDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    self.items.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _encode_batch(texts: list[str], encoder) -> torch.Tensor:
    embs = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return torch.from_numpy(embs.astype(np.float32))


def _build_synth_memory(item: dict, encoder) -> tuple[np.ndarray, float]:
    """Build a synthetic memory summary m and gate confidence for one training item.

    Returns:
      m_np (D,) : memory summary vector
      gate_target (float) : 0.0 if memory is empty, 1.0 if memory has useful entries
    """
    mem_queries = item.get("memory_queries", [])
    mem_contexts = item.get("memory_contexts", [])
    if not mem_queries:
        # Empty memory → gate should be 0
        return np.zeros(D, dtype=np.float32), 0.0

    n = min(len(mem_queries), SYNTH_MEMORY_SIZE)
    mb = MemoryBank(capacity=SYNTH_MEMORY_SIZE, lam=DEFAULT_LAMBDA, eta=DEFAULT_ETA)

    q_embs = encoder.encode(mem_queries[:n], normalize_embeddings=True, show_progress_bar=False)
    c_embs = encoder.encode(mem_contexts[:n], normalize_embeddings=True, show_progress_bar=False)
    rewards = item.get("memory_rewards", [1.0] * n)

    for i in range(n):
        mb.add(q_embs[i], c_embs[i], f"synth_{i}")

    # Simulate counterfactual updates based on rewards
    for i, reward in enumerate(rewards[:n]):
        if reward < 0:
            mb.update_counterfactual([f"synth_{i}"], [1.0 / n], reward=-1.0)

    query_emb = encoder.encode([item["query"]], normalize_embeddings=True)[0]
    m = mb.attend(query_emb.astype(np.float32))
    # Gate target: 1 if memory has positive entries, 0 if all negative
    pos_count = sum(1 for r in rewards[:n] if r > 0)
    gate_target = 1.0 if pos_count > 0 else 0.0
    return m, gate_target


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_path: str,
    out_path: str,
    encoder_name: str = "all-MiniLM-L6-v2",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    margin: float = MARGIN,
    gate_reg: float = GATE_REG,
    fp16: bool = False,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(encoder_name, device=str(device))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    model = ITMAHead().to(device)
    print(f"ITMAHead params: {model.n_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if fp16 and device.type == "cuda" else None

    dataset = TripletDataset(data_path)
    print(f"Training on {len(dataset)} triples for {epochs} epochs")

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start: batch_start + batch_size]
            batch = [dataset[i] for i in batch_idx]
            B = len(batch)

            queries = [b["query"] for b in batch]
            positives = [b["positive"] for b in batch]
            negatives = [b["negative"] for b in batch]

            # Encode all texts
            all_texts = queries + positives + negatives
            all_embs = encoder.encode(
                all_texts, normalize_embeddings=True,
                show_progress_bar=False, convert_to_tensor=True,
            )
            q_embs = all_embs[:B].to(device).clone()
            p_embs = all_embs[B:2*B].to(device).clone()
            n_embs = all_embs[2*B:].to(device).clone()

            # Build synthetic memory summaries for each item in batch
            m_list, gate_targets = [], []
            for b in batch:
                m_np, gt = _build_synth_memory(b, encoder)
                m_list.append(m_np)
                gate_targets.append(gt)
            m_batch = torch.from_numpy(np.stack(m_list)).to(device)          # (B, D)
            gate_t = torch.tensor(gate_targets, dtype=torch.float32).to(device)  # (B,)

            def _cos_sim(a, b):
                return (a * b).sum(dim=-1)   # already normalised

            cos_pos = _cos_sim(q_embs, p_embs)
            cos_neg = _cos_sim(q_embs, n_embs)

            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                s_pos = model(q_embs, p_embs, m_batch, cos_pos)   # (B,)
                s_neg = model(q_embs, n_embs, m_batch, cos_neg)   # (B,)

                # Triplet contrastive loss
                triplet_loss = F.relu(margin + s_neg - s_pos).mean()

                # Gate regularisation: gate should be near 0 when memory empty
                # m_batch is zero for empty-memory items (gate_t == 0)
                gate_logits_raw = model.gate(
                    torch.cat([q_embs, m_batch], dim=-1)
                ).squeeze(-1)
                # Penalise open gate when memory is empty
                empty_mask = (gate_t == 0).float()
                gate_reg_loss = (torch.sigmoid(gate_logits_raw) * empty_mask).mean()

                loss = triplet_loss + gate_reg * gate_reg_loss

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch}/{epochs}  avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), out_path)
            print(f"  -> Saved checkpoint (loss={best_loss:.4f}) -> {out_path}")

    print(f"\nPretraining done. Best checkpoint: {out_path}  (loss={best_loss:.4f})")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pretrain ITMA scoring head (one-time)")
    parser.add_argument("--data", required=True,
                        help="Path to triples.jsonl from build_itma_pretrain_data.py")
    parser.add_argument("--out", default="checkpoints/itma_head.pt",
                        help="Output checkpoint path")
    parser.add_argument("--encoder", default="all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--gate-reg", type=float, default=GATE_REG)
    parser.add_argument("--fp16", action="store_true",
                        help="Mixed precision (CUDA only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        data_path=args.data,
        out_path=args.out,
        encoder_name=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        margin=args.margin,
        gate_reg=args.gate_reg,
        fp16=args.fp16,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
