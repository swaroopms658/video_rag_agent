"""CFRAG-lite baseline — faithful re-implementation of CFRAG-style per-deployment
fine-tuning (Chen et al., arXiv:2504.05731).

Design:
  - A small cross-encoder (ms-marco-MiniLM-L-6-v2) is fine-tuned on the
    LectureRAG-75 *train* split (feedback-labelled query-context pairs).
  - At evaluation time, the fine-tuned model reranks dense-MiniLM candidates.
  - ITMA comparison: CFRAG-lite requires a labelled train split and offline
    gradient updates; ITMA requires neither.

Training:
    python -m src.baselines.cfrag_lite --train data/lecture_rag_75/train.jsonl \
                                        --store  data/lecture_rag_75 \
                                        --out    checkpoints/cfrag_lite

Evaluation (via evaluate.py):
    cfrag_retriever = CFRAGLiteRetriever(store_path, checkpoint="checkpoints/cfrag_lite")
    agent = BaselineAgent("cfrag_lite", cfrag_retriever, rag_agent)
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.baselines import BaseRetriever
from src.agent import SimpleRetriever, make_chunk_id

CROSS_ENCODER_BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FIRST_STAGE_K = 20
DEFAULT_LR = 2e-5
DEFAULT_EPOCHS = 5
DEFAULT_BATCH = 16


# ---------------------------------------------------------------------------
# Dataset for fine-tuning
# ---------------------------------------------------------------------------

class FeedbackDataset(Dataset):
    """
    Each item in the JSONL file is:
      { "question": "...", "context": "...", "label": 1 or 0 }
    where label=1 means the context was helpful for this question.
    """

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
        item = self.items[idx]
        return item["question"], item["context"], float(item["label"])


def _collate(batch):
    questions, contexts, labels = zip(*batch)
    return list(questions), list(contexts), torch.tensor(labels, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune(
    train_path: str,
    out_dir: str,
    base_model: str = CROSS_ENCODER_BASE,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
):
    """Fine-tune a cross-encoder on labelled query-context pairs."""
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

    os.makedirs(out_dir, exist_ok=True)

    dataset = FeedbackDataset(train_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)

    ce = CrossEncoder(base_model, num_labels=1)

    # Prepare optimizer and loss
    optimizer = torch.optim.AdamW(ce.model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    tokenizer = ce.tokenizer
    device = next(ce.model.parameters()).device

    print(f"Fine-tuning CFRAG-lite on {len(dataset)} examples, {epochs} epochs ...")
    for epoch in range(1, epochs + 1):
        ce.model.train()
        total_loss = 0.0
        n_batches = 0
        for questions, contexts, labels in loader:
            encoded = tokenizer(
                list(questions), list(contexts),
                padding=True, truncation=True, max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = ce.model(**encoded)
            logits = outputs.logits.squeeze(-1).float()
            labels = labels.to(device)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}/{epochs}  avg_loss={avg:.4f}")

    ce.save(out_dir)
    print(f"Saved fine-tuned model -> {out_dir}")
    return ce


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class CFRAGLiteRetriever(BaseRetriever):
    """Cross-encoder retriever fine-tuned per-deployment on feedback labels."""

    def __init__(self, store_path: str, checkpoint: Optional[str] = None):
        from sentence_transformers import CrossEncoder
        self._dense = SimpleRetriever(store_path)
        model_path = checkpoint if checkpoint else CROSS_ENCODER_BASE
        self._ce = CrossEncoder(model_path)
        self._checkpoint = model_path

    # ------------------------------------------------------------------

    def retrieve(self, query: str, model, top_k: int = 3, boost_ids=None) -> list:
        return [(c, s) for c, s, _ in self._rerank(query, model, top_k)]

    def retrieve_with_ids(self, query: str, model, top_k: int = 10, boost_ids=None) -> list:
        return self._rerank(query, model, top_k)

    # ------------------------------------------------------------------

    def _rerank(self, query: str, model, top_k: int) -> list:
        candidates = self._dense.retrieve_with_ids(query, model, top_k=FIRST_STAGE_K)
        if not candidates:
            return []
        pairs = [(query, c[0]) for c in candidates]
        logits = self._ce.predict(pairs)

        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        scored = sorted(zip(candidates, logits), key=lambda x: x[1], reverse=True)
        return [
            (chunk, _sigmoid(float(logit)), cid)
            for (chunk, _, cid), logit in scored[:top_k]
        ]


# ---------------------------------------------------------------------------
# CLI for standalone fine-tuning
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune CFRAG-lite cross-encoder")
    parser.add_argument("--train", required=True, help="JSONL with query/context/label")
    parser.add_argument("--store", required=True, help="Vector store path")
    parser.add_argument("--out", default="checkpoints/cfrag_lite")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    args = parser.parse_args()

    finetune(args.train, args.out, epochs=args.epochs, lr=args.lr, batch_size=args.batch)


if __name__ == "__main__":
    main()
