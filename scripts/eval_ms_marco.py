"""MS-MARCO retrieval-only evaluation.

Builds a MiniLM FAISS store from the 113-passage MS-MARCO slice, then
runs BM25, Dense-MiniLM, Dense-MPNet, Cross-Encoder, and ITMA (N=0)
against 100 queries. Reports Hit@1, Hit@5, MRR@10, nDCG@10, R@10.

Usage:
    python scripts/eval_ms_marco.py
    python scripts/eval_ms_marco.py --queries data/ms_marco_slice/queries.jsonl \
                                     --passages data/ms_marco_slice/passages.jsonl \
                                     --output analysis/ms_marco \
                                     --checkpoint checkpoints/itma_head.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import tempfile

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.agent import make_chunk_id
from src.eval_utils import hit_at_k, mrr, ndcg_at_k, recall_at_k

# unused helpers left as stubs — retrieve_itma replaced by ITMARetriever direct use

QUERIES_PATH   = "data/ms_marco_slice/queries.jsonl"
PASSAGES_PATH  = "data/ms_marco_slice/passages.jsonl"
OUTPUT_DIR     = "analysis/ms_marco"
CHECKPOINT     = "checkpoints/itma_head.pt"
TOP_K          = 10
MINILM_MODEL   = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Store builder
# ---------------------------------------------------------------------------

def build_minilm_store(passages: list[dict], store_dir: str):
    """Build a MiniLM FAISS store from passage list. Saves index.faiss + meta.pkl."""
    os.makedirs(store_dir, exist_ok=True)
    model = SentenceTransformer(MINILM_MODEL, device="cpu")

    texts = [p["passage"] for p in passages]
    embs  = model.encode(texts, normalize_embeddings=True,
                         show_progress_bar=True).astype("float32")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, os.path.join(store_dir, "index.faiss"))

    meta = {
        "chunks": texts,
        "ids": [p["query_id"] for p in passages],
        "timestamps": [None] * len(texts),
    }
    with open(os.path.join(store_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"Built MS-MARCO store: {len(texts)} passages, dim={embs.shape[1]}")
    return store_dir


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def retrieve_minilm(query: str, model: SentenceTransformer,
                    faiss_index, chunks: list[str], chunk_ids: list[str],
                    top_k: int) -> list[str]:
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    _, idxs = faiss_index.search(q, top_k)
    return [chunk_ids[i] for i in idxs[0] if i >= 0]


def retrieve_bm25(query: str, bm25, chunk_ids: list[str], top_k: int) -> list[str]:
    import re
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [chunk_ids[i] for i in top_idx]


def retrieve_mpnet(query: str, mpnet_model: SentenceTransformer,
                   mpnet_index, chunk_ids: list[str], top_k: int) -> list[str]:
    q = mpnet_model.encode([query], normalize_embeddings=True).astype("float32")
    _, idxs = mpnet_index.search(q, top_k)
    return [chunk_ids[i] for i in idxs[0] if i >= 0]


def retrieve_cross_encoder(query: str, dense_index, minilm_model,
                           cross_enc, chunks: list[str], chunk_ids: list[str],
                           top_k: int, first_stage_k: int = 20) -> list[str]:
    import math
    q = minilm_model.encode([query], normalize_embeddings=True).astype("float32")
    _, idxs = dense_index.search(q, min(first_stage_k, len(chunks)))
    candidates = [(chunks[i], chunk_ids[i]) for i in idxs[0] if i >= 0]
    if not candidates:
        return []
    pairs  = [(query, c[0]) for c in candidates]
    logits = cross_enc.predict(pairs)
    scored = sorted(zip(candidates, logits), key=lambda x: x[1], reverse=True)
    return [cid for (_, cid), _ in scored[:top_k]]



# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(retrieved_ids: list[str], gold_id: str) -> dict:
    gold = [gold_id]
    return {
        "hit_at_1":   hit_at_k(retrieved_ids, gold, 1),
        "hit_at_5":   hit_at_k(retrieved_ids, gold, 5),
        "mrr_score":  mrr(retrieved_ids, gold),
        "ndcg_at_10": ndcg_at_k(retrieved_ids, gold, 10),
        "recall_at_10": recall_at_k(retrieved_ids, gold, 10),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    os.makedirs(args.output, exist_ok=True)

    # Load data
    queries  = [json.loads(l) for l in open(args.queries,  encoding="utf-8")]
    passages = [json.loads(l) for l in open(args.passages, encoding="utf-8")]

    # Build a mapping: query_id → gold passage text
    gold_map: dict[str, str] = {}
    for p in passages:
        if p.get("is_selected") == 1:
            gold_map[p["query_id"]] = p["passage"]

    # Only keep queries that have a gold passage
    queries = [q for q in queries if q["query_id"] in gold_map]
    print(f"Queries with gold passage: {len(queries)}")

    # Build store
    store_dir = os.path.join(args.output, "store")
    build_minilm_store(passages, store_dir)

    with open(os.path.join(store_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    chunks     = meta["chunks"]
    chunk_ids  = [make_chunk_id(c) for c in chunks]

    # Gold chunk IDs
    gold_chunk_ids = {q["query_id"]: make_chunk_id(gold_map[q["query_id"]])
                      for q in queries}

    # Load FAISS + models
    print("Loading models...")
    minilm_model = SentenceTransformer(MINILM_MODEL, device="cpu")
    faiss_index  = faiss.read_index(os.path.join(store_dir, "index.faiss"))

    systems = {}

    # BM25
    from rank_bm25 import BM25Okapi
    import re
    tokenized = [re.findall(r"[a-z0-9]+", c.lower()) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    systems["bm25"] = lambda q: retrieve_bm25(q, bm25, chunk_ids, TOP_K)

    # Dense-MiniLM
    systems["dense_minilm"] = lambda q: retrieve_minilm(
        q, minilm_model, faiss_index, chunks, chunk_ids, TOP_K)

    # Dense-MPNet (rebuild index with mpnet embeddings)
    print("Building MPNet index...")
    mpnet_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    mpnet_embs  = mpnet_model.encode(chunks, normalize_embeddings=True,
                                     show_progress_bar=False).astype("float32")
    mpnet_index = faiss.IndexFlatIP(mpnet_embs.shape[1])
    mpnet_index.add(mpnet_embs)
    systems["dense_mpnet"] = lambda q: retrieve_mpnet(
        q, mpnet_model, mpnet_index, chunk_ids, TOP_K)

    # Cross-Encoder
    print("Loading Cross-Encoder...")
    from sentence_transformers import CrossEncoder
    cross_enc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    systems["cross_encoder"] = lambda q: retrieve_cross_encoder(
        q, faiss_index, minilm_model, cross_enc, chunks, chunk_ids, TOP_K)

    # ITMA (N=0)
    if os.path.exists(args.checkpoint):
        print("Loading ITMA head...")
        from src.itma.integration import ITMARetriever
        itma_retriever = ITMARetriever(
            store_path=store_dir,
            checkpoint=args.checkpoint,
            memory_path=None,
            memory_capacity=50,
        )
        systems["itma"] = lambda q: [
            cid for _, _, cid in
            itma_retriever.retrieve_with_ids(q, model=None, top_k=TOP_K)
        ]
    else:
        print(f"ITMA checkpoint not found at {args.checkpoint}, skipping.")

    # Evaluate each system
    for sys_name, retrieve_fn in systems.items():
        print(f"\n--- {sys_name} ---")
        rows = []
        metric_accum = {k: [] for k in
                        ["hit_at_1", "hit_at_5", "mrr_score", "ndcg_at_10", "recall_at_10"]}

        for q in queries:
            qid   = q["query_id"]
            query = q["query"]
            gold  = gold_chunk_ids[qid]

            try:
                retrieved = retrieve_fn(query)
            except Exception as e:
                print(f"  Error on {qid}: {e}")
                retrieved = []

            metrics = compute_metrics(retrieved, gold)
            for k, v in metrics.items():
                metric_accum[k].append(v)

            rows.append({"query_id": qid, "query": query,
                         "gold_passage": gold_map[qid][:80], **metrics})

        # Write CSV
        csv_path = os.path.join(args.output, f"{sys_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        # Print summary
        n = len(rows)
        print(f"  n={n}  "
              + "  ".join(f"{k}={sum(v)/n:.3f}"
                          for k, v in metric_accum.items()))

    print(f"\nDone. Results in {args.output}/")


def main():
    parser = argparse.ArgumentParser(description="MS-MARCO retrieval eval")
    parser.add_argument("--queries",    default=QUERIES_PATH)
    parser.add_argument("--passages",   default=PASSAGES_PATH)
    parser.add_argument("--output",     default=OUTPUT_DIR)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
