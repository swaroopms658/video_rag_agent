"""Latency micro-benchmark for ITMA — per-query retrieval overhead and the
counterfactual memory update cost, on a commodity CPU.

Backs the deployability claim in the paper (Section 3, Design Philosophy /
Implementation Details): ITMA adds negligible latency over the dense
first-stage retriever, and the O(|M|) weight update is sub-millisecond for the
cold-start memory sizes of interest.

What it measures (all CPU, wall-clock via time.perf_counter):

  1. RETRIEVAL latency, per query, top_k=10:
       - dense_minilm  (SimpleRetriever: query encode + FAISS top-10)
       - itma          (full: head + ID-boost) at memory sizes |M| in {0, 10, 50}
     Reported: mean / median / p95 / std, and the ITMA-minus-dense overhead.

  2. UPDATE latency, per query: record_feedback (memory add + counterfactual
     reweighting) at |M| in {10, 50}.

Methodology:
  - The chunk-embedding cache and the encoder/torch graph are warmed with an
    untimed pass before any timing, so we measure deployment steady state
    (chunk embeddings are precomputed in a real deployment, not re-encoded).
  - Each test query is timed individually; the whole test set is replayed
    ROUNDS times to gather enough samples.

Usage:
    python scripts/latency_benchmark.py \
        --qa data/lecture_rag_75/qa.jsonl \
        --splits data/lecture_rag_75/splits.json \
        --store data/lecture_rag_75/combined \
        --checkpoint-itma checkpoints/itma_head_v5.pt \
        --lam 0.01 --eta 0.05 \
        --out analysis/latency.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import time

import numpy as np


def load_items(qa_path: str, splits_path: str, split: str) -> list[dict]:
    items = []
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if split == "all" or splits_path is None:
        return items
    with open(splits_path) as f:
        splits = json.load(f)
    allowed = set(splits.get(split, []))
    return [it for it in items if it.get("id") in allowed]


def summarise(times_ms: list[float]) -> dict:
    """Summary stats for a list of per-query latencies in milliseconds."""
    s = sorted(times_ms)
    n = len(s)
    p95 = s[min(n - 1, int(round(0.95 * (n - 1))))]
    return {
        "n": n,
        "mean_ms": statistics.mean(s),
        "median_ms": statistics.median(s),
        "p95_ms": p95,
        "std_ms": statistics.pstdev(s) if n > 1 else 0.0,
    }


def grow_memory(retriever, embed_model, feed_items: list[dict], target: int):
    """Feed oracle feedback until the memory bank reaches `target` entries."""
    retriever.reset_memory()
    for it in feed_items:
        if retriever._memory.size() >= target:
            break
        gold = it.get("gold_context_ids", [])
        if not gold:
            continue
        retriever.retrieve_with_ids(it["question"], embed_model, top_k=10)
        retriever.record_feedback(helpful_chunk_ids=gold, reward=1.0)
    return retriever._memory.size()


def time_retrieval(retriever, embed_model, test_items: list[dict],
                   rounds: int) -> list[float]:
    """Warm the cache, then time each retrieve_with_ids call (ms)."""
    # Warm-up (untimed): fills chunk-embedding cache + torch graph.
    for it in test_items:
        retriever.retrieve_with_ids(it["question"], embed_model, top_k=10)
    times_ms = []
    for _ in range(rounds):
        for it in test_items:
            t0 = time.perf_counter()
            retriever.retrieve_with_ids(it["question"], embed_model, top_k=10)
            times_ms.append((time.perf_counter() - t0) * 1e3)
    return times_ms


def time_update(retriever, embed_model, test_items: list[dict],
                target: int, feed_items: list[dict], rounds: int) -> list[float]:
    """Time record_feedback (memory add + counterfactual update) at |M|≈target.

    Memory is grown to `target` once; because add() deduplicates by chunk_id
    (gold chunks are already stored), the bank stays ≈target across the loop,
    while update_counterfactual + the add() dedup scan still exercise the full
    O(|M|) path on every call.
    """
    actual = grow_memory(retriever, embed_model, feed_items, target)
    times_ms = []
    timed = [it for it in test_items if it.get("gold_context_ids")]
    for _ in range(rounds):
        for it in timed:
            # Populate internal state for this query (untimed).
            retriever.retrieve_with_ids(it["question"], embed_model, top_k=10)
            gold = it["gold_context_ids"]
            t0 = time.perf_counter()
            retriever.record_feedback(helpful_chunk_ids=gold, reward=1.0)
            times_ms.append((time.perf_counter() - t0) * 1e3)
    return times_ms, actual


def main():
    ap = argparse.ArgumentParser(description="ITMA latency micro-benchmark")
    ap.add_argument("--qa", default="data/lecture_rag_75/qa.jsonl")
    ap.add_argument("--splits", default="data/lecture_rag_75/splits.json")
    ap.add_argument("--store", default="data/lecture_rag_75/combined")
    ap.add_argument("--checkpoint-itma", default="checkpoints/itma_head_v5.pt")
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--mem-sizes", nargs="+", type=int, default=[0, 10, 50])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="analysis/latency.csv")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from src.agent import SimpleRetriever
    from src.itma.integration import ITMARetriever

    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    test_items = load_items(args.qa, args.splits, args.split)
    feed_items = load_items(args.qa, args.splits, "all")  # pool to grow memory
    print(f"Test items: {len(test_items)} (split={args.split}); "
          f"feed pool: {len(feed_items)}")

    cpu = platform.processor() or platform.machine()
    print(f"CPU: {cpu} | {platform.system()} {platform.release()} | "
          f"python {platform.python_version()}")

    rows = []

    # ---- 1. Dense baseline retrieval ----
    print("\n[dense_minilm] retrieval ...")
    dense = SimpleRetriever(args.store)
    dense_ms = time_retrieval(dense, embed_model, test_items, args.rounds)
    dstat = summarise(dense_ms)
    print(f"  mean {dstat['mean_ms']:.3f} ms  median {dstat['median_ms']:.3f}  "
          f"p95 {dstat['p95_ms']:.3f}")
    rows.append({"phase": "retrieval", "system": "dense_minilm",
                 "mem_size": "-", **dstat, "overhead_vs_dense_ms": 0.0})
    dense_mean = dstat["mean_ms"]

    # ---- 2. ITMA retrieval at each memory size ----
    itma = ITMARetriever(args.store, checkpoint=args.checkpoint_itma,
                         memory_path=None, lam=args.lam, eta=args.eta)
    for msize in args.mem_sizes:
        actual = grow_memory(itma, embed_model, feed_items, msize) if msize > 0 \
            else (itma.reset_memory() or 0)
        print(f"\n[itma] retrieval at |M|={actual} (target {msize}) ...")
        ms = time_retrieval(itma, embed_model, test_items, args.rounds)
        st = summarise(ms)
        print(f"  mean {st['mean_ms']:.3f} ms  median {st['median_ms']:.3f}  "
              f"p95 {st['p95_ms']:.3f}  overhead {st['mean_ms']-dense_mean:+.3f}")
        rows.append({"phase": "retrieval", "system": "itma",
                     "mem_size": actual, **st,
                     "overhead_vs_dense_ms": st["mean_ms"] - dense_mean})

    # ---- 3. Counterfactual update latency ----
    for msize in [s for s in args.mem_sizes if s > 0]:
        ms, actual = time_update(itma, embed_model, test_items, msize,
                                 feed_items, args.rounds)
        print(f"\n[itma] update (record_feedback) at |M|={actual} ...")
        st = summarise(ms)
        print(f"  mean {st['mean_ms']:.4f} ms  median {st['median_ms']:.4f}  "
              f"p95 {st['p95_ms']:.4f}")
        rows.append({"phase": "update", "system": "itma",
                     "mem_size": actual, **st, "overhead_vs_dense_ms": "-"})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = ["phase", "system", "mem_size", "n", "mean_ms", "median_ms",
              "p95_ms", "std_ms", "overhead_vs_dense_ms"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nWrote {len(rows)} rows -> {args.out}")
    print(f"CPU recorded: {cpu}")


if __name__ == "__main__":
    main()
