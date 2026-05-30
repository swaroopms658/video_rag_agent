"""Component-level latency breakdown for ITMA retrieval (CPU).

The end-to-end benchmark (scripts/latency_benchmark.py) measures total per-query
latency, but the ITMARetriever currently encodes the query twice (once in
_rank, once inside the dense first stage), so the raw "overhead vs dense" number
double-counts query encoding. This script replicates the _rank pipeline
step-by-step with the real objects and times each stage, so the paper can
attribute latency honestly:

  encode        — model.encode([q])          (shared with the dense baseline)
  faiss         — index.search(q_emb, 20)    (shared with the dense baseline)
  attend        — MemoryBank.attend(q_emb)    ITMA-specific, O(|M|)
  head          — FrozenScoringHead.score(...) over 20 candidates (ITMA-specific)
  boost+alpha   — ID-boost + attention tracking, O(|M|)  (ITMA-specific)

"ITMA-added compute" = attend + head + boost+alpha (the cost beyond what the
dense first stage already pays). The query encode is counted once, as a
deployment would.

Usage:
    python scripts/latency_components.py --mem-size 50 --rounds 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np


def load_items(qa_path, splits_path, split):
    items = []
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if split == "all":
        return items
    with open(splits_path) as f:
        splits = json.load(f)
    allowed = set(splits.get(split, []))
    return [it for it in items if it.get("id") in allowed]


def grow_memory(r, model, feed, target):
    r.reset_memory()
    for it in feed:
        if r._memory.size() >= target:
            break
        gold = it.get("gold_context_ids", [])
        if not gold:
            continue
        r.retrieve_with_ids(it["question"], model, top_k=10)
        r.record_feedback(helpful_chunk_ids=gold, reward=1.0)
    return r._memory.size()


def stat(xs):
    return f"{statistics.mean(xs):7.3f} ms  (median {statistics.median(xs):6.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="data/lecture_rag_75/qa.jsonl")
    ap.add_argument("--splits", default="data/lecture_rag_75/splits.json")
    ap.add_argument("--store", default="data/lecture_rag_75/combined")
    ap.add_argument("--checkpoint-itma", default="checkpoints/itma_head_v5.pt")
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--mem-size", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from src.itma.integration import ITMARetriever, FIRST_STAGE_K

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    test = load_items(args.qa, args.splits, "test")
    feed = load_items(args.qa, args.splits, "all")

    r = ITMARetriever(args.store, checkpoint=args.checkpoint_itma,
                      memory_path=None, lam=args.lam, eta=args.eta)
    actual = grow_memory(r, model, feed, args.mem_size)
    print(f"|M| = {actual};  {len(test)} test queries x {args.rounds} rounds")

    # Warm caches (chunk embeddings + torch graph).
    for it in test:
        r.retrieve_with_ids(it["question"], model, top_k=10)

    t = {k: [] for k in ["encode", "faiss", "attend", "head", "boost", "total_added"]}

    for _ in range(args.rounds):
        for it in test:
            q = it["question"]

            t0 = time.perf_counter()
            q_emb = model.encode(q, normalize_embeddings=True).astype(np.float32)
            t["encode"].append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            cands = r._dense.retrieve_with_ids(q, model, top_k=FIRST_STAGE_K)
            # subtract a second encode that retrieve_with_ids does internally:
            # we measure faiss as the search-only cost below instead.
            _ = cands
            # search-only timing on the already-computed embedding:
            qn = q_emb[None]
            t1 = time.perf_counter()
            r._dense._index.search(qn, FIRST_STAGE_K)
            t["faiss"].append((time.perf_counter() - t1) * 1e3)

            c_texts = [c[0] for c in cands]
            c_ids = [c[2] for c in cands]
            c_embs = r._get_chunk_embeddings(c_texts, c_ids, model)

            t0 = time.perf_counter()
            m = r._memory.attend(q_emb)
            t["attend"].append((time.perf_counter() - t0) * 1e3)
            m_batch = np.tile(m[None], (len(c_texts), 1))

            t0 = time.perf_counter()
            _ = r._head.score(q_emb, c_embs, m_batch)
            t["head"].append((time.perf_counter() - t0) * 1e3)

            # ID-boost + attention tracking, O(|M|)
            t0 = time.perf_counter()
            if r._memory.size() > 0:
                qn_vec = q_emb / (np.linalg.norm(q_emb) + 1e-8)
                c_ids_set = set(c_ids)
                bonus = np.zeros(len(c_ids), dtype=np.float32)
                for entry in r._memory._entries:
                    cid = entry["chunk_id"]
                    if cid not in c_ids_set:
                        continue
                    q_mem = entry["query_emb"]
                    q_mem_norm = q_mem / (np.linalg.norm(q_mem) + 1e-8)
                    q_sim = max(0.0, float(qn_vec @ q_mem_norm))
                    _eff = (np.exp(-r._memory.lam * entry["age"])
                            * entry["w_cf"] * q_sim)
                    bonus[c_ids.index(cid)] += _eff
            t["boost"].append((time.perf_counter() - t0) * 1e3)

            t["total_added"].append(
                t["attend"][-1] + t["head"][-1] + t["boost"][-1])

    print("\nComponent latency (per query):")
    for k in ["encode", "faiss", "attend", "head", "boost", "total_added"]:
        print(f"  {k:12s} {stat(t[k])}")
    print("\nencode + faiss are shared with the dense baseline;")
    print("attend + head + boost = ITMA-added compute (= total_added).")

    import csv as _csv
    import os as _os
    out = "analysis/latency_components.csv"
    _os.makedirs(_os.path.dirname(out) or ".", exist_ok=True)
    shared = {"encode", "faiss"}
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["component", "kind", "mem_size", "mean_ms", "median_ms", "n"])
        for k in ["encode", "faiss", "attend", "head", "boost"]:
            kind = "shared" if k in shared else "itma_added"
            w.writerow([k, kind, actual, f"{statistics.mean(t[k]):.4f}",
                        f"{statistics.median(t[k]):.4f}", len(t[k])])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
