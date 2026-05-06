"""Fast retrieval-only evaluation across all 115 QA items.

No Groq calls, no generation, no sleep. Only retrieval metrics:
Hit@1, Hit@5, MRR@10, nDCG@10, R@10.

Usage:
    python scripts/eval_retrieval_only.py
    python scripts/eval_retrieval_only.py --split all --output analysis/results_115
"""

import argparse
import csv
import json
import os

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scripts.cold_start_eval import build_retriever, load_test_data
from src.eval_utils import hit_at_k, mrr, ndcg_at_k, recall_at_k

QA_PATH     = "data/lecture_rag_75/qa.jsonl"
SPLITS_PATH = "data/lecture_rag_75/splits.json"
STORE_PATH  = "data/lecture_rag_75/combined"
OUTPUT_DIR  = "analysis/results_115"
CHECKPOINT_ITMA   = "checkpoints/itma_head.pt"
CHECKPOINT_CFRAG  = "checkpoints/cfrag_lite"
TOP_K = 10

ALL_SYSTEMS = ["bm25", "dense_minilm", "dense_mpnet",
               "cross_encoder", "static_memory", "cfrag_lite", "itma"]


def evaluate_retrieval(system_name: str, retriever, items: list[dict],
                       output_dir: str, embed_model=None) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    fieldnames = ["question", "domain", "difficulty",
                  "hit_at_1", "hit_at_5", "mrr_score", "ndcg_at_10", "recall_at_10"]
    rows = []
    accum = {k: [] for k in fieldnames[3:]}

    for item in items:
        q      = item.get("question", "")
        gold   = item.get("gold_context_ids", [])
        domain = item.get("domain", "")
        diff   = item.get("difficulty", "")
        if not q or not gold:
            continue

        try:
            results = retriever.retrieve_with_ids(q, model=embed_model, top_k=TOP_K)
            retrieved_ids = [r[2] for r in results]
        except Exception as e:
            print(f"  Error: {e}")
            retrieved_ids = []

        h1  = hit_at_k(retrieved_ids, gold, 1)
        h5  = hit_at_k(retrieved_ids, gold, 5)
        mrr_val = mrr(retrieved_ids, gold)
        ndcg = ndcg_at_k(retrieved_ids, gold, 10)
        rec  = recall_at_k(retrieved_ids, gold, 10)

        row = {"question": q, "domain": domain, "difficulty": diff,
               "hit_at_1": h1, "hit_at_5": h5, "mrr_score": round(mrr_val, 4),
               "ndcg_at_10": round(ndcg, 4), "recall_at_10": round(rec, 4)}
        rows.append(row)
        for k in accum:
            accum[k].append(row[k])

    csv_path = os.path.join(output_dir, f"{system_name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    summary = {k: round(sum(v) / n, 4) if n else 0 for k, v in accum.items()}
    summary["n"] = n
    print(f"  {system_name} (n={n}): "
          + "  ".join(f"{k}={v:.3f}" for k, v in summary.items() if k != "n"))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa",       default=QA_PATH)
    parser.add_argument("--splits",   default=SPLITS_PATH)
    parser.add_argument("--store",    default=STORE_PATH)
    parser.add_argument("--output",   default=OUTPUT_DIR)
    parser.add_argument("--split",    default="all",
                        choices=["all", "train", "dev", "test"])
    parser.add_argument("--systems",  nargs="+", default=ALL_SYSTEMS)
    parser.add_argument("--checkpoint-itma",  default=CHECKPOINT_ITMA)
    parser.add_argument("--checkpoint-cfrag", default=CHECKPOINT_CFRAG)
    args = parser.parse_args()

    items = load_test_data(args.qa, args.splits, split=args.split)
    print(f"Loaded {len(items)} items (split={args.split})")

    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    summaries = {}
    for sys_name in args.systems:
        print(f"\nLoading {sys_name}...")
        ckpt = args.checkpoint_itma if sys_name == "itma" else (
               args.checkpoint_cfrag if sys_name == "cfrag_lite" else None)
        try:
            retriever = build_retriever(sys_name, args.store, checkpoint=ckpt)
            summaries[sys_name] = evaluate_retrieval(
                sys_name, retriever, items, args.output, embed_model=embed_model)
        except Exception as e:
            print(f"  FAILED: {e}")

    # Print consolidated table
    print("\n\n=== RETRIEVAL RESULTS (n={}) ===".format(
        next(iter(summaries.values()), {}).get("n", "?")))
    print(f"{'System':<16} {'H@1':>6} {'H@5':>6} {'MRR':>6} {'nDCG':>6} {'R@10':>6}")
    print("-" * 50)
    for sys, s in summaries.items():
        print(f"{sys:<16} {s['hit_at_1']:>6.3f} {s['hit_at_5']:>6.3f} "
              f"{s['mrr_score']:>6.3f} {s['ndcg_at_10']:>6.3f} {s['recall_at_10']:>6.3f}")


if __name__ == "__main__":
    main()
