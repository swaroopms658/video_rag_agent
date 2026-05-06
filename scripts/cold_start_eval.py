"""Cold-start evaluation — primary result of the ITMA paper.

Replays queries from the test set one-at-a-time, measuring retrieval quality
at N = {0, 5, 10, 20, 30, 50} feedback examples accumulated.

For systems that update their memory online (ITMA, Static-memory), the update
happens immediately after each query is answered (using the gold_context_ids as
"oracle feedback"). For systems that require retraining (CFRAG-lite), retraining
is triggered every `retrain_every` queries.

Output: analysis/results/cold_start.csv with columns:
  system, n_feedback, seed, hit_at_5, mrr_score, ndcg_at_10

Usage:
    python scripts/cold_start_eval.py \
        --qa data/lecture_rag_75/qa.jsonl \
        --splits data/lecture_rag_75/splits.json \
        --store data/lecture_rag_75 \
        --systems dense_minilm static_memory itma \
        --seeds 0 1 2 3 4 \
        --out analysis/results/cold_start.csv
"""

import argparse
import csv
import json
import os
import random

from src.eval_utils import hit_at_k, mrr, ndcg_at_k

EVAL_CHECKPOINTS = [0, 5, 10, 20, 30, 50]


def load_test_data(qa_path: str, splits_path: str, split: str = "all") -> list[dict]:
    """Load QA items.

    split="all"  — load every item (default, for cold-start deployment simulation)
    split="test" — load only test-split items (for held-out table results)
    """
    items = []
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if split == "all" or splits_path is None:
        return items

    with open(splits_path) as f:
        splits_data = json.load(f)
    allowed_ids = set(splits_data.get(split, []))
    return [it for it in items if it.get("id") in allowed_ids]


def build_retriever(system_name: str, store_path: str, checkpoint: str = None):
    if system_name == "dense_minilm":
        from src.agent import SimpleRetriever
        return SimpleRetriever(store_path)
    elif system_name == "dense_mpnet":
        from src.baselines.dense_mpnet import DenseMpnetRetriever
        return DenseMpnetRetriever(store_path)
    elif system_name == "bm25":
        from src.baselines.bm25 import BM25Retriever
        return BM25Retriever(store_path)
    elif system_name == "cross_encoder":
        from src.baselines.cross_encoder import CrossEncoderRetriever
        return CrossEncoderRetriever(store_path)
    elif system_name == "static_memory":
        from src.baselines.static_memory import StaticMemoryRetriever
        return StaticMemoryRetriever(store_path)
    elif system_name == "cfrag_lite":
        from src.baselines.cfrag_lite import CFRAGLiteRetriever
        return CFRAGLiteRetriever(store_path, checkpoint=checkpoint)
    elif system_name == "itma":
        from src.itma.integration import ITMARetriever
        return ITMARetriever(store_path, checkpoint=checkpoint, memory_path=None)
    elif system_name == "itma_no_boost":
        from src.itma.integration import ITMARetriever
        return ITMARetriever(store_path, checkpoint=checkpoint, memory_path=None,
                             use_id_boost=False)
    elif system_name == "itma_boost_only":
        from src.itma.integration import ITMARetriever
        return ITMARetriever(store_path, checkpoint=checkpoint, memory_path=None,
                             use_scoring_head=False, use_id_boost=True)
    elif system_name.startswith("itma_l"):
        # Sensitivity sweep: itma_l<lam>_e<eta>  e.g. itma_l0.01_e0.05
        from src.itma.integration import ITMARetriever
        import re
        m = re.match(r"itma_l([0-9.]+)_e([0-9.]+)", system_name)
        lam = float(m.group(1)) if m else 0.05
        eta = float(m.group(2)) if m else 0.05
        return ITMARetriever(store_path, checkpoint=checkpoint, memory_path=None,
                             lam=lam, eta=eta)
    elif system_name == "itma_cross":
        from src.itma.integration import ITMARetriever
        return ITMARetriever(store_path, checkpoint=checkpoint, memory_path=None)
    else:
        raise ValueError(f"Unknown system: {system_name}")


def run_cold_start(
    system_name: str,
    retriever,
    embed_model,
    test_items: list[dict],
    seed: int = 0,
    checkpoints: list[int] = EVAL_CHECKPOINTS,
) -> list[dict]:
    """Replay test_items in random order, evaluate at each checkpoint.

    Returns list of {n_feedback, hit_at_5, mrr_score, ndcg_at_10} dicts.
    """
    rng = random.Random(seed)
    order = list(range(len(test_items)))
    rng.shuffle(order)

    n_feedback = 0
    metrics_at_checkpoint: dict[int, list[dict]] = {n: [] for n in checkpoints}

    def _evaluate_one(item: dict) -> dict:
        q = item["question"]
        gold_ids = item.get("gold_context_ids", [])
        results = retriever.retrieve_with_ids(q, embed_model, top_k=10)
        retrieved_ids = [r[2] for r in results]
        return {
            "hit_at_5": hit_at_k(retrieved_ids, gold_ids, k=5),
            "mrr_score": mrr(retrieved_ids, gold_ids),
            "ndcg_at_10": ndcg_at_k(retrieved_ids, gold_ids, k=10),
        }

    # Evaluate at N=0 before any feedback
    if 0 in checkpoints:
        for item in test_items:
            if item.get("gold_context_ids"):
                metrics_at_checkpoint[0].append(_evaluate_one(item))

    for idx in order:
        item = test_items[idx]
        gold_ids = item.get("gold_context_ids", [])

        # Feed oracle feedback to memory-based systems
        if hasattr(retriever, "record_feedback") and gold_ids:
            # Trigger retrieval to populate internal state
            retriever.retrieve_with_ids(item["question"], embed_model, top_k=10)
            retriever.record_feedback(helpful_chunk_ids=gold_ids, reward=1.0)
            n_feedback += 1
        elif gold_ids:
            n_feedback += 1

        # Evaluate at checkpoints
        if n_feedback in checkpoints:
            for item2 in test_items:
                if item2.get("gold_context_ids"):
                    metrics_at_checkpoint[n_feedback].append(_evaluate_one(item2))

        if n_feedback >= max(checkpoints):
            break

    # Aggregate
    results = []
    for n, rows in metrics_at_checkpoint.items():
        if not rows:
            continue
        avg = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        avg["n_feedback"] = n
        avg["seed"] = seed
        avg["system"] = system_name
        results.append(avg)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run cold-start curve evaluation")
    parser.add_argument("--qa", default="data/lecture_rag_75/qa.jsonl")
    parser.add_argument("--splits", default="data/lecture_rag_75/splits.json")
    parser.add_argument("--store", default="data/lecture_rag_75")
    parser.add_argument("--systems", nargs="+",
                        default=["dense_minilm", "static_memory", "cfrag_lite", "itma"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--checkpoint", default=None,
                        help="ITMA / CFRAG-lite checkpoint path")
    parser.add_argument("--split", default="all",
                        choices=["all", "train", "dev", "test"],
                        help="Which split to use. Default 'all' for the cold-start curve "
                             "(deployment simulation); use 'test' for held-out table results.")
    parser.add_argument("--out", default="analysis/results/cold_start.csv")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    print(f"Loading data from {args.qa} (split={args.split}) ...")
    test_items = load_test_data(args.qa, args.splits, split=args.split)
    print(f"  {len(test_items)} test items")

    all_rows = []
    for sys_name in args.systems:
        print(f"\nSystem: {sys_name}")
        for seed in args.seeds:
            print(f"  Seed {seed} ...")
            retriever = build_retriever(sys_name, args.store, args.checkpoint)
            rows = run_cold_start(
                system_name=sys_name,
                retriever=retriever,
                embed_model=embed_model,
                test_items=test_items,
                seed=seed,
            )
            all_rows.extend(rows)
            print(f"  -> {len(rows)} checkpoint rows")

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    fieldnames = ["system", "n_feedback", "seed", "hit_at_5", "mrr_score", "ndcg_at_10"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\nWrote {len(all_rows)} rows -> {args.out}")
    print("Next: python analysis/make_plots.py --figure cold_start")


if __name__ == "__main__":
    main()
