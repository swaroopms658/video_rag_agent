"""Compute H@1 and R@10 for ITMA at N=50, matching the cold_start_eval.py protocol.

The cold_start_eval uses --split test: it gives oracle feedback on 50 random
test items then evaluates all 59 test items. This script replicates that protocol
for H@1 and R@10 (which cold_start_eval.py doesn't compute).
"""
import json, random, statistics
from sentence_transformers import SentenceTransformer
from src.itma.integration import ITMARetriever
from src.eval_utils import hit_at_k, recall_at_k

embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

with open("data/lecture_rag_75/qa.jsonl", encoding="utf-8") as f:
    all_items = [json.loads(l) for l in f if l.strip()]
with open("data/lecture_rag_75/splits.json", encoding="utf-8") as f:
    splits = json.load(f)
test_ids = set(splits["test"])
test_items = [it for it in all_items if it.get("id") in test_ids and it.get("gold_context_ids")]


def run_seed(seed):
    r = ITMARetriever(
        store_path="data/lecture_rag_75/combined",
        checkpoint="checkpoints/itma_head.pt",
        memory_path=None,
    )
    rng = random.Random(seed)
    order = list(range(len(test_items)))
    rng.shuffle(order)

    n_feedback = 0
    for idx in order:
        item = test_items[idx]
        gold = item.get("gold_context_ids") or []
        if not gold:
            continue
        r.retrieve_with_ids(item["question"], embedder, top_k=10)
        r.record_feedback(helpful_chunk_ids=gold, reward=1.0)
        n_feedback += 1
        if n_feedback >= 50:
            break

    # Evaluate all test items
    h1_scores, h5_scores, r10_scores = [], [], []
    for item in test_items:
        gold = item.get("gold_context_ids") or []
        if not gold:
            continue
        res = r.retrieve_with_ids(item["question"], embedder, top_k=10)
        retrieved = [cid for _, _, cid in res]
        h1_scores.append(hit_at_k(retrieved, gold, k=1))
        h5_scores.append(hit_at_k(retrieved, gold, k=5))
        r10_scores.append(recall_at_k(retrieved, gold, k=10))

    return (statistics.mean(h1_scores), statistics.mean(h5_scores),
            statistics.mean(r10_scores))


results = []
for s in range(5):
    print(f"  seed {s}...", flush=True)
    h1, h5, r10 = run_seed(s)
    print(f"    H@1={h1:.4f}  H@5={h5:.4f}  R@10={r10:.4f}")
    results.append((h1, h5, r10))

h1_mean = statistics.mean(r[0] for r in results)
h5_mean = statistics.mean(r[1] for r in results)
r10_mean = statistics.mean(r[2] for r in results)
print(f"\nFINAL (5 seeds): H@1={h1_mean:.4f}  H@5={h5_mean:.4f}  R@10={r10_mean:.4f}")
print("H@5 sanity check (should match cold_start.csv ~0.9322):", round(h5_mean, 4))
