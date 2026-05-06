"""Build a 100-query MS MARCO passage dev slice for external validity evaluation.

Uses HuggingFace `datasets` in streaming mode — no need to download the full corpus.
Saves two files:
  data/ms_marco_slice/queries.jsonl    — 100 queries with query_id, query text
  data/ms_marco_slice/passages.jsonl  — relevant passages for those queries

Usage:
    python scripts/build_ms_marco_slice.py
    python scripts/build_ms_marco_slice.py --n 100 --seed 42 --out data/ms_marco_slice

Requirements:
    pip install datasets
"""

import argparse
import json
import os
import random


OUT_DIR = "data/ms_marco_slice"
N_QUERIES = 100
SEED = 42


def build_slice(n=N_QUERIES, seed=SEED, out_dir=OUT_DIR):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets")
        return

    os.makedirs(out_dir, exist_ok=True)
    queries_path = os.path.join(out_dir, "queries.jsonl")
    passages_path = os.path.join(out_dir, "passages.jsonl")

    if os.path.exists(queries_path) and os.path.exists(passages_path):
        print(f"Slice already exists at {out_dir}. Delete to rebuild.")
        return

    print(f"Loading MS MARCO v1.1 validation split (streaming)...")
    # ms_marco v1.1 has 'query', 'passages' (list of {'passage_text', 'is_selected', 'url'})
    ds = load_dataset("ms_marco", "v1.1", split="validation", streaming=True, trust_remote_code=True)

    rng = random.Random(seed)
    reservoir = []

    print("Sampling with reservoir sampling (seed={})...".format(seed))
    for i, example in enumerate(ds):
        # Only keep examples that have at least one selected (relevant) passage
        selected = [p for p in example.get("passages", {}).get("passage_text", [])
                    if example.get("passages", {}).get("is_selected", [0]*len(
                        example["passages"]["passage_text"]))[
                        example["passages"]["passage_text"].index(p)] == 1]
        if not selected:
            continue

        if len(reservoir) < n:
            reservoir.append((example, selected))
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = (example, selected)

        if i > 0 and i % 500 == 0:
            print(f"  Scanned {i} examples, reservoir={len(reservoir)}")

        if i > 10_000:  # cap scan to avoid downloading too much
            break

    print(f"Selected {len(reservoir)} examples from MS MARCO dev.")

    with open(queries_path, "w", encoding="utf-8") as qf, \
         open(passages_path, "w", encoding="utf-8") as pf:
        for example, selected_passages in reservoir:
            qid = str(example.get("query_id", ""))
            query = example.get("query", "").strip()

            qf.write(json.dumps({"query_id": qid, "query": query}, ensure_ascii=False) + "\n")

            for ptext in selected_passages:
                pf.write(json.dumps({
                    "query_id": qid,
                    "passage": ptext.strip(),
                    "is_selected": 1,
                }, ensure_ascii=False) + "\n")

    print(f"Saved {len(reservoir)} queries -> {queries_path}")
    print(f"Saved relevant passages -> {passages_path}")
    print(f"\nNext step: build a FAISS index from these passages, then run baseline eval on the slice.")


def main():
    parser = argparse.ArgumentParser(description="Build MS MARCO dev slice")
    parser.add_argument("--n", type=int, default=N_QUERIES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    build_slice(n=args.n, seed=args.seed, out_dir=args.out)


if __name__ == "__main__":
    main()
