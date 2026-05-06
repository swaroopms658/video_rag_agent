"""Add faithfulness column to gen eval CSVs using stored retrieved_contexts.

Requires retrieved_contexts column (JSON list) written by evaluate.py.
Uses the Groq API via AgenticRAG — set GROQ_API_KEY env var first.

Usage:
    python scripts/add_faithfulness.py [--systems bm25 cross_encoder cfrag_lite itma]
"""
import argparse
import csv
import json
from pathlib import Path

from src.eval_utils import calculate_faithfulness

RESULTS_DIR = Path("analysis/results_test")

parser = argparse.ArgumentParser()
parser.add_argument("--systems", nargs="+",
                    default=["bm25", "cross_encoder", "cfrag_lite", "itma"])
args = parser.parse_args()

from src.rag_chain import AgenticRAG
_rag = AgenticRAG("data/lecture_rag_75/combined")

for sys in args.systems:
    p = RESULTS_DIR / f"{sys}.csv"
    if not p.exists() or p.stat().st_size == 0:
        print(f"  [skip] {sys} — file missing")
        continue
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    if not rows:
        print(f"  [skip] {sys} — no rows")
        continue
    if "retrieved_contexts" not in rows[0]:
        print(f"  [skip] {sys} — no retrieved_contexts column (re-run evaluate.py)")
        continue
    missing = [r for r in rows if r.get("faithfulness", "").strip() in ("", "None")]
    if not missing:
        print(f"  [skip] {sys} — faithfulness already present in all rows")
        continue

    print(f"  Computing faithfulness for {sys} ({len(missing)} rows)...")
    scores_map = {}
    for i, r in enumerate(rows):
        if r.get("faithfulness", "").strip() not in ("", "None"):
            continue
        generated_answer = r.get("generated_answer", "") or ""
        contexts_raw = r.get("retrieved_contexts", "[]")
        try:
            contexts = json.loads(contexts_raw)
        except Exception:
            contexts = [contexts_raw]
        if not generated_answer or not contexts:
            scores_map[r["question"]] = None
            continue
        faith = calculate_faithfulness(_rag.groq_generate, generated_answer, contexts)
        scores_map[r["question"]] = faith
        print(f"    [{i+1}/{len(rows)}] {r['question'][:50]}... → {faith}")

    fieldnames = list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row["question"] in scores_map:
                row["faithfulness"] = scores_map[row["question"]]
            writer.writerow(row)

    computed = [v for v in scores_map.values() if v is not None]
    mean_f = sum(computed) / len(computed) if computed else float("nan")
    print(f"    Done. Mean Faithfulness = {mean_f:.4f} ({len(computed)} computed)")
