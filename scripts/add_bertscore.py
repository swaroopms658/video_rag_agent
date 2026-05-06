"""Add BERTScore column to gen eval CSVs that are missing it (--no-bertscore runs).

Usage:
    python scripts/add_bertscore.py [--systems bm25 cross_encoder cfrag_lite itma]
"""
import argparse, csv
from pathlib import Path
from src.eval_utils import bertscore_f1

RESULTS_DIR = Path("analysis/results_test")

parser = argparse.ArgumentParser()
parser.add_argument("--systems", nargs="+",
                    default=["bm25", "cross_encoder", "cfrag_lite", "itma"])
args = parser.parse_args()

for sys in args.systems:
    p = RESULTS_DIR / f"{sys}.csv"
    if not p.exists() or p.stat().st_size == 0:
        print(f"  [skip] {sys} — file missing")
        continue
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    if not rows:
        print(f"  [skip] {sys} — no rows")
        continue
    missing = [r for r in rows if r.get("bertscore_f1", "").strip() in ("", "None")]
    if not missing:
        print(f"  [skip] {sys} — BERTScore already present in all rows")
        continue

    print(f"  Computing BERTScore for {sys} ({len(rows)} rows)...")
    hypotheses = [r.get("generated_answer", "") or "" for r in rows]
    references = [r.get("gold_answer", "") or "" for r in rows]
    scores = bertscore_f1(hypotheses, references)

    fieldnames = list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, score in zip(rows, scores):
            row["bertscore_f1"] = round(score, 4)
            writer.writerow(row)
    print(f"    Done. Mean BS-F1 = {sum(scores)/len(scores):.4f}")
