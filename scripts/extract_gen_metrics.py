"""Extract generation quality metrics from analysis/results_test/*.csv and print a summary table.

Run after generation eval is complete:
    python scripts/extract_gen_metrics.py
"""
import csv
import statistics
from pathlib import Path

RESULTS_DIR = Path("analysis/results_test")
SYSTEMS = ["bm25", "dense_minilm", "cross_encoder", "cfrag_lite", "itma"]
DISPLAY = {
    "bm25": "BM25",
    "dense_minilm": "Dense-MiniLM",
    "cross_encoder": "Cross-Encoder",
    "cfrag_lite": "CFRAG-lite†",
    "itma": "ITMA (N=0)",
}
GEN_COLS = ["bertscore_f1", "rouge_l", "bleu_4", "faithfulness"]


def read_gen_metrics(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out = {}
    for col in GEN_COLS:
        vals = [float(r[col]) for r in rows if r.get(col, "").strip() not in ("", "None")]
        out[col] = (statistics.mean(vals), len(vals)) if vals else (None, 0)
    return out, len(rows)


def main():
    rows_out = []
    for sys in SYSTEMS:
        p = RESULTS_DIR / f"{sys}.csv"
        if not p.exists() or p.stat().st_size == 0:
            print(f"  [skip] {sys} — file missing or empty")
            continue
        metrics, n = read_gen_metrics(p)
        gen_cols_present = all(metrics[c][0] is not None for c in GEN_COLS)
        if not gen_cols_present:
            print(f"  [skip] {sys} — no generation columns (retrieval-only CSV, n={n})")
            continue
        bs = metrics["bertscore_f1"][0]
        rl = metrics["rouge_l"][0]
        b4 = metrics["bleu_4"][0]
        fa = metrics["faithfulness"][0]
        n_gen = metrics["bertscore_f1"][1]
        rows_out.append((sys, n, n_gen, bs, rl, b4, fa))
        print(f"  {sys:16s}  n={n_gen:2d}  BS-F1={bs:.4f}  ROUGE-L={rl:.4f}  BLEU-4={b4:.4f}  Faith={fa:.4f}")

    if rows_out:
        print("\n--- Markdown table ---")
        print("| System | n | BS-F1 | ROUGE-L | BLEU-4 | Faithfulness |")
        print("|---|---|---|---|---|---|")
        for sys, n, n_gen, bs, rl, b4, fa in rows_out:
            name = DISPLAY.get(sys, sys)
            print(f"| {name} | {n_gen} | {bs:.4f} | {rl:.4f} | {b4:.4f} | {fa:.4f} |")
    else:
        print("No completed generation evals found.")


if __name__ == "__main__":
    main()
