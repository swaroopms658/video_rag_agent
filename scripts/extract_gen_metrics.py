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
REQUIRED_COLS = ["rouge_l", "bleu_4"]          # must be present to include a system
OPTIONAL_COLS = ["bertscore_f1", "faithfulness"]  # included if available


def read_gen_metrics(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out = {}
    all_cols = REQUIRED_COLS + OPTIONAL_COLS
    for col in all_cols:
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
        if any(metrics[c][0] is None for c in REQUIRED_COLS):
            print(f"  [skip] {sys} — missing required gen cols (n={n})")
            continue
        rl = metrics["rouge_l"][0]
        b4 = metrics["bleu_4"][0]
        bs = metrics["bertscore_f1"][0]   # may be None
        fa = metrics["faithfulness"][0]   # may be None
        n_gen = metrics["rouge_l"][1]
        rows_out.append((sys, n, n_gen, bs, rl, b4, fa))
        bs_s = f"{bs:.4f}" if bs is not None else "n/a"
        fa_s = f"{fa:.4f}" if fa is not None else "n/a"
        print(f"  {sys:16s}  n={n_gen:2d}  BS-F1={bs_s}  ROUGE-L={rl:.4f}  BLEU-4={b4:.4f}  Faith={fa_s}")

    if rows_out:
        print("\n--- Markdown table ---")
        print("| System | n | BS-F1 | ROUGE-L | BLEU-4 | Faithfulness |")
        print("|---|---|---|---|---|---|")
        for sys, n, n_gen, bs, rl, b4, fa in rows_out:
            name = DISPLAY.get(sys, sys)
            bs_s = f"{bs:.4f}" if bs is not None else "—"
            fa_s = f"{fa:.4f}" if fa is not None else "—"
            print(f"| {name} | {n_gen} | {bs_s} | {rl:.4f} | {b4:.4f} | {fa_s} |")
    else:
        print("No completed generation evals found.")


if __name__ == "__main__":
    main()
