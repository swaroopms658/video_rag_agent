"""Generate LaTeX result tables from per-system CSV files in analysis/results/.

Usage:
    python analysis/make_tables.py                   # all systems, main table
    python analysis/make_tables.py --systems bm25 dense_minilm itma
    python analysis/make_tables.py --split test      # only test-split rows
    python analysis/make_tables.py --out paper/tables/main_results.tex

Expected CSV columns (from evaluate.py):
    question, gold_answer, generated_answer,
    hit_at_1, hit_at_5, mrr_score, ndcg_at_10, recall_at_10,
    bertscore_f1, rouge_l, bleu_4, faithfulness
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

RESULTS_DIR = "analysis/results_test"
METRICS = [
    ("hit_at_1",     "H@1",     ".3f"),
    ("hit_at_5",     "H@5",     ".3f"),
    ("mrr_score",    "MRR@10",  ".3f"),
    ("ndcg_at_10",   "nDCG@10", ".3f"),
    ("recall_at_10", "R@10",    ".3f"),
    ("bertscore_f1", "BS-F1",   ".3f"),
    ("rouge_l",      "ROUGE-L", ".3f"),
    ("bleu_4",       "BLEU-4",  ".3f"),
]

SYSTEM_ORDER = [
    "bm25", "dense_minilm", "dense_mpnet", "cross_encoder",
    "static_memory", "cfrag_lite", "itma", "itma_cross",
]


def load_system_scores(csv_path: str) -> dict[str, list[float]]:
    """Load all numeric metric columns from a CSV. Returns dict metric → [values]."""
    scores: dict[str, list[float]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col, _, _ in METRICS:
                val = row.get(col, "")
                if val and val.strip() not in ("", "None", "null"):
                    try:
                        scores[col].append(float(val))
                    except ValueError:
                        pass
    return dict(scores)


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    """Returns (mean, lower_95ci, upper_95ci) via bootstrap."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0], values[0]
    try:
        from analysis.stats import bootstrap_ci
        return bootstrap_ci(values)
    except Exception:
        a = np.array(values)
        m = float(np.mean(a))
        se = float(np.std(a, ddof=1) / np.sqrt(len(a)))
        return m, m - 1.96 * se, m + 1.96 * se


def make_main_table(
    results_dir: str = RESULTS_DIR,
    systems: Optional[list[str]] = None,
    out_path: Optional[str] = None,
    bold_best: bool = True,
) -> str:
    """Generate the main results LaTeX table. Returns the LaTeX string."""
    csv_dir = Path(results_dir)
    csv_files = {p.stem: p for p in csv_dir.glob("*.csv")}

    if systems is None:
        # Use canonical order, fall back to alphabetical for unknown systems
        known = [s for s in SYSTEM_ORDER if s in csv_files]
        unknown = sorted(s for s in csv_files if s not in SYSTEM_ORDER)
        systems = known + unknown
    else:
        systems = [s for s in systems if s in csv_files]

    if not systems:
        return "% No CSV files found in " + results_dir

    # Collect means per system per metric
    data: dict[str, dict[str, tuple]] = {}  # system → metric → (mean, lo, hi)
    for sys_name in systems:
        scores = load_system_scores(str(csv_files[sys_name]))
        data[sys_name] = {}
        for col, _, _ in METRICS:
            vals = scores.get(col, [])
            data[sys_name][col] = mean_ci(vals)

    # Find best per column for bolding
    best: dict[str, float] = {}
    if bold_best:
        for col, _, _ in METRICS:
            vals = [data[s][col][0] for s in systems if not np.isnan(data[s][col][0])]
            best[col] = max(vals) if vals else float("nan")

    # Build LaTeX
    n_metrics = len(METRICS)
    col_spec = "l" + "r" * n_metrics
    header_row = " & ".join(["System"] + [lab for _, lab, _ in METRICS]) + r" \\"
    midrule_after = ["static_memory"]  # draw \\midrule after these

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main results on LectureRAG-75 test set. Best result per column in \textbf{bold}.}",
        r"\label{tab:main_results}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header_row,
        r"\midrule",
    ]

    for sys_name in systems:
        cells = [sys_name.replace("_", r"\_")]
        for col, _, fmt in METRICS:
            mean_val, lo, hi = data[sys_name][col]
            if np.isnan(mean_val):
                cells.append("---")
            else:
                s = format(mean_val, fmt)
                if bold_best and abs(mean_val - best.get(col, float("nan"))) < 1e-9:
                    s = r"\textbf{" + s + "}"
                cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
        if sys_name in midrule_after:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",  # end resizebox
        r"\end{table}",
    ]

    table_str = "\n".join(lines)
    if out_path:
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(table_str)
        print(f"LaTeX table written -> {out_path}")
    return table_str


def make_ablation_table(
    results_dir: str = RESULTS_DIR,
    prefix: str = "itma_ablation_",
    out_path: Optional[str] = None,
) -> str:
    """Table comparing ITMA ablation variants."""
    csv_dir = Path(results_dir)
    ablation_systems = sorted(p.stem for p in csv_dir.glob(f"{prefix}*.csv"))
    if not ablation_systems:
        return "% No ablation CSVs found"
    return make_main_table(results_dir=results_dir, systems=ablation_systems, out_path=out_path)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX result tables")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--systems", nargs="*", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    if args.ablation:
        table = make_ablation_table(results_dir=args.results_dir, out_path=args.out)
    else:
        table = make_main_table(
            results_dir=args.results_dir,
            systems=args.systems,
            out_path=args.out,
        )
    if not args.out:
        print(table)


if __name__ == "__main__":
    main()
