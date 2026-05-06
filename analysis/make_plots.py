"""Generate figures for the ITMA paper.

Figures produced:
  1. cold_start_curve.pdf  — Hit@5 (or MRR@10) vs. number of feedback examples
                              (the primary result figure, Figure 1 of the paper)
  2. domain_bars.pdf        — Per-domain Hit@5 grouped bar chart
  3. ablation_heatmap.pdf   — Ablation grid (λ vs η, or K vs λ)

Usage:
    python analysis/make_plots.py                     # all figures
    python analysis/make_plots.py --figure cold_start
    python analysis/make_plots.py --figure domain_bars
    python analysis/make_plots.py --out paper/figures

Cold-start CSV format (from cold_start_eval.py):
    columns: system, n_feedback, hit_at_5, mrr_score (one row per evaluation point)
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

RESULTS_DIR = "analysis/results_test"
COLD_START_CSV = "analysis/cold_start.csv"
OUT_DIR = "analysis/figures"

# System display names and colours
SYSTEM_STYLE = {
    "dense_minilm":    ("Dense-MiniLM",          "#1f77b4", "-"),
    "static_memory":   ("Static τ+β",             "#ff7f0e", "--"),
    "cfrag_lite":      ("CFRAG-lite",             "#d62728", "-."),
    "itma":            ("ITMA (head+boost)",       "#2ca02c", "-"),
    "itma_no_boost":   ("ITMA no-boost (head)",   "#9467bd", ":"),
    "itma_boost_only": ("ITMA boost-only",        "#8c564b", "--"),
}


def _import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("pip install matplotlib to generate figures")


def plot_cold_start_curve(
    csv_path: str = COLD_START_CSV,
    metric: str = "hit_at_5",
    out_path: Optional[str] = None,
    systems: Optional[list[str]] = None,
):
    """Figure 1: metric vs. N feedback examples for each system."""
    plt = _import_mpl()

    # Load data: {system: {n: [values]}}
    data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sys = row.get("system", "")
            try:
                n = int(row.get("n_feedback", 0))
                val = float(row.get(metric, "nan"))
            except (ValueError, TypeError):
                continue
            if systems and sys not in systems:
                continue
            data[sys][n].append(val)

    if not data:
        print(f"No data found in {csv_path}")
        return

    import numpy as np
    fig, ax = plt.subplots(figsize=(6, 4))

    plot_systems = systems or list(SYSTEM_STYLE.keys())
    for sys_name in plot_systems:
        if sys_name not in data:
            continue
        label, color, ls = SYSTEM_STYLE.get(sys_name, (sys_name, None, "-"))
        ns = sorted(data[sys_name].keys())
        means = [np.mean(data[sys_name][n]) for n in ns]
        stds = [np.std(data[sys_name][n]) / max(1, len(data[sys_name][n]) ** 0.5) for n in ns]
        ax.plot(ns, means, label=label, color=color, linestyle=ls, linewidth=2, marker="o", ms=5)
        ax.fill_between(
            ns,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.12, color=color,
        )

    metric_label = {
        "hit_at_5": "Hit@5",
        "mrr_score": "MRR@10",
        "ndcg_at_10": "nDCG@10",
    }.get(metric, metric)

    ax.set_xlabel("Number of feedback examples", fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title("Cold-start adaptation curve", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    fig.tight_layout()

    if out_path is None:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, "cold_start_curve.pdf")
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cold-start curve -> {out_path}")


def plot_domain_bars(
    results_dir: str = RESULTS_DIR,
    metric: str = "hit_at_5",
    systems: Optional[list[str]] = None,
    out_path: Optional[str] = None,
):
    """Grouped bar chart: per-domain metric for each system."""
    plt = _import_mpl()
    import numpy as np

    csv_dir = Path(results_dir)
    sys_names = systems or list(SYSTEM_STYLE.keys())
    # {system: {domain: [values]}}
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for sys in sys_names:
        fpath = csv_dir / f"{sys}.csv"
        if not fpath.exists():
            continue
        with open(fpath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dom = row.get("domain", "unknown")
                try:
                    val = float(row.get(metric, "nan"))
                except (ValueError, TypeError):
                    continue
                data[sys][dom].append(val)

    all_domains = sorted({d for sys in data.values() for d in sys.keys()})
    if not all_domains or not data:
        print("No per-domain data found")
        return

    n_sys = len([s for s in sys_names if s in data])
    n_dom = len(all_domains)
    x = np.arange(n_dom)
    width = 0.8 / max(n_sys, 1)

    fig, ax = plt.subplots(figsize=(max(6, n_dom * 1.5), 4))
    for i, sys in enumerate([s for s in sys_names if s in data]):
        label, color, _ = SYSTEM_STYLE.get(sys, (sys, None, "-"))
        means = [np.mean(data[sys].get(d, [0.0])) for d in all_domains]
        ax.bar(x + i * width - (n_sys - 1) * width / 2, means, width, label=label, color=color)

    metric_label = {
        "hit_at_5": "Hit@5", "mrr_score": "MRR@10",
    }.get(metric, metric)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in all_domains], fontsize=9)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title("Per-domain retrieval performance", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if out_path is None:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, "domain_bars.pdf")
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved domain bars -> {out_path}")


def plot_ablation_curve(
    csv_path: str = "analysis/ablation_cold_start.csv",
    metric: str = "hit_at_5",
    out_path: Optional[str] = None,
):
    """Figure 2: ablation cold-start curve (itma vs itma_no_boost vs itma_boost_only)."""
    ABLATION_STYLE = {
        "itma":            ("ITMA (head+boost)",      "#2ca02c", "-"),
        "itma_no_boost":   ("ITMA no-boost (head)",   "#9467bd", ":"),
        "itma_boost_only": ("ITMA boost-only",        "#8c564b", "--"),
    }
    plot_cold_start_curve(
        csv_path=csv_path,
        metric=metric,
        out_path=out_path or "analysis/figures/ablation_curve.pdf",
        systems=list(ABLATION_STYLE.keys()),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--figure", choices=["cold_start", "domain_bars", "ablation", "all"],
                        default="all")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--cold-start-csv", default=COLD_START_CSV)
    parser.add_argument("--metric", default="hit_at_5")
    parser.add_argument("--out", default=None)
    parser.add_argument("--systems", nargs="*", default=None)
    args = parser.parse_args()

    if args.figure in ("ablation", "all"):
        ablation_csv = "analysis/ablation_cold_start.csv"
        if os.path.exists(ablation_csv):
            out = os.path.join(args.out, "ablation_curve.pdf") if args.out else None
            plot_ablation_curve(csv_path=ablation_csv, metric=args.metric, out_path=out)
        else:
            print(f"[skip] ablation_cold_start.csv not found")

    if args.figure in ("cold_start", "all"):
        if os.path.exists(args.cold_start_csv):
            out = os.path.join(args.out, "cold_start_curve.pdf") if args.out else None
            plot_cold_start_curve(
                csv_path=args.cold_start_csv,
                metric=args.metric,
                out_path=out,
                systems=args.systems,
            )
        else:
            print(f"[skip] cold_start.csv not found at {args.cold_start_csv}")

    if args.figure in ("domain_bars", "all"):
        out = os.path.join(args.out, "domain_bars.pdf") if args.out else None
        plot_domain_bars(
            results_dir=args.results_dir,
            metric=args.metric,
            systems=args.systems,
            out_path=out,
        )


if __name__ == "__main__":
    main()
