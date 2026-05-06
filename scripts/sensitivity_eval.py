"""Sensitivity analysis: sweep λ (freshness decay) × η (CF learning rate).

Runs cold-start eval for each (λ, η) combination at the N=50 checkpoint only
(most informative) with 3 seeds. Produces a CSV and a heatmap PDF.

Usage:
    python scripts/sensitivity_eval.py
    python scripts/sensitivity_eval.py --n-feedback 50 --seeds 0 1 2
"""

import argparse
import csv
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.cold_start_eval import load_test_data, run_cold_start, build_retriever

LAMBDA_VALUES = [0.01, 0.05, 0.20]
ETA_VALUES    = [0.01, 0.05, 0.20]
DEFAULT_CHECKPOINT = "checkpoints/itma_head.pt"
QA_PATH     = "data/lecture_rag_75/qa.jsonl"
SPLITS_PATH = "data/lecture_rag_75/splits.json"
STORE_PATH  = "data/lecture_rag_75/combined"
OUT_CSV     = "analysis/sensitivity.csv"
OUT_FIG     = "analysis/figures/sensitivity_heatmap.pdf"


def run_sweep(qa_path, splits_path, store_path, checkpoint,
              lam_values, eta_values, seeds, target_n, out_csv):
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    items = load_test_data(qa_path, splits_path, split="all")
    print(f"Loaded {len(items)} items")

    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)
    rows = []

    combos = list(itertools.product(lam_values, eta_values))
    for lam, eta in combos:
        sys_name = f"itma_l{lam}_e{eta}"
        print(f"\n{sys_name}  (λ={lam}, η={eta})")
        for seed in seeds:
            retriever = build_retriever(sys_name, store_path, checkpoint)
            seed_rows = run_cold_start(
                system_name=sys_name,
                retriever=retriever,
                embed_model=embed_model,
                test_items=items,
                seed=seed,
                checkpoints=[0, target_n],
            )
            for r in seed_rows:
                if r["n_feedback"] == target_n:
                    rows.append({"lam": lam, "eta": eta, "seed": seed,
                                 "hit_at_5": r["hit_at_5"],
                                 "mrr_score": r["mrr_score"],
                                 "ndcg_at_10": r["ndcg_at_10"]})
                    print(f"  seed={seed}  H@5={r['hit_at_5']:.3f}  MRR={r['mrr_score']:.3f}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lam", "eta", "seed",
                                               "hit_at_5", "mrr_score", "ndcg_at_10"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out_csv}")
    return rows


def plot_heatmap(csv_path, metric, out_path, lam_values, eta_values):
    import numpy as np
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib to generate heatmap")
        return

    data = {(lam, eta): [] for lam, eta in itertools.product(lam_values, eta_values)}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (float(row["lam"]), float(row["eta"]))
            if key in data:
                data[key].append(float(row[metric]))

    grid = np.array([[np.mean(data[(lam, eta)]) if data[(lam, eta)] else float("nan")
                      for eta in eta_values] for lam in lam_values])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(grid, aspect="auto", cmap="YlGn",
                   vmin=np.nanmin(grid) - 0.01, vmax=np.nanmax(grid) + 0.01)
    plt.colorbar(im, ax=ax, label=metric)
    ax.set_xticks(range(len(eta_values)))
    ax.set_yticks(range(len(lam_values)))
    ax.set_xticklabels([str(e) for e in eta_values])
    ax.set_yticklabels([str(l) for l in lam_values])
    ax.set_xlabel("η (counterfactual learning rate)", fontsize=10)
    ax.set_ylabel("λ (freshness decay)", fontsize=10)
    ax.set_title(f"ITMA sensitivity — {metric} at N=50", fontsize=11)

    for i, lam in enumerate(lam_values):
        for j, eta in enumerate(eta_values):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9,
                        color="black" if val < np.nanmax(grid) - 0.01 else "white")

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa",         default=QA_PATH)
    parser.add_argument("--splits",     default=SPLITS_PATH)
    parser.add_argument("--store",      default=STORE_PATH)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--n-feedback", type=int, default=50)
    parser.add_argument("--seeds",      nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--lam",        nargs="+", type=float, default=LAMBDA_VALUES)
    parser.add_argument("--eta",        nargs="+", type=float, default=ETA_VALUES)
    parser.add_argument("--out-csv",    default=OUT_CSV)
    parser.add_argument("--out-fig",    default=OUT_FIG)
    parser.add_argument("--metric",     default="hit_at_5")
    args = parser.parse_args()

    rows = run_sweep(args.qa, args.splits, args.store, args.checkpoint,
                     args.lam, args.eta, args.seeds, args.n_feedback, args.out_csv)
    if rows:
        plot_heatmap(args.out_csv, args.metric, args.out_fig, args.lam, args.eta)


if __name__ == "__main__":
    main()
