"""Plot the ITMA latency component breakdown from analysis/latency_components.csv.

Horizontal bar chart of per-query latency by pipeline stage at |M|=50, coloured
by whether the stage is shared with any dense retriever (query encode, FAISS
search) or is ITMA-added compute (memory summary, head scoring, ID-boost).
Shows that query encoding dominates and ITMA's added re-ranking compute is a
small fraction of per-query latency.  Matches analysis/make_plots.py style.

Usage:
    python analysis/plot_latency.py \
        --csv analysis/latency_components.csv \
        --out paper/figures/fig5_latency.pdf
"""

import argparse
import csv
import os


def _import_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


LABELS = {
    "encode": "Query encode\n(MiniLM)",
    "faiss": "FAISS search",
    "attend": "Memory summary",
    "head": "Head scoring\n(20 cand.)",
    "boost": "ID-boost",
}
SHARED_COLOR = "#1f77b4"   # shared with any dense retriever
ITMA_COLOR = "#2ca02c"     # ITMA-added compute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="analysis/latency_components.csv")
    ap.add_argument("--out", default="paper/figures/fig5_latency.pdf")
    args = ap.parse_args()
    plt = _import_mpl()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    # order: encode, faiss, attend, head, boost (top-to-bottom -> reverse for barh)
    order = ["encode", "faiss", "attend", "head", "boost"]
    rows.sort(key=lambda r: order.index(r["component"]))
    mem = rows[0]["mem_size"]

    names = [LABELS[r["component"]] for r in rows]
    means = [float(r["mean_ms"]) for r in rows]
    colors = [SHARED_COLOR if r["kind"] == "shared" else ITMA_COLOR for r in rows]
    added = sum(float(r["mean_ms"]) for r in rows if r["kind"] == "itma_added")

    fig, ax = plt.subplots(figsize=(6, 4))
    y = range(len(names))
    bars = ax.barh(list(y), means, color=colors, edgecolor="black",
                   linewidth=0.6, alpha=0.9)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()  # encode on top
    ax.set_xlabel("Per-query latency (ms)", fontsize=11)
    ax.set_title(f"ITMA retrieval latency breakdown (CPU, $|\\mathcal{{M}}|$={mem})",
                 fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    for b, v in zip(bars, means):
        ax.text(b.get_width() + max(means) * 0.01,
                b.get_y() + b.get_height() / 2,
                f"{v:.2f} ms", va="center", fontsize=8)

    # legend via proxy handles
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=SHARED_COLOR, edgecolor="black",
              label="Shared with dense retriever"),
        Patch(facecolor=ITMA_COLOR, edgecolor="black",
              label=f"ITMA-added ({added:.2f} ms total)"),
    ], fontsize=8, loc="lower right")

    ax.set_xlim(0, max(means) * 1.18)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}  (ITMA-added compute = {added:.2f} ms)")


if __name__ == "__main__":
    main()
