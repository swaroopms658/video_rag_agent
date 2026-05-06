"""Generate LaTeX tables for the ITMA paper.

Usage:
    python analysis/make_paper_tables.py --out paper/tables/

Produces:
    table1_retrieval.tex   -- Table 1: retrieval metrics (includes ITMA N=50)
    table_gen.tex          -- Table X: generation quality
    table_ablation.tex     -- Table 3: ablation
    table_msmarco.tex      -- Table 2: MS-MARCO
"""
import argparse
import csv
import os
from pathlib import Path

RETRIEVAL_DIR = Path("analysis/results_test_retrieval_only")
GEN_DIR = Path("analysis/results_test")
MSMARCO_DIR = Path("analysis/ms_marco")

# Canonical system order and display names for Table 1
TABLE1_SYSTEMS = [
    ("bm25",           "BM25"),
    ("dense_minilm",   r"Dense-MiniLM"),
    ("dense_mpnet",    r"Dense-MPNet"),
    ("cross_encoder",  r"Cross-Encoder"),
    ("static_memory",  r"Static $\tau$+$\beta$"),
    ("cfrag_lite",     r"CFRAG-lite$^\dagger$"),
    ("itma_n0",        r"\textbf{ITMA (N=0)}"),
    ("itma_n50",       r"\textbf{ITMA (N=50) $\star$}"),
]

# Hard-coded numbers from results.md (verified, final — n=59 test split)
HARDCODED = {
    "bm25":           dict(h1=0.593, h5=0.848, mrr=0.692, ndcg=0.722, r10=0.864),
    "dense_minilm":   dict(h1=0.525, h5=0.848, mrr=0.638, ndcg=0.695, r10=0.898),
    "dense_mpnet":    dict(h1=0.475, h5=0.797, mrr=0.623, ndcg=0.692, r10=0.941),
    "cross_encoder":  dict(h1=0.644, h5=0.898, mrr=0.747, ndcg=0.789, r10=0.949),
    "static_memory":  dict(h1=0.525, h5=0.848, mrr=0.638, ndcg=0.695, r10=0.898),
    "cfrag_lite":     dict(h1=0.729, h5=0.915, mrr=0.811, ndcg=0.837, r10=0.949),
    "itma_n0":        dict(h1=0.508, h5=0.831, mrr=0.625, ndcg=0.688, r10=0.907),
    "itma_n50":       dict(h1=0.688, h5=0.932, mrr=0.790, ndcg=0.832, r10=0.951),
}

RETRIEVAL_COLS = ["hit_at_1", "hit_at_5", "mrr_score", "ndcg_at_10", "recall_at_10"]
RETRIEVAL_HEADERS = ["H@1", "H@5", "MRR@10", "nDCG@10", "R@10"]


def _load_retrieval(csv_path):
    """Load retrieval metrics from a CSV, return dict or None."""
    try:
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    except Exception:
        return None
    if not rows:
        return None
    out = {}
    for col in RETRIEVAL_COLS:
        vals = [float(r[col]) for r in rows if r.get(col, "").strip() not in ("", "None")]
        out[col] = sum(vals) / len(vals) if vals else None
    return out


def make_table1(out_path=None):
    """Table 1: Static retrieval on LectureRAG-75 test split (n=59)."""
    rows = []
    for sys_key, display in TABLE1_SYSTEMS:
        if sys_key in HARDCODED:
            hc = HARDCODED[sys_key]
            data = dict(hit_at_1=hc["h1"], hit_at_5=hc["h5"],
                        mrr_score=hc["mrr"], ndcg_at_10=hc["ndcg"], recall_at_10=hc["r10"])
        else:
            # Try retrieval_only dir first, then gen dir
            csv_p = RETRIEVAL_DIR / f"{sys_key}.csv"
            if not csv_p.exists():
                csv_p = GEN_DIR / f"{sys_key}.csv"
            data = _load_retrieval(str(csv_p)) if csv_p.exists() else None

        if data is None:
            vals = ["---"] * 5
        else:
            vals = []
            for col in RETRIEVAL_COLS:
                v = data.get(col)
                vals.append(f"{v:.3f}" if v is not None else "---")
        rows.append((display, vals))

    # Find best per column for bolding
    best = [None] * 5
    for _, vals in rows:
        for i, v in enumerate(vals):
            try:
                f = float(v)
                if best[i] is None or f > best[i]:
                    best[i] = f
            except ValueError:
                pass

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Static retrieval on LectureRAG-75 held-out test split (n=59, k=10).",
        r"\textsuperscript{\textdagger}CFRAG-lite fine-tuned on 174-item train split.",
        r"$\star$ ITMA N=50: 50 oracle-feedback examples, no retraining.}",
        r"\label{tab:retrieval}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"System & H@1 & H@5 & MRR@10 & nDCG@10 & R@10 \\",
        r"\midrule",
    ]

    midrule_before = {"itma_n0"}
    for (sys_key, display), vals in zip(TABLE1_SYSTEMS, [r[1] for r in rows]):
        if sys_key in midrule_before:
            lines.append(r"\midrule")
        cells = [display]
        for i, v in enumerate(vals):
            try:
                if best[i] is not None and abs(float(v) - best[i]) < 1e-4:
                    cells.append(r"\textbf{" + v + "}")
                else:
                    cells.append(v)
            except ValueError:
                cells.append(v)
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"Written: {out_path}")
    return tex


def make_table_gen(out_path=None):
    """Generation quality table (BS-F1, ROUGE-L, BLEU-4, Faithfulness)."""
    import statistics
    GEN_SYSTEMS = [
        ("bm25",         "BM25"),
        ("dense_minilm", "Dense-MiniLM"),
        ("cross_encoder","Cross-Encoder"),
        ("cfrag_lite",   r"CFRAG-lite$^\dagger$"),
        ("itma",         r"\textbf{ITMA (N=0)}"),
    ]
    GEN_COLS = [
        ("bertscore_f1", "BS-F1"),
        ("rouge_l",      "ROUGE-L"),
        ("bleu_4",       "BLEU-4"),
        ("faithfulness", "Faithfulness"),
    ]

    rows = []
    for sys_key, display in GEN_SYSTEMS:
        csv_p = GEN_DIR / f"{sys_key}.csv"
        if not csv_p.exists():
            rows.append((display, ["---"] * len(GEN_COLS)))
            continue
        try:
            data_rows = list(csv.DictReader(open(csv_p, encoding="utf-8")))
        except Exception:
            rows.append((display, ["---"] * len(GEN_COLS)))
            continue
        vals_out = []
        for col, _ in GEN_COLS:
            vals = [float(r[col]) for r in data_rows
                    if r.get(col, "").strip() not in ("", "None")]
            if vals:
                vals_out.append(f"{statistics.mean(vals):.4f}")
            else:
                vals_out.append("---")
        rows.append((display, vals_out))

    # Bold best per column
    best = [None] * len(GEN_COLS)
    for _, vals in rows:
        for i, v in enumerate(vals):
            try:
                f = float(v)
                if best[i] is None or f > best[i]:
                    best[i] = f
            except ValueError:
                pass

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Generation quality on LectureRAG-75 test split (n=59).",
        r"BS-F1 = BERTScore F1; Faithfulness = LLM-judge.}",
        r"\label{tab:gen_quality}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "System & " + " & ".join(h for _, h in GEN_COLS) + r" \\",
        r"\midrule",
    ]
    for display, vals in rows:
        cells = [display]
        for i, v in enumerate(vals):
            try:
                if best[i] is not None and abs(float(v) - best[i]) < 1e-5:
                    cells.append(r"\textbf{" + v + "}")
                else:
                    cells.append(v)
            except ValueError:
                cells.append(v)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"Written: {out_path}")
    return tex


def make_table_ablation(out_path=None):
    """Table 3: Ablation study."""
    rows = [
        (r"ITMA (head + boost)",        "0.8305", "0.9379", r"\checkmark"),
        (r"ITMA no-boost (head only)",  "0.8305", "0.8305", r"$\times$ (flat)"),
        (r"ITMA boost-only",            "0.8475", "0.9379", r"\checkmark"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Ablation study: head vs.\ ID-boost contribution.",
        r"LectureRAG-75 test split (n=59, 3 seeds). H@5 at N=0 and N=50.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Variant & N=0 H@5 & N=50 H@5 & Adapts? \\",
        r"\midrule",
    ]
    for v, n0, n50, adapts in rows:
        lines.append(f"{v} & {n0} & {n50} & {adapts} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"Written: {out_path}")
    return tex


def make_table_msmarco(out_path=None):
    """Table 2: MS-MARCO retrieval (external benchmark, n=100)."""
    MSMARCO_SYSTEMS = [
        ("bm25",          "BM25"),
        ("dense_minilm",  r"Dense-MiniLM"),
        ("dense_mpnet",   r"Dense-MPNet"),
        ("cross_encoder", r"Cross-Encoder"),
        ("itma",          r"\textbf{ITMA (N=0)}"),
    ]

    rows = []
    for sys_key, display in MSMARCO_SYSTEMS:
        csv_p = MSMARCO_DIR / f"{sys_key}.csv"
        data = _load_retrieval(str(csv_p)) if csv_p.exists() else None
        if data is None:
            vals = ["---"] * 5
        else:
            vals = [f"{data.get(col, 0):.3f}" if data.get(col) is not None else "---"
                    for col in RETRIEVAL_COLS]
        rows.append((display, vals))

    # Bold best per column
    best = [None] * 5
    for _, vals in rows:
        for i, v in enumerate(vals):
            try:
                f = float(v)
                if best[i] is None or f > best[i]:
                    best[i] = f
            except ValueError:
                pass

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{MS-MARCO passage retrieval (external benchmark, n=100).",
        r"ITMA (N=0) evaluated cold-start with no in-domain feedback.}",
        r"\label{tab:msmarco}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"System & H@1 & H@5 & MRR@10 & nDCG@10 & R@10 \\",
        r"\midrule",
    ]
    for display, vals in rows:
        cells = [display]
        for i, v in enumerate(vals):
            try:
                if best[i] is not None and abs(float(v) - best[i]) < 1e-4:
                    cells.append(r"\textbf{" + v + "}")
                else:
                    cells.append(v)
            except ValueError:
                cells.append(v)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"Written: {out_path}")
    return tex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="paper/tables")
    args = parser.parse_args()
    out = args.out

    print("\n=== Table 1 (Retrieval) ===")
    print(make_table1(os.path.join(out, "table1_retrieval.tex")))

    print("\n=== Table 2 (MS-MARCO) ===")
    print(make_table_msmarco(os.path.join(out, "table_msmarco.tex")))

    print("\n=== Table Gen (Generation Quality) ===")
    print(make_table_gen(os.path.join(out, "table_gen.tex")))

    print("\n=== Table 3 (Ablation) ===")
    print(make_table_ablation(os.path.join(out, "table_ablation.tex")))


if __name__ == "__main__":
    main()
