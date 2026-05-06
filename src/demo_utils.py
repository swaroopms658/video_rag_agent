"""Shared helpers for the ITMA Streamlit demo.

Caching strategy:
- @st.cache_resource: heavy objects (embedder, warm retriever) — shared across reruns
- st.session_state: per-session state (live retriever on Page 2)
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

QA_PATH = "data/lecture_rag_75/qa.jsonl"
SPLITS_PATH = "data/lecture_rag_75/splits.json"
STORE_PATH = "data/lecture_rag_75/combined"
CHECKPOINT = "checkpoints/itma_head.pt"
COLD_START_CSV = "analysis/cold_start.csv"
ABLATION_CSV = "analysis/ablation_cold_start.csv"


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


@st.cache_resource(show_spinner="Initialising ITMA retriever (N=0)…")
def get_cold_retriever():
    from src.itma.integration import ITMARetriever
    return ITMARetriever(
        store_path=STORE_PATH,
        checkpoint=CHECKPOINT,
        memory_path=None,
    )


@st.cache_resource(show_spinner="Pre-warming ITMA with 50 feedback examples (one-time, ~30 s)…")
def get_warm_retriever():
    """ITMA pre-warmed with 50 oracle-feedback signals from the train split.

    Uses gold_context_ids as oracle (same protocol as cold_start_eval.py:147-155).
    Cached for the entire Streamlit session after first load.
    """
    from src.itma.integration import ITMARetriever
    r = ITMARetriever(
        store_path=STORE_PATH,
        checkpoint=CHECKPOINT,
        memory_path=None,
    )
    embedder = get_embedder()
    train_items = _load_split("train")
    rng = random.Random(0)
    rng.shuffle(train_items)
    n = 0
    for item in train_items:
        gold = item.get("gold_context_ids") or []
        if not gold:
            continue
        r.retrieve_with_ids(item["question"], embedder, top_k=10)
        r.record_feedback(helpful_chunk_ids=gold, reward=1.0)
        n += 1
        if n >= 50:
            break
    return r


def _load_split(split: str) -> list[dict]:
    items = []
    with open(QA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if split == "all":
        return items
    with open(SPLITS_PATH) as f:
        splits_data = json.load(f)
    allowed = set(splits_data.get(split, []))
    return [it for it in items if it.get("id") in allowed]


def get_test_items() -> list[dict]:
    return _load_split("test")


# ---------------------------------------------------------------------------
# Cold-start curve figure (renders inline so PDF files are not needed)
# ---------------------------------------------------------------------------

SYSTEM_STYLE = {
    "dense_minilm":  ("Dense-MiniLM",        "#1f77b4", "-"),
    "static_memory": ("Static τ+β",           "#ff7f0e", "--"),
    "cfrag_lite":    ("CFRAG-lite",           "#d62728", "-."),
    "itma":          ("ITMA (head + boost)",  "#2ca02c", "-"),
}


def build_cold_start_figure(csv_path: str = COLD_START_CSV, metric: str = "hit_at_5",
                             systems: Optional[list[str]] = None):
    """Return a matplotlib Figure of the cold-start adaptation curve.

    Uses the same CSV produced by cold_start_eval.py (columns: system,
    n_feedback, seed, hit_at_5, mrr_score, ndcg_at_10).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sys = row.get("system", "")
            try:
                n = int(row.get("n_feedback", 0))
                val = float(row.get(metric, "nan"))
            except (ValueError, TypeError):
                continue
            if systems and sys not in systems:
                continue
            data[sys][n].append(val)

    BG = "#1a1a2e"
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    plot_order = systems or list(SYSTEM_STYLE.keys())
    for sys_name in plot_order:
        if sys_name not in data:
            continue
        label, color, ls = SYSTEM_STYLE.get(sys_name, (sys_name, None, "-"))
        ns = sorted(data[sys_name].keys())
        means = [np.mean(data[sys_name][n]) for n in ns]
        stds = [np.std(data[sys_name][n]) / max(1, len(data[sys_name][n]) ** 0.5) for n in ns]
        lw = 2.5 if sys_name == "itma" else 1.8
        ax.plot(ns, means, label=label, color=color, linestyle=ls, linewidth=lw,
                marker="o", ms=5 if sys_name == "itma" else 4)
        ax.fill_between(ns,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.12, color=color)

    metric_label = {"hit_at_5": "Hit@5", "mrr_score": "MRR@10",
                    "ndcg_at_10": "nDCG@10"}.get(metric, metric)
    ax.set_xlabel("Number of feedback examples (N)", fontsize=11, color="#e0e0e0")
    ax.set_ylabel(metric_label, fontsize=11, color="#e0e0e0")
    ax.set_title("ITMA cold-start adaptation curve", fontsize=12, fontweight="bold", color="white")
    ax.legend(fontsize=9, facecolor="#252540", edgecolor="#444", labelcolor="white")
    ax.grid(True, alpha=0.2, color="gray")
    ax.tick_params(colors="#c0c0c0")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)
    fig.tight_layout(pad=0.8)
    return fig


def build_ablation_figure(csv_path: str = ABLATION_CSV):
    """Return a matplotlib Figure for the ablation study (3 ITMA variants)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ABLATION_STYLE = {
        "itma":            ("ITMA (head + boost)", "#2ca02c", "-"),
        "itma_no_boost":   ("ITMA no-boost (head only)", "#ff7f0e", "--"),
        "itma_boost_only": ("ITMA boost-only", "#1f77b4", "-."),
    }

    data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sys = row.get("system", "")
            try:
                n = int(row.get("n_feedback", 0))
                val = float(row.get("hit_at_5", "nan"))
            except (ValueError, TypeError):
                continue
            data[sys][n].append(val)

    BG = "#1a1a2e"
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for sys_name, (label, color, ls) in ABLATION_STYLE.items():
        if sys_name not in data:
            continue
        ns = sorted(data[sys_name].keys())
        means = [np.mean(data[sys_name][n]) for n in ns]
        stds = [np.std(data[sys_name][n]) / max(1, len(data[sys_name][n]) ** 0.5) for n in ns]
        lw = 2.2 if sys_name == "itma" else 1.6
        ax.plot(ns, means, label=label, color=color, linestyle=ls, linewidth=lw,
                marker="o", ms=4)
        ax.fill_between(ns,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.12, color=color)

    ax.set_xlabel("Number of feedback examples (N)", fontsize=10, color="#e0e0e0")
    ax.set_ylabel("Hit@5", fontsize=10, color="#e0e0e0")
    ax.set_title("Ablation: head vs. ID-boost contribution", fontsize=11,
                 fontweight="bold", color="white")
    ax.legend(fontsize=8, facecolor="#252540", edgecolor="#444", labelcolor="white")
    ax.grid(True, alpha=0.2, color="gray")
    ax.tick_params(colors="#c0c0c0")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)
    fig.tight_layout(pad=0.6)
    return fig


# ---------------------------------------------------------------------------
# Table 1 data (static, from results.md / analysis/results_test/)
# ---------------------------------------------------------------------------

TABLE1_ROWS = [
    {"System": "BM25",          "H@1": 0.593, "H@5": 0.848, "MRR@10": 0.692, "nDCG@10": 0.722, "R@10": 0.864, "_itma": False},
    {"System": "Dense-MiniLM",  "H@1": 0.525, "H@5": 0.848, "MRR@10": 0.638, "nDCG@10": 0.695, "R@10": 0.898, "_itma": False},
    {"System": "Dense-MPNet",   "H@1": 0.475, "H@5": 0.797, "MRR@10": 0.623, "nDCG@10": 0.692, "R@10": 0.941, "_itma": False},
    {"System": "Cross-Encoder", "H@1": 0.644, "H@5": 0.898, "MRR@10": 0.747, "nDCG@10": 0.789, "R@10": 0.949, "_itma": False},
    {"System": "Static τ+β",    "H@1": 0.525, "H@5": 0.848, "MRR@10": 0.638, "nDCG@10": 0.695, "R@10": 0.898, "_itma": False},
    {"System": "CFRAG-lite†",   "H@1": 0.729, "H@5": 0.915, "MRR@10": 0.811, "nDCG@10": 0.837, "R@10": 0.949, "_itma": False},
    {"System": "ITMA (N=0)",    "H@1": 0.508, "H@5": 0.831, "MRR@10": 0.625, "nDCG@10": 0.688, "R@10": 0.907, "_itma": True},
    {"System": "ITMA (N=50) ★", "H@1": 0.688, "H@5": 0.932, "MRR@10": 0.790, "nDCG@10": 0.832, "R@10": 0.951, "_itma": True},
]

DEMO_QUERY = {
    "id": "3a9222e8b2ec36b9",
    "question": "When was Internet Protocol version 4 (IPv4) first specified?",
    "ground_truth_answer": "IPv4 was first specified in 1983.",
    "gold_context_ids": ["81d62bb9741552dc"],
    "domain": "computer_networks",
}

SAMPLE_QUERIES = [
    "What year did the ARPANET begin?",
    "What is the purpose of the OSI model?",
    "How does a database transaction ensure consistency?",
    "What is gradient descent in machine learning?",
    "What are the main components of an operating system kernel?",
]
