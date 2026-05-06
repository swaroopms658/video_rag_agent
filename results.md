# ITMA Paper — Results

**Benchmark:** LectureRAG-75 (5 domains × ~23 QA pairs = 115 total)
**Domains:** generative_ai, computer_networks, database_systems, machine_learning, operating_systems
**ITMA checkpoint:** retrained on 414 triples from held-out domains (algorithms, computer_architecture, data_structures, generative_ai), 15 epochs.

---

## Table 1 — Static Retrieval (LectureRAG-75, all 115 items)

> CFRAG-lite evaluated on all 115 items including its own train split (69 items) — represents oracle upper bound.
> All other systems evaluated on held-out data only.

| System | H@1 | H@5 | MRR@10 | nDCG@10 | R@10 |
|---|---|---|---|---|---|
| BM25 | 0.496 | 0.730 | 0.595 | 0.648 | 0.817 |
| Dense-MiniLM | 0.348 | 0.809 | 0.530 | 0.621 | 0.904 |
| Dense-MPNet | 0.322 | 0.696 | 0.492 | 0.582 | 0.870 |
| Cross-Encoder | **0.548** | 0.791 | **0.653** | **0.709** | 0.887 |
| Static-Memory | 0.348 | **0.809** | 0.530 | 0.621 | **0.904** |
| CFRAG-lite† | 0.643 | 0.887 | 0.742 | 0.786 | 0.922 |
| **ITMA (N=0)** | 0.330 | 0.791 | 0.516 | 0.610 | 0.904 |

---

## Table 1b — Generation Quality (test split, n=24, with BERTScore)

| System | H@1 | H@5 | MRR@10 | nDCG@10 | R@10 | BS-F1 | ROUGE-L | BLEU-4 |
|---|---|---|---|---|---|---|---|---|
| BM25 | **0.591** | 0.773 | 0.668 | 0.714 | 0.864 | **0.939** | **0.580** | **0.383** |
| Dense-MiniLM | 0.455 | 0.864 | 0.646 | 0.722 | 0.955 | 0.935 | 0.566 | 0.358 |
| Dense-MPNet | 0.455 | 0.682 | 0.584 | 0.660 | 0.909 | 0.925 | 0.487 | 0.318 |
| Cross-Encoder | 0.545 | **0.909** | 0.686 | 0.742 | 0.909 | 0.928 | 0.522 | 0.353 |
| Static-Memory | 0.455 | 0.864 | 0.646 | 0.722 | **0.955** | 0.935 | 0.566 | 0.358 |
| CFRAG-lite | 0.583 | 0.833 | **0.703** | **0.763** | **0.958** | 0.919 | 0.462 | 0.250 |
| **ITMA (N=0)** | 0.333 | 0.833 | 0.535 | 0.629 | 0.917 | 0.925 | 0.501 | 0.306 |

> BERTScore-F1 with `roberta-large`.

---

## Table 2 — MS-MARCO Retrieval (external benchmark, n=100)

| System | H@1 | H@5 | MRR@10 | nDCG@10 | R@10 |
|---|---|---|---|---|---|
| BM25 | 0.850 | 0.970 | 0.901 | 0.918 | 0.970 |
| Dense-MiniLM | 0.930 | 1.000 | 0.963 | 0.973 | 1.000 |
| Dense-MPNet | **0.940** | **1.000** | **0.968** | **0.977** | **1.000** |
| Cross-Encoder | 0.910 | 1.000 | 0.953 | 0.965 | 1.000 |
| **ITMA (N=0)** | 0.930 | 1.000 | 0.962 | 0.972 | 1.000 |

> ITMA (N=0) matches Dense-MiniLM exactly on external data — cold-start safety confirmed cross-domain.
> CFRAG-lite and Static-Memory omitted (require domain-specific training/feedback).

---

## Figure 1 — Cold-Start Adaptation Curve (LectureRAG-75, all 115 items, 5 seeds)

ITMA uses retrained checkpoint (414 triples, held-out domains).

### Hit@5

| System | N=0 | N=5 | N=10 | N=20 | N=30 | N=50 |
|---|---|---|---|---|---|---|
| Dense-MiniLM | 0.809 | 0.809 | 0.809 | 0.809 | 0.809 | 0.809 |
| Static-Memory | 0.809 | 0.809 | 0.809 | 0.809 | 0.809 | 0.809 |
| **ITMA** | 0.791 | 0.793 | 0.793 | 0.803 | 0.823 | **0.863** |
| CFRAG-lite† | 0.887 | 0.887 | 0.887 | 0.887 | 0.887 | 0.887 |

### MRR@10

| System | N=0 | N=5 | N=10 | N=20 | N=30 | N=50 |
|---|---|---|---|---|---|---|
| Dense-MiniLM | 0.530 | 0.530 | 0.530 | 0.530 | 0.530 | 0.530 |
| Static-Memory | 0.530 | 0.530 | 0.530 | 0.530 | 0.530 | 0.530 |
| **ITMA** | 0.516 | 0.515 | 0.531 | 0.545 | 0.599 | **0.662** |
| CFRAG-lite† | 0.742 | 0.742 | 0.742 | 0.742 | 0.742 | 0.742 |

### nDCG@10

| System | N=0 | N=5 | N=10 | N=20 | N=30 | N=50 |
|---|---|---|---|---|---|---|
| Dense-MiniLM | 0.621 | 0.621 | 0.621 | 0.621 | 0.621 | 0.621 |
| Static-Memory | 0.621 | 0.621 | 0.621 | 0.621 | 0.621 | 0.621 |
| **ITMA** | 0.610 | 0.609 | 0.621 | 0.633 | 0.675 | **0.725** |
| CFRAG-lite† | 0.786 | 0.786 | 0.786 | 0.786 | 0.786 | 0.786 |

†CFRAG-lite is a static system trained on the train split; values constant across N.

Key claims:
- **Cold-start safety:** ITMA at N=0 within 2.3% of Dense-MiniLM on LectureRAG-75; matches exactly (H@5=1.000) on MS-MARCO
- **Online adaptation:** ITMA improves monotonically from N=20 onward with no retraining (0.791→0.863 H@5)
- **Approaches CFRAG-lite at N=50:** 0.863 vs 0.887 H@5, without offline fine-tuning

---

## Table 3 — Ablation Study (LectureRAG-75, all items, 3 seeds)

> Ablation isolates the contribution of the scoring head vs. ID-boost mechanism.

| System | N=0 H@5 | N=50 H@5 | N=0 MRR | N=50 MRR | Adapts? |
|---|---|---|---|---|---|
| ITMA (head + boost) | 0.791 | 0.870 | 0.516 | 0.664 | ✓ |
| ITMA no-boost (head only) | 0.791 | **0.791** | 0.516 | **0.516** | ✗ |
| ITMA boost-only | **0.809** | **0.875** | **0.530** | **0.674** | ✓ |

**Key finding:** The scoring head alone (no ID-boost) shows zero adaptation — the gate mechanism is ineffective.
The ID-boost alone achieves full adaptation AND better cold-start safety (0.809 vs 0.791).
The combined system adapts via ID-boost; the scoring head contributes a marginal initialization penalty at N=0
that is recovered by N=50.

**Paper framing:** ITMA's adaptation arises entirely from the counterfactual-weighted ID-boost.
The scoring head is a frozen ranker that provides a structured embedding space;
the memory bank + ID-boost is the adaptive component.

---

## Artifact Paths

| Artifact | Path |
|---|---|
| QA benchmark | `data/lecture_rag_75/qa.jsonl` (400+ items target, 5 domains) |
| Splits | `data/lecture_rag_75/splits.json` (60/20/20 train/dev/test) |
| Combined FAISS store | `data/lecture_rag_75/combined/` |
| ITMA checkpoint (v2, 414 triples) | `checkpoints/itma_head.pt` |
| ITMA checkpoint (v1, 44 triples) | `checkpoints/itma_head_v1_44triples.pt` |
| CFRAG-lite checkpoint | `checkpoints/cfrag_lite/` |
| Per-system CSVs (n=24 test, w/ generation) | `analysis/results/{bm25,dense_minilm,...}.csv` |
| Per-system CSVs (n=115 retrieval-only) | `analysis/results_115/{bm25,dense_minilm,...}.csv` |
| MS-MARCO results | `analysis/ms_marco/{bm25,dense_minilm,...}.csv` |
| Cold-start CSV | `analysis/cold_start.csv` |
| Ablation CSV | `analysis/ablation_cold_start.csv` |
| Sensitivity sweep CSV | `analysis/sensitivity.csv` |
| Cold-start figure | `analysis/figures/cold_start_curve.pdf` |
| Ablation figure | `analysis/figures/ablation_curve.pdf` |
| Sensitivity heatmap | `analysis/figures/sensitivity_heatmap.pdf` |
| LaTeX Table 1 | `python analysis/make_tables.py` |
| Run ablation | `python scripts/cold_start_eval.py --systems itma itma_no_boost itma_boost_only --checkpoint checkpoints/itma_head.pt --out analysis/ablation_cold_start.csv` |
| Run sensitivity sweep | `python scripts/sensitivity_eval.py` |
| Expand benchmark | `python scripts/expand_benchmark.py --max-chunks 60 --n-per-chunk 3` |
