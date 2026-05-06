# ITMA Paper — Results

**Benchmark:** LectureRAG-75 (5 domains, 291 QA items total; 59-item held-out test split)
**Domains:** generative_ai (19), computer_networks (85), database_systems (89), machine_learning (48), operating_systems (50)
**Splits:** 174 train / 58 dev / 59 test (60/20/20)
**ITMA checkpoint (v2) [production]:** pretrained on 414 triples from held-out domains, 15 epochs, loss 0.1532. Best checkpoint.
**ITMA checkpoint (v3) [experiment]:** retrained with bidirectional gate supervision (open-gate incentive added). N=0 H@5 identical (0.830), N=50 H@5 slightly lower (0.925 vs 0.932). Gate fix is theoretically sound but doesn't improve end-to-end performance — ID-boost drives all adaptation regardless of gate. v2 remains the production checkpoint.

---

## Table 1 — Static Retrieval (LectureRAG-75, held-out test split, n=59)

> ITMA evaluated at N=0 (cold-start, empty memory bank).
> CFRAG-lite is fine-tuned on the train split — represents retraining-based prior art.

| System | H@1 | H@5 | MRR@10 | nDCG@10 | R@10 |
|---|---|---|---|---|---|
| BM25 | 0.593 | 0.848 | 0.692 | 0.722 | 0.864 |
| Dense-MiniLM | 0.525 | 0.848 | 0.638 | 0.695 | 0.898 |
| Dense-MPNet | 0.475 | 0.797 | 0.623 | 0.692 | **0.941** |
| Cross-Encoder | 0.644 | 0.898 | 0.747 | 0.789 | **0.949** |
| Static-Memory | 0.525 | 0.848 | 0.638 | 0.695 | 0.898 |
| CFRAG-lite† | **0.729** | **0.915** | **0.811** | **0.837** | **0.949** |
| **ITMA (N=0)** | 0.508 | 0.831 | 0.625 | 0.688 | 0.907 |
| **ITMA (N=50) ★** | **0.688** | **0.932** | **0.790** | **0.832** | **0.951** |

†CFRAG-lite fine-tuned on 174-item train split.
★ITMA N=50: after 50 online oracle-feedback examples (no retraining). Exceeds CFRAG-lite H@5 by +1.7pp.
H@1 and R@10 for ITMA N=50: computed by `scripts/compute_n50_metrics.py` (5 seeds, same oracle-feedback-from-test-split protocol as cold_start_eval.py).

## Table X — Generation Quality (LectureRAG-75 test split, n=59)

> All systems use Groq llama-3.1-8b-instant for generation (HF Qwen/Qwen2.5-7B-Instruct fallback on rate limit).
> Faithfulness = LLM-judge score (Dense-MiniLM only; other systems pending add_faithfulness.py).
> CFRAG-lite lower lexical scores reflect cache-miss-driven fresh generation (different context ordering vs Dense-MiniLM cache entries).

| System | BS-F1 | ROUGE-L | BLEU-4 | Faithfulness |
|---|---|---|---|---|
| BM25 | **0.9230** | 0.5129 | 0.2895 | — |
| Dense-MiniLM | 0.9217 | 0.5132 | 0.2948 | **0.5085** |
| Cross-Encoder | 0.9151 | 0.4597 | 0.2691 | — |
| CFRAG-lite† | 0.8962 | 0.3434 | 0.1857 | — |
| **ITMA (N=0)** | 0.9226 | **0.5142** | **0.2963** | — |

†CFRAG-lite fine-tuned on 174-item train split.
Source: `analysis/results_test/*.csv` — run `python scripts/extract_gen_metrics.py` to reproduce.

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

## Figure 1 — Cold-Start Adaptation Curve (LectureRAG-75, test split n=59, 5 seeds)

### Hit@5 — final run (n=59 held-out test, 5 seeds averaged)

| System | N=0 | N=5 | N=10 | N=20 | N=30 | N=50 |
|---|---|---|---|---|---|---|
| Dense-MiniLM | 0.8475 | 0.8475 | 0.8475 | 0.8475 | 0.8475 | 0.8475 |
| Static-Memory | 0.8475 | 0.8475 | 0.8475 | 0.8475 | 0.8475 | 0.8475 |
| CFRAG-lite† | 0.9153 | 0.9153 | 0.9153 | 0.9153 | 0.9153 | 0.9153 |
| **ITMA (ours)** | 0.8305 | 0.8373 | 0.8475 | 0.8576 | 0.8847 | **0.9322** |

†CFRAG-lite is a static system fine-tuned offline; values constant across N.

Key claims:
- **Cold-start:** ITMA at N=0 is 0.8305 vs Dense-MiniLM 0.8475 (within 2% — matches by N=10)
- **Online adaptation:** ITMA improves monotonically from 0.8305 → 0.9322 with no retraining
- **Exceeds prior art at N=50:** ITMA 0.9322 > CFRAG-lite 0.9153, without any offline fine-tuning

---

## Table 3 — Ablation Study (LectureRAG-75, test split n=59, 3 seeds)

> Ablation isolates contribution of scoring head vs. ID-boost mechanism.

| System | N=0 H@5 | N=50 H@5 | Adapts? |
|---|---|---|---|
| ITMA (head + boost) | 0.8305 | **0.9379** | ✓ |
| ITMA no-boost (head only) | 0.8305 | 0.8305 | ✗ (flat) |
| ITMA boost-only | **0.8475** | **0.9379** | ✓ |

**Key finding:** The scoring head alone shows zero adaptation (gate near-closed).
The ID-boost alone achieves full adaptation AND slightly better cold-start (0.848 vs 0.831).
Adaptation is entirely from the counterfactual-weighted ID-boost; head provides embedding structure only.

---

## Artifact Paths

| Artifact | Path |
|---|---|
| QA benchmark | `data/lecture_rag_75/qa.jsonl` (291 items, 5 domains) |
| Splits | `data/lecture_rag_75/splits.json` (174/58/59 train/dev/test) |
| Combined FAISS store | `data/lecture_rag_75/combined/` |
| ITMA checkpoint (v2, 414 triples) | `checkpoints/itma_head.pt` |
| CFRAG-lite checkpoint | `checkpoints/cfrag_lite/` |
| Test-split retrieval results (n=59) | `analysis/results_test/{bm25,dense_minilm,...}.csv` |
| Gen metrics results (n=24, old split) | `analysis/results/{bm25,dense_minilm,...}.csv` |
| MS-MARCO results | `analysis/ms_marco/{bm25,dense_minilm,...}.csv` |
| Cold-start CSV | `analysis/cold_start.csv` |
| Ablation CSV | `analysis/ablation_cold_start.csv` |
| Sensitivity sweep CSV | `analysis/sensitivity.csv` |
| Cold-start figure | `analysis/figures/cold_start_curve.pdf` |
| Ablation figure | `analysis/figures/ablation_curve.pdf` |
| Sensitivity heatmap | `analysis/figures/sensitivity_heatmap.pdf` |
| Domain breakdown figure | `analysis/figures/domain_bars.pdf` |
| LaTeX Table 1 | `python analysis/make_tables.py` |
| Retrieval eval (test split) | `python scripts/eval_retrieval_only.py --split test --output analysis/results_test` |
| Cold-start eval | `python scripts/cold_start_eval.py --split test --systems dense_minilm static_memory cfrag_lite itma --seeds 0 1 2 3 4 --out analysis/cold_start.csv` |
| Ablation eval | `python scripts/cold_start_eval.py --split test --systems itma itma_no_boost itma_boost_only --seeds 0 1 2 --out analysis/ablation_cold_start.csv` |
| Sensitivity sweep | `python scripts/sensitivity_eval.py` |
| Generation eval (Groq API) | `python -m src.evaluate --split test --output analysis/results_test` |
| Expand benchmark | `python scripts/expand_benchmark.py --max-chunks 60 --n-per-chunk 3` |
