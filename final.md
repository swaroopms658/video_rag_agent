# ITMA — Final Implementation Plan for Q1
*What to build, where to build it, and the single Q1 unlock.*
*Created: 2026-05-06*

---

## Part A — Gaps from gaps.md that can be implemented NOW

All of these use existing infrastructure. No new data collection needed.

---

### A1. Breakeven-N metric (2 hours)
**Gap addressed:** Cold-start claim is only visual in Figure 3. Needs a number.
**What:** For each baseline, find the exact N where ITMA first matches or exceeds it.

**Where:** `analysis/cold_start.csv` already has all the data. Add a script:

```python
# scripts/compute_breakeven_n.py
import pandas as pd
import numpy as np

df = pd.read_csv("analysis/cold_start.csv")
baselines = {"dense_minilm": 0.848, "cfrag_lite": 0.915, "cross_encoder": 0.898}

itma = df[df.system == "itma"].groupby("n_feedback")["hit_at_5"].mean()

for name, threshold in baselines.items():
    crossed = itma[itma >= threshold]
    n = crossed.index[0] if len(crossed) else "never"
    print(f"ITMA matches {name} ({threshold:.3f}) at N={n}")
```

**Output:** One sentence to add to Section 6: *"ITMA matches Dense-MiniLM at N=X, Cross-Encoder at N=Y, and exceeds CFRAG-lite at N=50."*

---

### A2. Expand MS-MARCO to 500 queries (2 hours)
**Gap addressed:** 100 queries → too small for out-of-domain claim.
**What:** Re-run with 5× more queries.

**Where:** Two files need changing:

`scripts/build_ms_marco_slice.py` — change the slice size:
```python
# current: MAX_QUERIES = 100
MAX_QUERIES = 500   # change this one line
```

Then re-run:
```powershell
python scripts/build_ms_marco_slice.py
python scripts/eval_ms_marco.py
```

Results go to `analysis/ms_marco/` — update `paper/tables/table_msmarco.tex` and the n=100 caption in journal.tex to n=500.

---

### A3. Explanation Coverage metric (1 day)
**Gap addressed:** Legal RAG survey calls for interpretability — ITMA already has it, just not measured.
**What:** For each test query, report what % of top-5 rank changes are explained by the ID-boost.

**Where:** `src/itma/integration.py` — modify `retrieve_with_ids()` to also return per-chunk boost breakdown:

```python
# In ITMARetriever.retrieve_with_ids(), after computing s'_j
# add a boost_contributions dict:
boost_contributions = {}
for j, chunk_id in enumerate(candidate_ids):
    boost = sum(
        np.dot(q_emb, entry.embedding) * entry.weight
        for entry in self.memory_bank.entries
        if entry.chunk_id == chunk_id
    )
    boost_contributions[chunk_id] = float(boost)
```

Then in `scripts/cold_start_eval.py`, compute:
```python
# explanation_coverage = fraction of top-5 where boost > 0.05
coverage = sum(1 for cid in top5_ids if boost_contributions.get(cid, 0) > 0.05) / 5
```

Report as a column in Table 2. Expected: coverage rises from 0% at N=0 to ~60-80% at N=50.

---

### A4. Embedding noise degradation experiment (1 day)
**Gap addressed:** Paper 6 (Embedding Quality) identifies embedding failure as unsolved. ITMA solves it via ID-boost.
**What:** Add Gaussian noise to chunk embeddings, measure how fast each system degrades.

**Where:** New script `scripts/eval_embedding_noise.py`:

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from src.itma.integration import ITMARetriever
from src.baselines import DenseMiniLMRetriever  # or SimpleRetriever

noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]

for sigma in noise_levels:
    # Load FAISS store, add noise to all stored chunk embeddings
    # noise = np.random.normal(0, sigma, embedding.shape)
    # noisy_embedding = embedding + noise
    # Re-run retrieval eval at N=50 for ITMA and N=0 for Dense-MiniLM
    # Record hit_at_5
    pass
```

**Expected result:** Dense-MiniLM degrades linearly with noise. ITMA degrades sub-linearly because ID-boost bypasses embedding quality for known-helpful chunks. One plot, one paragraph in Section 9 (MS-MARCO section).

---

### A5. CA-RAG citation in Related Work (30 minutes)
**Gap addressed:** CA-RAG (IEEE Access 2025) does context validation within a query. ITMA does adaptation across queries. Citing them as complementary strengthens both claims.

**Where:** `journal.tex` Section 2.1 (Related Work — RAG), add after the SELF-RAG sentence:

```latex
Context-Aware RAG~\citep{carag2025} reduces hallucinations by validating
retrieved chunks via similarity scoring within a single query; unlike ITMA,
it holds no state across queries and therefore cannot address cold-start
adaptation.
```

Add bibitem:
```latex
\bibitem[Zhang et al.(2025)]{carag2025}
Zhang, X., et al., 2025.
Context-aware retrieval-augmented generation using similarity validation
to handle context inconsistencies in large language models.
IEEE Access. doi:10.1109/ACCESS.2025.3614553.
```

---

## Part B — Items from future.md implementable in current codebase

---

### B1. Bootstrap confidence intervals (1 day)
**Where:** `src/eval_utils.py` — add one function:

```python
# src/eval_utils.py  (add after line 92)
from scipy.stats import bootstrap as scipy_bootstrap

def bootstrap_ci(scores: list[float], n_boot: int = 2000,
                 ci: float = 0.95) -> tuple[float, float]:
    result = scipy_bootstrap(
        (scores,), np.mean,
        n_resamples=n_boot,
        confidence_level=ci,
        method="percentile",
        random_state=42,
    )
    return result.confidence_interval.low, result.confidence_interval.high
```

**Where used:** `scripts/cold_start_eval.py` line 229 — extend CSV fieldnames:
```python
fieldnames = ["system", "n_feedback", "seed",
              "hit_at_5", "hit_at_5_ci_lo", "hit_at_5_ci_hi",
              "mrr_score", "ndcg_at_10"]
```

Then in `analysis/make_plots.py`, render CI as shaded band (already done for std — swap in CI bounds).

**Impact:** Every table and figure now reports `mean [95% CI]`. Removes the #1 statistical objection.

---

### B2. Noisy feedback experiment (1 day)
**Where:** `scripts/cold_start_eval.py` — add `noise_rate` argument and corrupt function:

```python
# Add to cold_start_eval.py after line 109
def corrupt_feedback(gold_ids: list, noise_rate: float,
                     all_chunk_ids: list, rng) -> list:
    return [
        rng.choice(all_chunk_ids) if rng.random() < noise_rate else cid
        for cid in gold_ids
    ]
```

Add CLI argument at line 199:
```python
parser.add_argument("--noise-rate", type=float, default=0.0,
                    help="Fraction of feedback labels to corrupt (0=oracle)")
```

Run:
```powershell
python scripts/cold_start_eval.py --system itma --noise-rate 0.0  # oracle
python scripts/cold_start_eval.py --system itma --noise-rate 0.1
python scripts/cold_start_eval.py --system itma --noise-rate 0.2
python scripts/cold_start_eval.py --system itma --noise-rate 0.3
python scripts/cold_start_eval.py --system itma --noise-rate 0.5
```

**Output:** One new figure: Hit@5 at N=50 vs. noise rate. Expected sub-linear degradation. Addresses oracle-feedback criticism directly.

---

### B3. Rocchio PRF baseline (2 days)
**Where:** New file `src/baselines/rocchio.py`:

```python
class RocchioPRFRetriever(BaseRetriever):
    """Classic Rocchio relevance feedback — alpha=1, beta=0.75, gamma=0."""

    def __init__(self, store_path, alpha=1.0, beta=0.75):
        self.alpha = alpha
        self.beta = beta
        self._base = SimpleRetriever(store_path)   # Dense-MiniLM
        self._feedback_vecs: list[np.ndarray] = []

    def record_feedback(self, helpful_chunk_ids, **_):
        for cid in helpful_chunk_ids:
            vec = self._base.get_embedding_by_id(cid)
            if vec is not None:
                self._feedback_vecs.append(vec)

    def retrieve_with_ids(self, query, embedder, top_k=10, **_):
        q_vec = embedder.encode(query, normalize_embeddings=True)
        if self._feedback_vecs:
            pos_mean = np.mean(self._feedback_vecs, axis=0)
            q_vec = self.alpha * q_vec + self.beta * pos_mean
            q_vec /= np.linalg.norm(q_vec) + 1e-9
        return self._base._search(q_vec, top_k)
```

Register in `cold_start_eval.py` `build_retriever()` as `"rocchio_prf"`.

Run same cold-start eval. If ITMA beats Rocchio PRF by N=50 → novelty claim is defended against IR-expert reviewers.

---

## Part C — The ONE thing that clinches Q1

**`expand_benchmark.py` already exists and is already written.**

```
scripts/expand_benchmark.py   ← ALREADY IN CODEBASE
```

This script expands LectureRAG-75 from ~115 to 400+ QA pairs using:
1. Draft new candidates via Groq Llama-3.1-8B (draft_qa.py pipeline)
2. Verify via BERTScore matching
3. Regenerate train/dev/test splits preserving domain proportions

**Running it:**
```powershell
python scripts/expand_benchmark.py
```

**What it unlocks:**
- Test set: 30 items → **100+ items** (borderline Q1 → Q1)
- Train set: 60 items → **240+ items** (more feedback examples available)
- Bootstrap CIs at n=100+ are tight enough (±2-3 pp) to make gains significant
- Re-running cold_start_eval.py on the new split takes ~2 hours

**After expansion, re-run all experiments:**
```powershell
python scripts/cold_start_eval.py --split test --seeds 5
python scripts/eval_ms_marco.py          # already at 500 queries after A2
python scripts/sensitivity_eval.py
```

Regenerate all figures:
```powershell
python analysis/make_plots.py
```

Update paper/tables/ with new numbers, recompile journal.tex.

---

## Combined effort estimate

| Task | File(s) | Time | Q1 impact |
|---|---|---|---|
| A1 Breakeven-N | new compute script | 2 hrs | +citation strength |
| A2 MS-MARCO 500q | build_ms_marco_slice.py L1 | 2 hrs | +out-of-domain claim |
| A3 Explanation metric | itma/integration.py | 1 day | +interpretability |
| A4 Embedding noise | new eval script | 1 day | +directly answers Paper 6 |
| A5 CA-RAG citation | journal.tex §2.1 | 30 min | +positioning |
| B1 Bootstrap CIs | eval_utils.py + cold_start_eval.py | 1 day | **+statistical validity** |
| B2 Noisy feedback | cold_start_eval.py | 1 day | +realism |
| B3 Rocchio PRF | new src/baselines/rocchio.py | 2 days | +IR baseline defence |
| **C  expand_benchmark.py** | **already written — just run it** | **1 day** | **THE Q1 UNLOCK** |

**Total: ~8 days of work → Q1-ready submission**

---

## Score after full implementation

| After | Score | Target journal |
|---|---|---|
| Current | 5/10 | MTAP Q2 (risky) |
| A items only | 6/10 | MTAP Q2 (comfortable) |
| A + B1 (CIs) | 7/10 | ESWA Q1 (submittable) |
| A + B + C (expand) | 8.5/10 | IP&M / ESWA Q1 (strong) |

---

*See gaps.md for paper-by-paper literature comparison.*
*See future.md for longer-horizon Q1 experiments (user study, gate fix).*
