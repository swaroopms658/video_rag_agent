# Top-Tier Upgrade Roadmap — ITMA Journal Paper
*Written: 2026-05-07. Goal: move from ESWA borderline-Accept → TKDE / IP&M / ESWA strong-Accept.*
*All experiments use existing infrastructure. No new hardware beyond Colab T4 free tier.*

---

## Exp 1 — Noisy Feedback Curve
**Effort:** 2 hours | **Resources:** zero | **Impact:** kills the #1 reviewer objection

**The objection:** "Oracle feedback is unrealistic. Real users give noisy signals."

**What to build:** Add `--noise-rate` flag to `scripts/cold_start_eval.py`.

```python
# Add after line ~109 in cold_start_eval.py
def corrupt_feedback(gold_ids, noise_rate, all_chunk_ids, rng):
    return [
        rng.choice(all_chunk_ids) if rng.random() < noise_rate else cid
        for cid in gold_ids
    ]

# Add CLI arg:
parser.add_argument("--noise-rate", type=float, default=0.0,
                    help="Fraction of feedback labels to corrupt (0=oracle)")
```

**Run:**
```powershell
python scripts/cold_start_eval.py --system itma --noise-rate 0.0   # oracle (existing)
python scripts/cold_start_eval.py --system itma --noise-rate 0.1
python scripts/cold_start_eval.py --system itma --noise-rate 0.2
python scripts/cold_start_eval.py --system itma --noise-rate 0.3
```

**Expected output:** One new figure — Hit@5 at N=50 vs. noise rate. Sub-linear degradation expected because counterfactual reweighting and bounded weights provide structural robustness. Even at 30% noise, ITMA should still match Dense-MiniLM.

**Paper addition:** New subsection in §8 "Robustness to Noisy Feedback". One figure. Two paragraphs. Changes "planned" in §9 Limitations to "we find sub-linear degradation, confirming the structural robustness argument."

---

## Exp 2 — Rocchio PRF Baseline
**Effort:** 1 day | **Resources:** no GPU needed | **Impact:** empirically defends novelty against IR experts

**The objection:** "Pseudo-relevance feedback (Rocchio) is a classical IR method. How does ITMA differ?"

**What to build:** `src/baselines/rocchio.py` (~50 lines):

```python
class RocchioPRFRetriever(BaseRetriever):
    """alpha=1, beta=0.75 — standard Rocchio with explicit relevance feedback."""

    def __init__(self, store_path, alpha=1.0, beta=0.75):
        self.alpha = alpha
        self.beta = beta
        self._base = SimpleRetriever(store_path)  # Dense-MiniLM backbone
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

Register as `"rocchio_prf"` in `cold_start_eval.py` `build_retriever()`. Run same cold-start eval.

**Expected result:** Rocchio PRF plateaus early (query drift). ITMA keeps improving past N=30. One plot. One paragraph in §8 showing that ID-boost + counterfactual reweighting outperforms query-space adaptation.

**Paper addition:** Add Rocchio column to Figure 1 (cold-start curve). Add one paragraph to §4.3 comparing mechanisms formally. Converts the Rocchio discussion from defensive to empirical.

---

## Exp 3 — Expand Benchmark to 400+ Items
**Effort:** 1 day | **Resources:** Groq free tier (~$0) | **Impact:** biggest single credibility jump

**The objection:** "89 test items is too small. Results may not be stable."

**What to run:** `scripts/expand_benchmark.py` is already written and ready.

```powershell
python scripts/expand_benchmark.py
```

This expands LectureRAG-75 from 442 → 400+ verified QA pairs using Groq Llama-3.1-8B for generation and verification. New splits: ~240 train / ~80 dev / ~120 test.

**After expansion, re-run everything:**
```powershell
python scripts/cold_start_eval.py --split test --seeds 5
python scripts/eval_ms_marco.py
python scripts/sensitivity_eval.py
python analysis/make_plots.py
```

**Expected result:** Bootstrap CIs tighten from ±1-2 pp to ±0.5-1 pp. The ITMA N=50 > CFRAG-lite significance becomes more robust. With 120 test items the evaluation scale objection disappears entirely.

**Paper impact:** Update all tables with new numbers. Caption changes: "n=89 test items" → "n=120 test items". This is the single change most likely to move a reviewer from "Major Revision" to "Minor Revision."

---

## Exp 4 — Fix Gate / Curriculum Pretraining
**Effort:** 3–5 days | **Resources:** Colab T4 free tier | **Impact:** neural head actually contributes → strongest novelty claim

**The objection:** "Gate bias=-3 means g≈0.047. The scoring head is essentially dead. ITMA reduces to ID-boost with a fixed dense retriever."

**What to build:** Modify pretraining to include memory-populated examples so the gate learns to open.

```python
# In scripts/pretrain_itma_head.py — add curriculum stage:
# Stage 1 (existing): pairs where memory_bank is empty → gate stays closed
# Stage 2 (new): pairs where memory_bank has 5-20 relevant entries → gate should open
#   - Positive signal: query matches memory entry, chunk_id matches → gate loss targets g=0.8
#   - Use synthetic memory population during training

# Key change: add gate supervision loss
gate_target = torch.ones_like(gate_logits) * 0.8  # gate should be open when memory has hits
gate_loss = F.binary_cross_entropy_with_logits(gate_logits, gate_target) * 0.5
total_loss = retrieval_loss + gate_loss
```

Retrain on Colab T4 (~2 hours). Save new checkpoint. Re-run cold_start_eval.py with new checkpoint.

**Expected result:** Gate opens at N≥5 (g≈0.6-0.8). Scoring head contributes alongside ID-boost. Cold-start curve improves further, especially at N=5-20. The "dead gate" becomes "gate that learns when to trust memory" — a genuine finding about adaptive retrieval architectures.

**Paper impact:** Converts the gate-closure limitation from a bug to a feature that was never fully exercised. §4.1 "Design Philosophy" gets stronger. The "minimal sufficient mechanism" framing becomes "we show the gate adds value when trained with curriculum." This is the change most likely to move ESWA Minor Revision → TKDE / IP&M submittable.

---

## Priority Order

| Priority | Experiment | Time | Impact |
|---|---|---|---|
| 1 | **Exp 3: Expand benchmark** | 1 day | Credibility (89→120 test items) |
| 2 | **Exp 1: Noisy feedback** | 2 hours | Kills oracle objection |
| 3 | **Exp 2: Rocchio PRF** | 1 day | Empirical novelty defense |
| 4 | **Exp 4: Gate curriculum** | 3-5 days | Architecture completeness |

**Doing just Exp 1 + Exp 3** gets the paper to solid ESWA Accept (no revision needed).
**Doing all 4** makes it submittable to TKDE or IP&M.

---

## What These Unlock

| After | Score | Realistic target |
|---|---|---|
| Current (journal.tex as-is) | 7/10 | ESWA Minor Revision |
| + Exp 1 (noisy feedback) | 7.5/10 | ESWA Accept |
| + Exp 3 (expand benchmark) | 8/10 | ESWA strong Accept |
| + Exp 1 + 3 | 8.5/10 | ESWA Accept / IP&M submittable |
| + All 4 | 9/10 | IP&M / TKDE submittable |
