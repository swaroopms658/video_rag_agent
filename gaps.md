# RAG Literature Gap Analysis vs. ITMA
*Mapped against 9 recent IEEE/Q1 papers. Created: 2026-05-06*

---

## Paper-by-paper gap map

| # | Paper | Their gap | What ITMA offers | What ITMA still lacks |
|---|---|---|---|---|
| 1 | Domain-Adapted RAG (IEEE Access 2026) | No cold-start — assumes domain data exists before deployment | Cold-start safety at N=0 | Multi-domain generalization beyond 5 lecture domains |
| 2 | RAG4DS (IEEE Access 2025) | Lifecycle architecture only — no actual adaptation mechanism | Online memory bank fills the "adaptation phase" of their lifecycle | Integration with data-space governance layer |
| 3 | Legal RAG Interpretability Survey (IEEE Access 2025) | Survey only — no novel mechanism; interpretability not implemented | Every rank change in ITMA is traceable to a specific (query, chunk, weight) memory entry — inherently auditable | No formal interpretability metrics (e.g., faithfulness score per retrieved chunk) |
| 4 | CA-RAG Similarity Validation (IEEE Access 2025) | Static retrieval — no learning across queries; addresses hallucination but not feedback | ITMA's memory bank accumulates which chunks are consistently helpful — complementary to their validation step | No hallucination detection or context-inconsistency filtering |
| 5 | Permission-Aware RAG (IEEE Access 2025) | Access control only — ignores adaptation quality | Orthogonal contribution — ITMA could sit on top of their access-filtered corpus | No access control, no security layer |
| 6 | Embedding Quality in RAG (IEEE Access 2025) | Static embeddings — no mechanism for when embeddings fail in a new domain | ID-boost compensates for embedding failures by directly promoting helpful chunk IDs regardless of embedding quality | No embedding adaptation or fine-tuning component |
| 7 | RAGVA Virtual Assistants (JSS 2025) | Single case study (Transurban) — no learning from user interactions | ITMA's memory bank is exactly the missing "learning from user feedback" component they identify as future work | No engineering-level deployment study or real practitioner evaluation |
| 8 | Adaptive Iterative RAG (Neurocomputing 2026) | Iterative retrieval improves per-query quality but doesn't accumulate knowledge across queries | ITMA accumulates cross-query knowledge — complementary to iterative retrieval | ITMA does not iterate within a single query; combining both is an open problem |
| 9 | RAG Architectures Survey (CS Review 2026) | Survey identifies online adaptation as a key open challenge explicitly | ITMA is a direct answer to this gap | No graph-RAG component; no multimodal coverage |

---

## Unified gap map (what the whole field is missing)

### Gap 1 — Cold-start is completely ignored ⭐ (ITMA's core contribution)
**Evidence:** None of the 9 papers address N=0 deployment.
Papers 1, 2, 7 all assume some domain data exists before evaluation.
CA-RAG (Paper 4) validates context quality but still needs a populated retrieval corpus.

**What ITMA does:** Starts within 2% of Dense-MiniLM at N=0. No other paper in this list claims cold-start safety.

**How to strengthen the claim:**
- Add a "cold-start latency" experiment: how many feedback examples until ITMA matches each baseline?
- Report this as a "breakeven N" metric (currently only shown for CFRAG-lite)

---

### Gap 2 — No feedback loop in any of the 9 papers ⭐ (ITMA's second core contribution)
**Evidence:**
- RAG4DS lists "adaptation" as a lifecycle phase but provides no mechanism
- RAGVA identifies real-user feedback as future work
- Domain-Adapted RAG requires offline fine-tuning (not online)
- CA-RAG validates context quality per-query but has no memory across queries

**What ITMA does:** Counterfactual reweighting updates memory weights after every query — O(|M|) cost, no gradients.

**How to strengthen the claim:**
- Compare explicitly against CA-RAG's similarity validation as a complementary baseline
- Show ITMA + CA-RAG (ID-boost + context validation) outperforms either alone

---

### Gap 3 — Interpretability is claimed but not implemented
**Evidence:** Legal RAG paper (Paper 3) surveys interpretability but proposes no mechanism.
CA-RAG is validation-based but doesn't explain WHY a chunk was retrieved.

**What ITMA already has:** Every score change is traceable: `s'_j = ŝ + Σ cos(q,eᵢ)·wᵢ` — each term names the exact past query and chunk responsible for the boost.

**How to exploit this gap:**
- Add an "explanation" output to the retriever: for each retrieved chunk, list which memory entries contributed to its boost and by how much
- Add one metric: "Explanation Coverage" — % of rank changes where the ID-boost term > 0.05

---

### Gap 4 — Embedding failure is addressed statically, not adaptively
**Evidence:** Paper 6 (Embedding Quality) shows embedding quality is the bottleneck but offers no solution — only measurement.

**What ITMA does:** ID-boost bypasses embedding quality for chunks that have been helpful before. If `cos(q, c_j)` is low but the chunk was previously helpful for similar queries, the ID-boost overrides the embedding score.

**Concrete experiment to add:**
- Deliberately degrade embeddings (add Gaussian noise to chunk vectors)
- Show ITMA degrades less than Dense-MiniLM as embedding noise increases
- This directly responds to Paper 6's gap

---

### Gap 5 — No educational domain benchmark exists in any of the 9 papers ⭐
**Evidence:**
- Legal RAG uses legal documents (no standard benchmark)
- CA-RAG uses TriviaQA, NQ, SQuAD (general knowledge)
- Domain-Adapted RAG uses industrial docs (not public)
- RAGVA uses road operation company data (proprietary)

**What ITMA does:** LectureRAG-75 is the only public educational RAG benchmark with cold-start splits.

**How to exploit this gap:**
- Position LectureRAG-75 as a community benchmark contribution — submit separately to a data/benchmark track
- Mention in the paper that no existing benchmark supports cold-start evaluation protocol

---

### Gap 6 — Iterative within-query vs. cross-query learning are conflated
**Evidence:** Adaptive Iterative RAG (Paper 8) improves retrieval by re-querying within a single interaction. ITMA improves retrieval across queries over time. These are orthogonal problems that no paper combines.

**Experiment to add:**
- Combine ITMA memory bank with 2-round iterative retrieval
- Query 1: ITMA retrieves top-20, LLM reads and generates a follow-up query
- Query 2: ITMA re-retrieves with enriched query
- Compare: ITMA alone vs. Iterative alone vs. ITMA + Iterative

---

### Gap 7 — Security / access control + adaptation not combined
**Evidence:** Permission-Aware RAG (Paper 5) filters the corpus but doesn't learn from which filtered results were useful.

**Niche extension:** ITMA could be deployed on top of a permission-filtered corpus — memory bank only stores feedback on chunks the user was permitted to see. This is a deployment contribution no paper has made.

---

## What ITMA uniquely covers (publish as explicit claims)

| Unique claim | Supporting evidence |
|---|---|
| First cold-start-safe adaptive retriever | No other paper in this list demonstrates N=0 safety |
| Gradient-free online adaptation | All adaptive systems here require offline fine-tuning or LLM calls |
| First educational lecture RAG benchmark with cold-start splits | CA-RAG, Legal, Domain-Adapted all use existing or proprietary data |
| Interpretable adaptation (every rank change auditable) | Legal RAG survey calls for this but provides no mechanism |
| Embedding-failure compensation via ID-boost | Directly addresses Paper 6's identified gap |

---

## Recommended additions to journal.tex based on this analysis

1. **Add one sentence to Related Work §2.4** citing CA-RAG (Paper 4) and explaining ITMA is complementary (memory across queries vs. validation within query)
2. **Add one sentence to Related Work §2.1** citing the RAG architectures survey (Paper 9) and noting online adaptation as the open challenge ITMA addresses
3. **Add "breakeven N" metric** — at what N does ITMA match each baseline? Currently only done visually in Figure 3
4. **Add embedding noise degradation experiment** (1 day) — directly responds to Paper 6
5. **Add "Explanation Coverage" metric** to Table 2 — exploits the interpretability gap Paper 3 identifies

---

## Target journal recommendation (updated)

Given the gap analysis, **Information Processing & Management (Q1, IF=8.6)** is a better fit than MTAP because:
- IP&M has published both RAG papers and educational IR papers
- Their recent call-for-papers explicitly mentions "adaptive retrieval" and "feedback-driven IR"
- The gap analysis shows ITMA addresses open problems stated in their published papers
- Citing their published papers in the related work strengthens the submission

*See future.md for the full Q1 upgrade roadmap.*
