"""Page 1 — ITMA Simulation walkthrough.

Six-step narrative driven by st.session_state["sim_step"]:
  0. Architecture — how ITMA works (text sketch)
  1. Demo query — the query we will run twice
  2. Side-by-side retrieval — N=0 vs N=50
  3. Cold-start curve — Figure 1 rendered inline
  4. Table 1 — static test-set retrieval results
  5. Closing — key claim summary
"""

import streamlit as st
import pandas as pd

from src.demo_utils import (
    get_embedder,
    get_cold_retriever,
    get_warm_retriever,
    build_cold_start_figure,
    build_ablation_figure,
    TABLE1_ROWS,
    DEMO_QUERY,
    COLD_START_CSV,
    ABLATION_CSV,
    inject_global_css,
    ORANGE, ORANGE_DARK, ORANGE_BG, ORANGE_100, ORANGE_200,
)


_ARROW_HTML = """
<div style="display:flex;align-items:center;justify-content:center;height:100%;
            font-size:1.8rem;color:#F97316;font-weight:700;line-height:1;
            padding-top:1.2rem">→</div>
"""

_DOWN_HINT_HTML = """
<div style="text-align:center;margin:0.4rem 0 0.6rem 0;color:#F97316;
            font-size:0.85rem;font-weight:600;letter-spacing:0.05em">
    ↓ &nbsp; then &nbsp; ↓
</div>
"""


def _flow_arrow_block(label: str = "after 50 feedback signals") -> str:
    """Vertical arrow + caption used between two side-by-side panels (e.g. cold → warm)."""
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                height:100%;padding-top:2.5rem">
        <div style="font-size:2.4rem;color:{ORANGE};font-weight:800;line-height:1">→</div>
        <div style="font-size:0.75rem;color:{ORANGE_DARK};font-weight:600;
                    text-align:center;margin-top:0.35rem;max-width:80px">
            {label}
        </div>
    </div>
    """


st.set_page_config(
    page_title="ITMA — Simulation",
    page_icon="📖",
    layout="wide",
)

inject_global_css()

N_STEPS = 6

# Pre-load both retrievers at page open — the spinner fires once here,
# so step 4 (side-by-side) renders instantly when the user navigates to it.
# (~30 s on first visit, then cached for the whole session.)
_embedder = get_embedder()
_cold_r = get_cold_retriever()
_warm_r = get_warm_retriever()

if "sim_step" not in st.session_state:
    st.session_state.sim_step = 0


def go_next():
    st.session_state.sim_step = min(st.session_state.sim_step + 1, N_STEPS - 1)


def go_back():
    st.session_state.sim_step = max(st.session_state.sim_step - 1, 0)


def go_restart():
    st.session_state.sim_step = 0


# ── page header ─────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="background:linear-gradient(135deg,{ORANGE_DARK} 0%,{ORANGE} 100%);
                border-radius:12px;padding:1.2rem 2rem;margin-bottom:1.2rem;color:#fff;
                display:flex;align-items:center;justify-content:space-between">
        <div>
            <h3 style="margin:0;color:#fff;font-weight:800">ITMA — Simulation Walkthrough</h3>
            <p style="margin:0;opacity:0.88;font-size:0.88rem">
                Stepped narrative: cold-start retrieval · memory adaptation · results
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── progress bar + navigation ────────────────────────────────────────────────
step = st.session_state.sim_step
st.progress((step + 1) / N_STEPS, text=f"Step {step + 1} of {N_STEPS}")

nav_left, nav_mid, nav_right = st.columns([1, 6, 1])
with nav_left:
    if step > 0:
        st.button("← Back", on_click=go_back, use_container_width=True)
with nav_right:
    if step < N_STEPS - 1:
        st.button("Next →", on_click=go_next, type="primary", use_container_width=True)
    else:
        st.button("↩ Restart", on_click=go_restart, use_container_width=True)

st.divider()

# ── step content ─────────────────────────────────────────────────────────────

# ── Step 0: Architecture ─────────────────────────────────────────────────────
if step == 0:
    st.title("Agentic Video RAG, An Explainable Framework For Video Retrieval")

    col_arch, col_legend = st.columns([3, 2], gap="large")

    with col_arch:
        st.markdown(
            f"""
            <style>
              .arch-box {{
                border: 1.5px solid {ORANGE_200}; border-radius: 10px;
                background: #FFF7ED; padding: 0.65rem 0.9rem;
                color: #1C1917; box-shadow: 0 1px 4px rgba(249,115,22,0.07);
              }}
              .arch-head {{ background: {ORANGE_BG}; border-color: {ORANGE}; }}
              .arch-title {{ font-weight: 700; font-size: 0.93rem; color: #1C1917; }}
              .arch-sub {{ font-size: 0.78rem; color: #57534E; margin-top: 0.15rem;
                           line-height: 1.45; }}
              .arch-arrow-row {{
                display: flex; align-items: center; justify-content: center;
                gap: 0.55rem; margin: 0.30rem 0;
              }}
              .arch-num {{
                display: inline-flex; align-items: center; justify-content: center;
                width: 26px; height: 26px; border-radius: 50%;
                background: {ORANGE}; color: #fff; font-weight: 800;
                font-size: 0.82rem; box-shadow: 0 1px 3px rgba(249,115,22,0.35);
                flex-shrink: 0;
              }}
              .arch-arrow {{
                font-size: 1.4rem; color: {ORANGE_DARK}; font-weight: 800;
                line-height: 1;
              }}
              .arch-lbl {{ font-size: 0.78rem; color: {ORANGE_DARK}; font-weight: 600; }}
              .arch-phase {{
                font-size: 0.70rem; font-weight: 800; color: {ORANGE_DARK};
                letter-spacing: 0.10em; text-transform: uppercase;
                border-top: 1px dashed {ORANGE_200}; padding-top: 0.45rem;
                margin: 0.6rem 0 0.25rem 0;
              }}
              .arch-pair {{
                display: grid; grid-template-columns: 1fr auto 1fr;
                align-items: stretch; gap: 0.55rem; margin: 0.30rem 0;
              }}
              .arch-pair-mid {{
                display: flex; flex-direction: column; align-items: center;
                justify-content: center; gap: 0.25rem;
                font-size: 0.72rem; color: {ORANGE_DARK}; font-weight: 600;
              }}
              .arch-loop {{
                font-size: 0.74rem; color: {ORANGE_DARK}; font-style: italic;
                background: #FFFBF5; border-left: 3px solid {ORANGE};
                padding: 0.35rem 0.6rem; margin: 0.30rem 0; line-height: 1.45;
              }}
            </style>

            <div class="arch-phase">Phase A · Ingestion (per new video)</div>

            <div class="arch-box arch-head">
              <div class="arch-title">🎬 Video Input Handler</div>
              <div class="arch-sub">Accepts uploaded lecture/training video.</div>
            </div>

            <div class="arch-arrow-row">
              <span class="arch-num">1</span><span class="arch-arrow">↓</span>
              <span class="arch-lbl">Extracted Audio</span>
            </div>

            <div class="arch-box">
              <div class="arch-title">🎙️ ASR Module</div>
              <div class="arch-sub">faster-whisper INT8 — audio → timestamped text.</div>
            </div>

            <div class="arch-arrow-row">
              <span class="arch-num">2</span><span class="arch-arrow">↓</span>
              <span class="arch-lbl">Transcript</span>
            </div>

            <div class="arch-box">
              <div class="arch-title">✂️ Text Chunking Module</div>
              <div class="arch-sub">Splits transcript into overlapping chunks.</div>
            </div>

            <div class="arch-arrow-row">
              <span class="arch-num">3</span><span class="arch-arrow">↓</span>
              <span class="arch-lbl">Text Chunks</span>
            </div>

            <div class="arch-box">
              <div class="arch-title">🧬 Embedding Generator</div>
              <div class="arch-sub">all-MiniLM-L6-v2 — 384-d vector per chunk.</div>
            </div>

            <div class="arch-arrow-row">
              <span class="arch-num">4</span><span class="arch-arrow">↓</span>
              <span class="arch-lbl">Embeddings</span>
            </div>

            <div class="arch-box arch-head">
              <div class="arch-title">📚 FAISS Indexing</div>
              <div class="arch-sub">Flat inner-product index over all chunks.</div>
            </div>

            <div class="arch-phase">Phase B · User entry &amp; agentic routing</div>

            <div class="arch-box arch-head">
              <div class="arch-title">👤 User</div>
              <div class="arch-sub">Submits a question or a new video.</div>
            </div>

            <div class="arch-arrow-row">
              <span class="arch-num">5</span><span class="arch-arrow">↓</span>
              <span class="arch-lbl">Query / Video Input</span>
            </div>

            <div class="arch-pair">
              <div class="arch-box arch-head" style="grid-column:1">
                <div class="arch-title">🤖 Agentic Decision Engine</div>
                <div class="arch-sub">Routes: cache hit · retrieve · ingest new video.</div>
              </div>
              <div class="arch-pair-mid">
                <div><span class="arch-num">6</span></div>
                <div class="arch-arrow">→</div>
                <div>Check Cache</div>
                <div class="arch-arrow">←</div>
                <div><span class="arch-num">7</span> Cache Status</div>
              </div>
              <div class="arch-box" style="grid-column:3">
                <div class="arch-title">💾 Cache Manager</div>
                <div class="arch-sub">Serves cached answers for repeat queries.</div>
              </div>
            </div>

            <div class="arch-loop">
              <span class="arch-num" style="width:22px;height:22px;font-size:0.74rem">8</span>
              &nbsp;<b>If Needed</b> — Agentic Decision Engine routes back up to the
              <b>Video Input Handler</b> when a new video must be ingested before
              the query can be answered.
            </div>

            <div class="arch-arrow-row">
              <span class="arch-num">9</span><span class="arch-arrow">↓</span>
              <span class="arch-lbl">Process Query</span>
            </div>

            <div class="arch-phase">Phase C · Retrieval &amp; response</div>

            <div class="arch-pair">
              <div class="arch-box arch-head" style="grid-column:1">
                <div class="arch-title">🔍 Query Processor</div>
                <div class="arch-sub">Embeds q, queries FAISS, assembles context.</div>
              </div>
              <div class="arch-pair-mid">
                <div><span class="arch-num">10</span></div>
                <div class="arch-arrow">→</div>
                <div>Search Relevant Data</div>
                <div class="arch-arrow">←</div>
                <div><span class="arch-num">11</span> Retrieved Chunks</div>
              </div>
              <div class="arch-box" style="grid-column:3">
                <div class="arch-title">📚 FAISS Indexing</div>
                <div class="arch-sub">Same index built in Phase&nbsp;A.</div>
              </div>
            </div>

            <div class="arch-arrow-row">
              <span class="arch-num">12</span><span class="arch-arrow">↓</span>
              <span class="arch-lbl">Query + Context</span>
            </div>

            <div class="arch-box">
              <div class="arch-title">🧠 LLM Response Generator</div>
              <div class="arch-sub">Grounded answer conditioned on retrieved chunks.</div>
            </div>

            <div class="arch-arrow-row">
              <span class="arch-num">13</span><span class="arch-arrow">↓</span>
              <span class="arch-lbl">Final Answer</span>
            </div>

            <div class="arch-box">
              <div class="arch-title">🖥️ Output Interface</div>
              <div class="arch-sub">Renders answer + citations to the UI.</div>
            </div>

            <div class="arch-arrow-row">
              <span class="arch-num">14</span><span class="arch-arrow">↓</span>
              <span class="arch-lbl">Display Response</span>
            </div>

            <div class="arch-box arch-head">
              <div class="arch-title">👤 User</div>
              <div class="arch-sub">Reads response · may mark chunks helpful · may ask the next question (loops back to arrow&nbsp;5).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_legend:
        st.markdown(
            """
            **Arrow legend (14 labelled edges):**

            **Phase A — Ingestion**
            **1** Extracted Audio · **2** Transcript · **3** Text Chunks ·
            **4** Embeddings

            **Phase B — User entry & routing**
            **5** Query / Video Input · **6** Check Cache ·
            **7** Cache Status · **8** *If Needed* loop back to Video Input ·
            **9** Process Query

            **Phase C — Retrieval & response**
            **10** Search Relevant Data · **11** Retrieved Chunks ·
            **12** Query + Context · **13** Final Answer ·
            **14** Display Response

            ---

            **Key components**

            🤖 **Agentic Decision Engine**
            The router — decides whether to ingest a new video, hit the
            cache, or run retrieval. Closes the loop between user, cache,
            and the retrieval pipeline.

            💾 **Cache Manager**
            Skips ASR / embedding / retrieval for repeat queries.

            📚 **FAISS Indexing**
            The single shared index — written by Phase A, read by Phase C.

            🧠 **LLM Response Generator**
            Produces the grounded answer from query + retrieved chunks.
            """
        )

    st.divider()
    st.markdown("##### How the six ITMA pipeline steps work together")

    def _sim_card(num, icon, title, items, note, bg="#FFF7ED"):
        li = "".join(f"<li style='color:#44403C'>{s}</li>" for s in items)
        return f"""
        <div style="border:1.5px solid #FED7AA;border-radius:10px;
                    padding:1rem;background:{bg};color:#1C1917;height:100%;
                    box-shadow:0 1px 6px rgba(249,115,22,0.07)">
          <div style="font-size:0.68rem;font-weight:700;color:#F97316;margin-bottom:0.3rem;
                      letter-spacing:0.08em;text-transform:uppercase">STEP {num}</div>
          <div style="font-weight:700;font-size:0.97rem;margin-bottom:0.5rem;color:#1C1917">{icon} {title}</div>
          <ol style="margin:0 0 0.5rem 0;padding-left:1.15rem;font-size:0.86rem;line-height:1.75">{li}</ol>
          <div style="font-size:0.8rem;color:#78716C;border-top:1px solid #FED7AA;
                      padding-top:0.4rem">{note}</div>
        </div>"""

    r1a, r1arrA, r1b, r1arrB, r1c = st.columns([6, 1, 6, 1, 6], gap="small")
    with r1arrA:
        st.markdown(_ARROW_HTML, unsafe_allow_html=True)
    with r1arrB:
        st.markdown(_ARROW_HTML, unsafe_allow_html=True)
    with r1a:
        st.markdown(_sim_card("1", "⚡", "Efficient Caching",
            ["FAISS index loaded once from disk, reused every query",
             "Chunk embeddings cached after first encode",
             "Repeated queries served without re-encoding"],
            "src: src/itma/integration.py · src/agent.py · src/demo_utils.py"), unsafe_allow_html=True)
    with r1b:
        st.markdown(_sim_card("2", "🎙️", "Optimised ASR for Edge Computing",
            ["faster-whisper INT8 quantization: ~4× less memory than FP32",
             "Built-in VAD filter skips silent segments, cuts hallucinations",
             "Falls back to openai-whisper if faster-whisper unavailable",
             "Transcripts chunked → embedded → 103 FAISS chunks across 5 domains"],
            "src: src/transcribe.py · src/build_vectorstore.py · scripts/build_domain_stores.py", "#FFEDD5"), unsafe_allow_html=True)
    with r1c:
        st.markdown(_sim_card("3", "🤖", "Agentic Reasoning Framework",
            ["Observe: FAISS retrieves 20 candidate chunks",
             "Reason: scoring head evaluates (query, chunk, memory summary)",
             "Act: re-ranked top-K returned as retrieval response",
             "Reflect: feedback updates memory → future queries benefit"],
            "src: src/itma/integration.py:_rank() · src/itma/scoring_head.py · src/rag_chain.py"), unsafe_allow_html=True)

    st.markdown(_DOWN_HINT_HTML, unsafe_allow_html=True)

    r2a, r2arrA, r2b, r2arrB, r2c = st.columns([6, 1, 6, 1, 6], gap="small")
    with r2arrA:
        st.markdown(_ARROW_HTML, unsafe_allow_html=True)
    with r2arrB:
        st.markdown(_ARROW_HTML, unsafe_allow_html=True)
    with r2a:
        st.markdown(_sim_card("4", "🧠", "Smart Memory",
            ["Memory bank stores helpful (query, chunk) pairs with weights",
             "At query time: bank is attended → memory summary m",
             "ID-boost re-ranks chunks matched to similar past queries",
             "Counterfactual reweighting updates weights from feedback"],
            "src: src/itma/memory_bank.py · src/itma/integration.py:record_feedback() · src/itma/scoring_head.py", "#FFEDD5"), unsafe_allow_html=True)
    with r2b:
        st.markdown(_sim_card("5", "📊", "Performance Evaluation",
            ["Hit@5, MRR@10, nDCG@10 measured at N=0,5,10,20,30,50",
             "Compared against 5 baselines on 59-item held-out test split",
             "ITMA N=50: H@5 0.932 > CFRAG-lite 0.915 without retraining"],
            "src: scripts/cold_start_eval.py · scripts/sensitivity_eval.py · scripts/eval_retrieval_only.py · analysis/make_plots.py"), unsafe_allow_html=True)
    with r2c:
        st.markdown(_sim_card("6", "💾", "Cost-Aware Caching Logic",
            ["FAISS store checked on disk before rebuild (skip-if-exists)",
             "Warm retriever built once, cached for entire session",
             "Chunk embedding cache avoids redundant encoder calls",
             "Answer cache skips LLM for repeated query+context pairs"],
            "src: src/answer_cache.py · src/rag_chain.py · scripts/build_domain_stores.py · src/demo_utils.py", "#FFEDD5"), unsafe_allow_html=True)

# ── Step 1: Demo query ───────────────────────────────────────────────────────
elif step == 1:
    st.title("The demo query")
    st.markdown(
        "We will run the **same query** through ITMA twice — once with an empty memory "
        "bank (cold start) and once after 50 oracle-feedback examples have been seen."
    )

    st.markdown(
        f"""
        <div style="
            background: {ORANGE_BG};
            border-left: 4px solid {ORANGE};
            border-radius: 6px;
            padding: 1rem 1.2rem;
            font-size: 1.15rem;
            font-weight: 600;
            color: #1C1917;
            box-shadow: 0 1px 6px rgba(249,115,22,0.08);
        ">
        ❓ {DEMO_QUERY['question']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown(
        f"**Domain:** `{DEMO_QUERY['domain']}`  |  "
        f"**Gold chunk:** `{DEMO_QUERY['gold_context_ids'][0]}`  |  "
        f"**Answer:** {DEMO_QUERY.get('ground_truth_answer', '')}"
    )
    st.info(
        "At N=0 (empty memory bank) the gold chunk ranks **#4**. "
        "After 50 feedback examples from the training split the gold chunk rises to **#1** — "
        "with no retraining.",
        icon="👁️",
    )

# ── Step 2: Side-by-side retrieval ──────────────────────────────────────────
elif step == 2:
    st.title("Cold start vs. 50 feedback examples — same query")

    query = DEMO_QUERY["question"]
    gold_ids = set(DEMO_QUERY["gold_context_ids"])

    cold_results = _cold_r.retrieve_with_ids(query, _embedder, top_k=5)
    warm_results = _warm_r.retrieve_with_ids(query, _embedder, top_k=5)

    def _render_results(results, label):
        st.subheader(label)
        for rank, (text, score, chunk_id) in enumerate(results, start=1):
            is_gold = chunk_id in gold_ids
            border = ORANGE if is_gold else "#FED7AA"
            bg = ORANGE_BG if is_gold else "#FAFAF9"
            gold_badge = f" &nbsp;<span style='color:{ORANGE_DARK};font-weight:700'>★ gold chunk</span>" if is_gold else ""
            short_id = chunk_id[:12] + "…"
            st.markdown(
                f"""
                <div style="
                    border: 2px solid {border};
                    border-radius: 6px;
                    background: {bg};
                    padding: 0.7rem 1rem;
                    margin-bottom: 0.5rem;
                    color: inherit;
                ">
                <span style="font-weight:600">#{rank}</span>
                &nbsp;<code style="font-size:0.8rem">{short_id}</code>
                &nbsp;<code style="font-size:0.8rem;opacity:0.7">{score:.4f}</code>{gold_badge}<br>
                <span style="font-size:0.9rem;line-height:1.5">{text[:280]}{'…' if len(text) > 280 else ''}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    col_cold, col_arrow, col_warm = st.columns([10, 1, 10], gap="medium")
    with col_cold:
        _render_results(cold_results, "ITMA at N=0 (empty memory bank)")
    with col_arrow:
        st.markdown(_flow_arrow_block("after 50 feedback signals"), unsafe_allow_html=True)
    with col_warm:
        _render_results(warm_results, "ITMA at N=50 (50 feedback examples)")

    cold_ranks = [r[2] for r in cold_results]
    warm_ranks = [r[2] for r in warm_results]
    cold_pos = next((i + 1 for i, cid in enumerate(cold_ranks) if cid in gold_ids), None)
    warm_pos = next((i + 1 for i, cid in enumerate(warm_ranks) if cid in gold_ids), None)

    if cold_pos and warm_pos:
        if warm_pos < cold_pos:
            st.success(
                f"Gold chunk moved from rank **{cold_pos}** → rank **{warm_pos}** "
                f"after 50 feedback examples, with no retraining.",
                icon="📈",
            )
        elif warm_pos == cold_pos:
            st.info(
                f"Gold chunk is rank **{cold_pos}** in both cases — already well-ranked at N=0.",
                icon="ℹ️",
            )
    elif warm_pos and not cold_pos:
        st.success(
            f"Gold chunk **entered** the top-5 (rank {warm_pos}) after 50 feedback examples. "
            "Was outside top-5 at cold start.",
            icon="📈",
        )

# ── Step 3: Cold-start curve ─────────────────────────────────────────────────
elif step == 3:
    st.title("Figure 1 — Cold-start adaptation curve")
    st.caption(
        "Hit@5 vs. number of feedback examples. Averaged over 5 seeds, "
        "held-out test split (n=59)."
    )

    import os
    if os.path.exists(COLD_START_CSV):
        fig = build_cold_start_figure(COLD_START_CSV, metric="hit_at_5")
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning(f"Cold-start CSV not found at `{COLD_START_CSV}`. Run `scripts/cold_start_eval.py` first.")

    st.markdown(
        """
        **Three key claims:**

        1. **Cold-start safety:** ITMA at N=0 is 0.831 vs Dense-MiniLM 0.848 — within 2%.
           Recovers fully by N=10. No degradation at deployment.

        2. **Monotonic online adaptation:** Hit@5 rises 0.831 → **0.932** from N=0 to N=50
           with no gradient updates, no retraining, no LLM in the loop.

        3. **Exceeds prior art:** ITMA at N=50 (0.932) > CFRAG-lite (0.915), which was
           fine-tuned offline on the 174-item train split.
        """
    )

# ── Step 4: Table 1 ──────────────────────────────────────────────────────────
elif step == 4:
    st.title("Table 1 — Static retrieval results")
    st.caption(
        "LectureRAG-75 held-out test split, n=59. ITMA shown at N=0 (cold start) and N=50 (★). "
        "†CFRAG-lite fine-tuned on the 174-item train split."
    )

    df = pd.DataFrame([{k: v for k, v in row.items() if k != "_itma"} for row in TABLE1_ROWS])
    df = df.set_index("System")

    def _highlight_itma(row):
        if "ITMA" in row.name:
            bg = "#FFEDD5" if "N=50" in row.name else "#FFF7ED"
            return [f"background-color: {bg}; font-weight: bold"] * len(row)
        return [""] * len(row)

    def _bold_best(col):
        best = col.max()
        return [f"font-weight: bold; color: {ORANGE_DARK}" if v == best else "" for v in col]

    styled = (
        df.style
        .apply(_highlight_itma, axis=1)
        .apply(_bold_best, axis=0)
        .format("{:.3f}")
    )
    st.dataframe(styled, use_container_width=True)

    # ── Metric definitions panel ──────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:#FFF7ED;border:1.5px solid #FED7AA;border-radius:10px;
                    padding:1rem 1.25rem;margin-top:0.6rem">
        <div style="font-weight:700;color:#EA580C;margin-bottom:0.45rem;
                    font-size:0.95rem">📐 What each metric measures</div>
        <table style="width:100%;border-collapse:collapse;font-size:0.86rem;
                      line-height:1.6;color:#1C1917">
          <thead>
            <tr style="border-bottom:1px solid #FED7AA;color:#78716C;text-align:left">
              <th style="padding:0.25rem 0.6rem 0.35rem 0">Metric</th>
              <th style="padding:0.25rem 0.6rem 0.35rem 0">What it asks</th>
              <th style="padding:0.25rem 0.6rem 0.35rem 0;text-align:center">Math range</th>
              <th style="padding:0.25rem 0;text-align:center">Good on LectureRAG-75</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="padding:0.3rem 0.6rem 0.3rem 0;font-weight:600">H@1</td>
              <td style="padding:0.3rem 0.6rem 0.3rem 0">Did the gold chunk land at rank&nbsp;1?</td>
              <td style="padding:0.3rem 0.6rem 0.3rem 0;text-align:center">[0, 1]</td>
              <td style="padding:0.3rem 0;text-align:center">≥ 0.60 strong, ≥ 0.70 SOTA</td>
            </tr>
            <tr style="background:#FFFBF5">
              <td style="padding:0.3rem 0.6rem 0.3rem 0;font-weight:600">H@5</td>
              <td style="padding:0.3rem 0.6rem 0.3rem 0">Is the gold chunk anywhere in the top-5?</td>
              <td style="padding:0.3rem 0.6rem 0.3rem 0;text-align:center">[0, 1]</td>
              <td style="padding:0.3rem 0;text-align:center">≥ 0.85 strong, ≥ 0.90 SOTA</td>
            </tr>
            <tr>
              <td style="padding:0.3rem 0.6rem 0.3rem 0;font-weight:600">MRR@10</td>
              <td style="padding:0.3rem 0.6rem 0.3rem 0">
                Average of 1/(rank of gold) — rewards getting gold higher.
                0 if gold is outside the top-10.
              </td>
              <td style="padding:0.3rem 0.6rem 0.3rem 0;text-align:center">[0, 1]</td>
              <td style="padding:0.3rem 0;text-align:center">≥ 0.65 strong, ≥ 0.80 SOTA</td>
            </tr>
            <tr style="background:#FFFBF5">
              <td style="padding:0.3rem 0.6rem 0.3rem 0;font-weight:600">nDCG@10</td>
              <td style="padding:0.3rem 0.6rem 0.3rem 0">
                Discounted-gain measure of the entire top-10 ordering,
                normalised to the ideal ordering.
              </td>
              <td style="padding:0.3rem 0.6rem 0.3rem 0;text-align:center">[0, 1]</td>
              <td style="padding:0.3rem 0;text-align:center">≥ 0.70 strong, ≥ 0.82 SOTA</td>
            </tr>
            <tr>
              <td style="padding:0.3rem 0.6rem 0.3rem 0;font-weight:600">R@10</td>
              <td style="padding:0.3rem 0.6rem 0.3rem 0">
                Fraction of all gold chunks present anywhere in the top-10.
              </td>
              <td style="padding:0.3rem 0.6rem 0.3rem 0;text-align:center">[0, 1]</td>
              <td style="padding:0.3rem 0;text-align:center">≥ 0.90 strong, ≥ 0.95 SOTA</td>
            </tr>
          </tbody>
        </table>
        <div style="font-size:0.78rem;color:#78716C;margin-top:0.5rem">
          All five metrics are bounded by [0, 1] with <b>higher = better</b>.
          They progress in stringency from H@1 (must be rank&nbsp;1) → H@5 (must be in top-5) →
          MRR@10 (rank-aware over top-10) → nDCG@10 (full-order-aware over top-10) →
          R@10 (multi-gold recall).
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        **Note:** At N=0, ITMA lags CFRAG-lite on H@1 (0.508 vs 0.729) — expected,
        since CFRAG-lite has already seen the training data. After 50 online feedback
        examples (★ row), ITMA H@5 **0.932 > CFRAG-lite 0.915** with no offline fine-tuning.
        """
    )

    st.divider()
    st.subheader("Table 3 — Ablation: what drives adaptation?")
    st.caption("3 seeds, same test split. Isolates the scoring head vs. ID-boost mechanism.")

    ablation_rows = [
        {"Variant": "ITMA (head + boost)", "N=0 H@5": 0.8305, "N=50 H@5": 0.9379, "Adapts": "✓"},
        {"Variant": "ITMA no-boost (head only)", "N=0 H@5": 0.8305, "N=50 H@5": 0.8305, "Adapts": "✗ (flat)"},
        {"Variant": "ITMA boost-only", "N=0 H@5": 0.8475, "N=50 H@5": 0.9379, "Adapts": "✓"},
    ]
    abl_df = pd.DataFrame(ablation_rows).set_index("Variant")

    def _abl_style(row):
        if row.name == "ITMA no-boost (head only)":
            return ["opacity: 0.55"] * len(row)
        return ["font-weight: bold"] * len(row)

    st.dataframe(
        abl_df.style.apply(_abl_style, axis=1).format("{:.4f}", subset=["N=0 H@5", "N=50 H@5"]),
        use_container_width=True,
    )
    st.info(
        "**Key finding:** The scoring head alone shows **zero adaptation** (gate bias ≈ −3, "
        "effectively closed at cold start). All improvement comes from the ID-boost — "
        "a lightweight, no-gradient instance-level memory mechanism.",
        icon="💡",
    )

    import os
    if os.path.exists(ABLATION_CSV):
        abl_fig = build_ablation_figure(ABLATION_CSV)
        st.pyplot(abl_fig, use_container_width=True)

# ── Step 5: Closing slide ─────────────────────────────────────────────────────
elif step == 5:
    st.title("Summary")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {ORANGE_DARK} 0%, {ORANGE} 100%);
            border-radius: 10px;
            padding: 1.5rem 2rem;
            font-size: 1.2rem;
            font-weight: 800;
            color: #fff;
            letter-spacing: 0.01em;
            box-shadow: 0 4px 18px rgba(249,115,22,0.25);
        ">
        Same retriever. No retraining. Adapts as it serves.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cold-start gap vs Dense-MiniLM", "−1.7 pp", help="H@5 at N=0")
    with col2:
        st.metric("ITMA at N=50 vs CFRAG-lite", "+1.7 pp", help="0.932 vs 0.915 H@5")
    with col3:
        st.metric("Feedback examples to match baseline", "N=10", help="H@5 ≥ Dense-MiniLM by N=10")

    st.markdown(
        """
        **What makes ITMA different from prior art:**

        | Property | CFRAG / R3 / RankRAG | **ITMA** |
        |---|---|---|
        | Retraining at deployment | ✓ (required) | **✗ never** |
        | LLM in adaptation loop | Some | **✗ none** |
        | Cold-start safe | Varies | **✓ within 2%** |
        | Adapts online | ✗ | **✓** |

        **Core mechanism:** Counterfactual-reweighted ID-boost in the memory bank
        (`src/itma/integration.py:192-208`). The ablation shows that the scoring
        head alone shows zero adaptation (gate bias ≈ −3); all improvement comes
        from the ID-boost — a simple but effective form of instance-level memory.
        """
    )

    st.divider()
    st.markdown("**→ Try it yourself on the Live Demo page.**")
    st.page_link("pages/2_Live_Demo.py", label="Open live demo →", icon="🔬")
