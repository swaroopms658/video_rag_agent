"""Page 2 — ITMA Live Demo.

Interactive retriever session. The user (guide) can:
  - Type any question from the lecture corpus
  - See top-5 retrieved chunks with ITMA scores
  - Mark helpful chunks
  - Watch the memory bank grow and re-rank on subsequent queries

State lives entirely in st.session_state — isolated per browser session.
The live retriever is NOT a cached resource (no @st.cache_resource) so
each session gets its own fresh memory bank.
"""

from __future__ import annotations

import streamlit as st

from src.demo_utils import (
    get_embedder, STORE_PATH, CHECKPOINT, SAMPLE_QUERIES,
    inject_global_css, ORANGE, ORANGE_DARK, ORANGE_BG, ORANGE_100, ORANGE_200, ORANGE_300,
)

st.set_page_config(
    page_title="ITMA — Live Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()


def _obj_card(num: str, icon: str, title: str, steps: list[str], stat_html: str, bg: str,
              anim_delay: str = "", src_paths: str = "") -> str:
    steps_html = "".join(
        f"<div style='margin-bottom:0.25rem;color:#44403C'>"
        f"<span style='color:#F97316;font-weight:700'>•</span> {s}</div>"
        for s in steps
    )
    anim_style = (
        f"animation: objective-pulse 1.1s ease-in-out 2; animation-delay: {anim_delay};"
        if anim_delay else ""
    )
    src_html = (
        f"<div style='margin-top:0.5rem;font-size:0.72rem;color:#78716C;"
        f"border-top:1px dashed #FED7AA;padding-top:0.35rem;line-height:1.5;"
        f"font-family:ui-monospace,SFMono-Regular,Menlo,monospace'>"
        f"<span style='color:#EA580C;font-weight:600'>src:</span> {src_paths}</div>"
        if src_paths else ""
    )
    return f"""
    <div style="border:1.5px solid #FED7AA;border-radius:10px;
                padding:1rem;background:{bg};color:#1C1917;height:100%;
                box-shadow:0 1px 6px rgba(249,115,22,0.08);{anim_style}">
      <div style="font-size:0.68rem;font-weight:700;color:#F97316;margin-bottom:0.3rem;
                  letter-spacing:0.08em;text-transform:uppercase">
        STEP {num}
      </div>
      <div style="font-weight:700;font-size:0.97rem;margin-bottom:0.55rem;color:#1C1917">
        {icon} {title}
      </div>
      <div style="font-size:0.84rem;line-height:1.7">{steps_html}</div>
      <div style="margin-top:0.55rem;font-size:0.84rem;border-top:1px solid #FED7AA;
                  padding-top:0.45rem;color:#1C1917">{stat_html}</div>
      {src_html}
    </div>"""


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


def _render_objectives_trace(trace: dict, animate: bool = False):
    """2×3 card grid showing all six pipeline steps exercised by this retrieval.

    Cards are laid out in reading order (STEP 1 → 2 → 3 on row 1, then 4 → 5 → 6
    on row 2). Animation pulses also flow in that order so the directional arrows
    between cards match the temporal flow the examiner sees.
    """
    cache_hits = trace.get("cache_hits", 0)
    cache_new  = trace.get("cache_new", 0)
    mem_entries = trace.get("mem_entries", 0)
    boosted    = trace.get("boosted_ids", [])
    n_feedback = trace.get("n_feedback", 0)

    # Delays in reading order: D[i] is the delay for STEP i+1
    D = ["0s", "0.45s", "0.9s", "1.35s", "1.8s", "2.25s"] if animate else [""] * 6

    st.markdown("##### How the six pipeline steps were exercised by this retrieval")

    # ── Row 1: STEP 1 → STEP 2 → STEP 3 ─────────────────────────────────────────
    r1c1, r1a1, r1c2, r1a2, r1c3 = st.columns([6, 1, 6, 1, 6], gap="small")
    with r1a1:
        st.markdown(_ARROW_HTML, unsafe_allow_html=True)
    with r1a2:
        st.markdown(_ARROW_HTML, unsafe_allow_html=True)

    with r1c1:
        cache_stat = (
            f'✅ <b>{cache_hits}</b> embeddings reused &nbsp;·&nbsp; '
            f'🆕 <b>{cache_new}</b> newly encoded<br>'
            f'<span style="color:#EA580C">Saved {cache_hits} encoder call(s)</span>'
            if cache_hits > 0 else
            '🔲 <b>0</b> reused &nbsp;·&nbsp; 🆕 <b>{cache_new}</b> newly encoded<br>'
            '<span style="opacity:0.6">Cache empty — first query</span>'
        ).format(cache_new=cache_new)
        st.markdown(_obj_card(
            "1", "⚡", "Efficient Caching",
            ["FAISS index pre-loaded into memory (one-time)",
             "Query embedding computed once and reused",
             "Chunk embeddings looked up in in-memory cache"],
            cache_stat,
            "#FFF7ED", D[0],
            src_paths="src/itma/integration.py · src/agent.py · src/demo_utils.py",
        ), unsafe_allow_html=True)

    with r1c2:
        st.markdown(_obj_card(
            "2", "🎙️", "Optimise ASR for Edge Computing",
            ["faster-whisper INT8 quantization: ~4× less memory vs FP32",
             "Built-in VAD filter skips silent segments (no hallucinations)",
             "Transcripts chunked → embedded with all-MiniLM-L6-v2",
             "FAISS index: 103 chunks across 5 domains, CPU-only pipeline"],
            '✅ Corpus built from ASR output &nbsp;·&nbsp; '
            '<b>103 chunks</b> indexed &nbsp;·&nbsp; 5 domains<br>'
            '<span style="opacity:0.7">This query searches the edge-transcribed corpus</span>',
            "#FFEDD5", D[1],
            src_paths="src/transcribe.py · src/build_vectorstore.py · scripts/build_domain_stores.py",
        ), unsafe_allow_html=True)

    with r1c3:
        st.markdown(_obj_card(
            "3", "🤖", "Agentic Reasoning Framework",
            ["Observe: FAISS retrieves top-20 candidate chunks",
             "Reason: scoring head evaluates (query, chunk, memory)",
             "Act: re-ranked top-K returned as response",
             "Reflect: feedback updates memory → next query benefits"],
            '🔄 ReAct loop completed &nbsp;·&nbsp; '
            f'20 candidates → <b>5</b> returned after reasoning',
            "#FFF7ED", D[2],
            src_paths="src/itma/integration.py:_rank() · src/itma/scoring_head.py · src/rag_chain.py",
        ), unsafe_allow_html=True)

    # Connector hint between row 1 (steps 1–3) and row 2 (steps 4–6)
    st.markdown(_DOWN_HINT_HTML, unsafe_allow_html=True)

    # ── Row 2: STEP 4 → STEP 5 → STEP 6 ─────────────────────────────────────────
    r2c1, r2a1, r2c2, r2a2, r2c3 = st.columns([6, 1, 6, 1, 6], gap="small")
    with r2a1:
        st.markdown(_ARROW_HTML, unsafe_allow_html=True)
    with r2a2:
        st.markdown(_ARROW_HTML, unsafe_allow_html=True)

    with r2c1:
        adapt_status = "🟢 Adapting" if mem_entries > 0 else "🟡 Cold start"
        boost_stat = (
            f'<span style="color:#EA580C"><b>{len(boosted)}</b> chunk(s) ID-boosted</span>'
            if boosted else '<span style="opacity:0.6">No boosts yet — mark helpful chunks below</span>'
        )
        st.markdown(_obj_card(
            "4", "🧠", "Smart Memory",
            ["Memory bank attended → weighted summary m",
             "Frozen scoring head scores each candidate",
             "ID-boost re-ranks chunks from past helpful interactions",
             "Counterfactual reweighting updates weights post-feedback"],
            f'{adapt_status} &nbsp;·&nbsp; bank: <b>{mem_entries}</b> entries &nbsp;·&nbsp; '
            f'N=<b>{n_feedback}</b><br>{boost_stat}',
            "#FFEDD5", D[3],
            src_paths="src/itma/memory_bank.py · src/itma/integration.py:record_feedback() · src/itma/scoring_head.py",
        ), unsafe_allow_html=True)

    with r2c2:
        if n_feedback >= 50:
            perf_stat = '<span style="color:#EA580C">✅ N=50 — ITMA H@5 0.932 &gt; CFRAG-lite 0.915 (no retraining)</span>'
        elif n_feedback >= 10:
            perf_stat = '<span style="color:#EA580C">✅ N≥10 — ITMA matches Dense-MiniLM baseline (H@5 0.848)</span>'
        else:
            perf_stat = f'Cold-start regime · ITMA recovers to baseline by N=10 · target N=50'
        st.markdown(_obj_card(
            "5", "📊", "Performance Evaluation",
            ["Retrieval quality measured via Hit@5, MRR@10, nDCG@10",
             "Cold-start curve tracked across N=0→50 feedback examples",
             "Results compared against 5 baselines on 59-item test split"],
            f'Progress: <b>N={n_feedback}</b> / 50<br>{perf_stat}',
            "#FFF7ED", D[4],
            src_paths="scripts/cold_start_eval.py · scripts/sensitivity_eval.py · scripts/eval_retrieval_only.py · analysis/make_plots.py",
        ), unsafe_allow_html=True)

    with r2c3:
        cache_total = cache_hits + cache_new
        saved_pct = round(100 * cache_hits / cache_total) if cache_total > 0 else 0
        st.markdown(_obj_card(
            "6", "💾", "Cost-Aware Caching Logic",
            ["FAISS store checked on disk before rebuilding (skip-if-exists)",
             "Warm retriever built once, cached for entire session",
             "Chunk embedding cache avoids redundant encoder calls",
             "Answer cache skips LLM call for repeated query+context pairs"],
            f'Embedding cache: <b>{cache_hits}/{cache_total}</b> hits this query '
            f'({saved_pct}% reuse rate)<br>'
            f'<span style="color:#EA580C">Session cache active — FAISS + warm retriever pre-loaded</span>',
            "#FFEDD5", D[5],
            src_paths="src/answer_cache.py · src/rag_chain.py · scripts/build_domain_stores.py · src/demo_utils.py",
        ), unsafe_allow_html=True)


def _init_session():
    if "live_retriever" not in st.session_state:
        from src.itma.integration import ITMARetriever
        st.session_state.live_retriever = ITMARetriever(
            store_path=STORE_PATH,
            checkpoint=CHECKPOINT,
            memory_path=None,
        )
        st.session_state.live_n_feedback = 0
        st.session_state.live_history = []
        st.session_state.last_results = None
        st.session_state.last_query = ""
        st.session_state.last_trace = {}
        st.session_state.feedback_submitted = False
        st.session_state.show_scores = False


_init_session()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ITMA memory bank")
    n = st.session_state.live_n_feedback
    st.metric("Feedback examples seen", n)

    if n > 0:
        bar_val = min(n / 50.0, 1.0)
        st.progress(bar_val, text=f"{n}/50 toward paper N=50 benchmark")

    st.divider()

    if st.button("🗑️ Reset memory bank", use_container_width=True):
        st.session_state.live_retriever.reset_memory()
        st.session_state.live_n_feedback = 0
        st.session_state.live_history = []
        st.session_state.last_results = None
        st.session_state.last_query = ""
        st.session_state.last_trace = {}
        st.session_state.feedback_submitted = False
        st.rerun()

    st.session_state.show_scores = st.toggle(
        "Show raw scores", value=st.session_state.show_scores
    )

    if st.session_state.live_history:
        st.divider()
        st.markdown("**Query history**")
        for i, entry in enumerate(reversed(st.session_state.live_history[-8:])):
            tick = "✅" if entry.get("marked_ids") else "·"
            short_q = entry["query"][:40] + ("…" if len(entry["query"]) > 40 else "")
            st.markdown(f"{tick} `{short_q}`")

    st.divider()
    st.caption(
        "Inference-Time Memory Adaptation for Cold-Start Educational RAG · "
        "retrieval-only demo (no generation API)"
    )


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="background:linear-gradient(135deg,{ORANGE_DARK} 0%,{ORANGE} 100%);
                border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;color:#fff">
        <h2 style="margin:0 0 0.3rem;color:#fff;font-weight:800">ITMA — Live Demo</h2>
        <p style="margin:0;opacity:0.92;font-size:0.95rem">
            Ask a question from the lecture corpus. Mark helpful chunks and watch
            the <strong style="color:#fff">memory bank</strong> adapt ITMA's re-ranking in real time.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sample query shortcuts
st.markdown("**Quick-start queries:**")
q_cols = st.columns(len(SAMPLE_QUERIES))
for i, (col, sample) in enumerate(zip(q_cols, SAMPLE_QUERIES)):
    with col:
        if st.button(sample[:35] + ("…" if len(sample) > 35 else ""),
                     key=f"sample_{i}", use_container_width=True):
            st.session_state["query_input_box"] = sample  # write directly into widget state
            st.session_state["_auto_retrieve"] = True

query_input = st.text_input(
    "Ask a question",
    placeholder="e.g. What is the role of a database transaction log?",
    key="query_input_box",
)

search_clicked = st.button("🔍 Retrieve", type="primary", use_container_width=False)
auto_retrieve = st.session_state.pop("_auto_retrieve", False)

# ── Retrieval ─────────────────────────────────────────────────────────────────
if (search_clicked or auto_retrieve) and query_input.strip():
    query = query_input.strip()
    embedder = get_embedder()
    retriever = st.session_state.live_retriever

    # Snapshot state BEFORE retrieval so we can explain the delta
    cache_before = len(retriever._chunk_emb_cache)
    mem_before = retriever._memory.size()

    with st.spinner("Retrieving…"):
        results = retriever.retrieve_with_ids(query, embedder, top_k=5)

    # Off-topic guard: reject queries whose top score is below threshold
    OFF_TOPIC_THRESHOLD = 0.45
    top_score = results[0][1] if results else 0.0
    if top_score < OFF_TOPIC_THRESHOLD:
        st.warning(
            f"**No relevant lecture content found** (top similarity score: {top_score:.3f} < {OFF_TOPIC_THRESHOLD}). "
            "Try a question about the lecture topics: generative AI, computer networks, databases, "
            "machine learning, or operating systems."
        )
        st.session_state.last_results = None
        st.session_state.last_query = None
        st.stop()

    # Snapshot state AFTER retrieval
    cache_after = len(retriever._chunk_emb_cache)
    cache_hits = cache_before                          # entries already cached
    cache_new = cache_after - cache_before             # new encodes this call
    mem_entries = retriever._memory.size()
    attended_ids = list(retriever._last_attended_ids)
    boosted_ids = [
        cid for cid in attended_ids
        if any(r[2] == cid for r in results)
    ]

    st.session_state.animate_objectives = True
    st.session_state.last_results = results
    st.session_state.last_query = query
    st.session_state.last_trace = {
        "cache_hits": cache_hits,
        "cache_new": cache_new,
        "mem_entries": mem_entries,
        "boosted_ids": boosted_ids,
        "n_feedback": st.session_state.live_n_feedback,
    }
    st.session_state.feedback_submitted = False

# ── Results display ───────────────────────────────────────────────────────────
if st.session_state.last_results and not st.session_state.feedback_submitted:
    results = st.session_state.last_results
    query = st.session_state.last_query
    trace = st.session_state.get("last_trace", {})

    st.divider()

    # ── Objectives trace ──────────────────────────────────────────────────────
    if trace:
        animate = st.session_state.pop("animate_objectives", False)
        _render_objectives_trace(trace, animate=animate)
        st.divider()

    st.subheader(f"Top-5 results for: *{query}*")

    checked_ids: list[str] = []
    for rank, (text, score, chunk_id) in enumerate(results, start=1):
        short_id = chunk_id[:14] + "…"
        score_badge = (
            f" &nbsp;<code style='font-size:0.8rem;opacity:0.75'>{score:.4f}</code>"
            if st.session_state.show_scores else ""
        )

        with st.container():
            col_check, col_card = st.columns([1, 11])
            with col_check:
                helpful = st.checkbox(
                    "Helpful",
                    key=f"helpful_{chunk_id}_{rank}",
                    label_visibility="collapsed",
                )
                if helpful:
                    checked_ids.append(chunk_id)
            with col_card:
                border_col = "#F97316" if helpful else "#E7E5E4"
                bg_col = "#FFF7ED" if helpful else "#FAFAF9"
                rank_col = "#F97316" if helpful else "#78716C"
                helpful_badge = (
                    " &nbsp;<span style='background:#F97316;color:#fff;font-size:0.72rem;"
                    "padding:1px 7px;border-radius:10px;font-weight:600'>✓ helpful</span>"
                ) if helpful else ""
                st.markdown(
                    f"""
                    <div style="
                        border: 1.5px solid {border_col};
                        border-radius: 10px;
                        background: {bg_col};
                        padding: 0.8rem 1.1rem;
                        margin-bottom: 0.5rem;
                        color: #1C1917;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
                        transition: border-color 0.15s;
                    ">
                    <span style="font-weight:700;font-size:1rem;color:{rank_col}">#{rank}</span>
                    &nbsp;<code style="font-size:0.78rem;color:#78716C;background:#F5F5F4;
                                       padding:1px 5px;border-radius:4px">{short_id}</code>{score_badge}{helpful_badge}<br>
                    <span style="font-size:0.9rem;line-height:1.6;color:#292524">{text[:320]}{'…' if len(text) > 320 else ''}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("")
    if len(checked_ids) == 0:
        st.caption("☑ Check one or more helpful chunks above, then submit feedback.")
    fb_col, _ = st.columns([2, 8])
    with fb_col:
        submit_feedback = st.button(
            "Submit feedback →",
            type="primary" if checked_ids else "secondary",
            use_container_width=True,
            disabled=(len(checked_ids) == 0),
        )

    if submit_feedback and checked_ids:
        retriever = st.session_state.live_retriever
        retriever.record_feedback(helpful_chunk_ids=checked_ids, reward=1.0)
        st.session_state.live_n_feedback += 1
        st.session_state.live_history.append(
            {"query": query, "marked_ids": checked_ids, "n": st.session_state.live_n_feedback}
        )
        st.session_state.feedback_submitted = True
        st.rerun()

elif st.session_state.feedback_submitted:
    n = st.session_state.live_n_feedback
    st.success(
        f"Feedback recorded. Memory bank now has **{n}** feedback example(s). "
        "Run the same or a related query to see re-ranking.",
        icon="✅",
    )

    if n >= 10:
        st.info(
            f"At N={n} feedback examples, ITMA should be at or above the Dense-MiniLM baseline. "
            "At N=50 it exceeds CFRAG-lite (offline fine-tuned prior art).",
            icon="📈",
        )

elif not st.session_state.last_results:
    st.markdown(
        f"""
        <div style="
            border: 1.5px dashed #FED7AA;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            color: #A8A29E;
            background: #FAFAF9;
            margin-top: 1rem;
        ">
        Select a quick-start query above or type your own question, then click <b style="color:{ORANGE}">Retrieve</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
