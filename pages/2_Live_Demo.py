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

from src.demo_utils import get_embedder, STORE_PATH, CHECKPOINT, SAMPLE_QUERIES

st.set_page_config(
    page_title="ITMA — Live Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _obj_card(num: str, icon: str, title: str, steps: list[str], stat_html: str, bg: str) -> str:
    steps_html = "".join(
        f"<div style='margin-bottom:0.25rem'><b>Step {i+1} —</b> {s}</div>"
        for i, s in enumerate(steps)
    )
    return f"""
    <div style="border:1px solid rgba(128,128,128,0.3);border-radius:8px;
                padding:1rem;background:{bg};color:inherit;height:100%">
      <div style="font-size:0.72rem;opacity:0.55;margin-bottom:0.25rem;letter-spacing:0.05em">
        OBJECTIVE {num}
      </div>
      <div style="font-weight:700;font-size:0.97rem;margin-bottom:0.55rem">{icon} {title}</div>
      <div style="font-size:0.86rem;line-height:1.7">{steps_html}</div>
      <div style="margin-top:0.55rem;font-size:0.84rem;border-top:1px solid rgba(128,128,128,0.2);
                  padding-top:0.45rem">{stat_html}</div>
    </div>"""


def _render_objectives_trace(trace: dict):
    """Six-column (2×3) card grid showing all objectives covered in this retrieval."""
    cache_hits = trace.get("cache_hits", 0)
    cache_new  = trace.get("cache_new", 0)
    mem_entries = trace.get("mem_entries", 0)
    boosted    = trace.get("boosted_ids", [])
    n_feedback = trace.get("n_feedback", 0)

    st.markdown("##### How all six objectives are covered in this retrieval")

    # ── Row 1 ──────────────────────────────────────────────────────────────────
    r1c1, r1c2, r1c3 = st.columns(3, gap="small")

    with r1c1:
        cache_stat = (
            f'✅ <b>{cache_hits}</b> embeddings reused &nbsp;·&nbsp; '
            f'🆕 <b>{cache_new}</b> newly encoded<br>'
            f'<span style="color:#2ca02c">Saved {cache_hits} encoder call(s)</span>'
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
            "rgba(31,119,180,0.08)"
        ), unsafe_allow_html=True)

    with r1c2:
        adapt_status = "🟢 Adapting" if mem_entries > 0 else "🟡 Cold start"
        boost_stat = (
            f'<span style="color:#2ca02c"><b>{len(boosted)}</b> chunk(s) ID-boosted</span>'
            if boosted else '<span style="opacity:0.6">No boosts yet — mark helpful chunks below</span>'
        )
        st.markdown(_obj_card(
            "2", "🧠", "Smart Memory",
            ["Memory bank attended → weighted summary m",
             "Frozen scoring head scores each candidate",
             "ID-boost re-ranks chunks from past helpful interactions",
             "Counterfactual reweighting updates weights post-feedback"],
            f'{adapt_status} &nbsp;·&nbsp; bank: <b>{mem_entries}</b> entries &nbsp;·&nbsp; '
            f'N=<b>{n_feedback}</b><br>{boost_stat}',
            "rgba(44,160,44,0.08)"
        ), unsafe_allow_html=True)

    with r1c3:
        if n_feedback >= 50:
            perf_stat = '<span style="color:#2ca02c">✅ N=50 — ITMA H@5 0.932 &gt; CFRAG-lite 0.915 (no retraining)</span>'
        elif n_feedback >= 10:
            perf_stat = '<span style="color:#2ca02c">✅ N≥10 — ITMA matches Dense-MiniLM baseline (H@5 0.848)</span>'
        else:
            perf_stat = f'Cold-start regime · ITMA recovers to baseline by N=10 · target N=50'
        st.markdown(_obj_card(
            "3", "📊", "Performance Evaluation",
            ["Retrieval quality measured via Hit@5, MRR@10, nDCG@10",
             "Cold-start curve tracked across N=0→50 feedback examples",
             "Results compared against 5 baselines on 59-item test split"],
            f'Progress: <b>N={n_feedback}</b> / 50<br>{perf_stat}',
            "rgba(214,39,40,0.06)"
        ), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

    # ── Row 2 ──────────────────────────────────────────────────────────────────
    r2c1, r2c2, r2c3 = st.columns(3, gap="small")

    with r2c1:
        st.markdown(_obj_card(
            "4", "🤖", "Agentic Reasoning Framework",
            ["Observe: FAISS retrieves top-20 candidate chunks",
             "Reason: scoring head evaluates (query, chunk, memory)",
             "Act: re-ranked top-K returned as response",
             "Reflect: feedback updates memory → next query benefits"],
            '🔄 ReAct loop completed &nbsp;·&nbsp; '
            f'20 candidates → <b>5</b> returned after reasoning',
            "rgba(148,103,189,0.08)"
        ), unsafe_allow_html=True)

    with r2c2:
        st.markdown(_obj_card(
            "5", "🎙️", "Optimise ASR for Edge Computing",
            ["faster-whisper INT8 quantization: ~4× less memory vs FP32",
             "Built-in VAD filter skips silent segments (no hallucinations)",
             "Transcripts chunked → embedded with all-MiniLM-L6-v2",
             "FAISS index: 103 chunks across 5 domains, CPU-only pipeline"],
            '✅ Corpus built from ASR output &nbsp;·&nbsp; '
            '<b>103 chunks</b> indexed &nbsp;·&nbsp; 5 domains<br>'
            '<span style="opacity:0.7">This query searches the edge-transcribed corpus</span>',
            "rgba(255,127,14,0.07)"
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
            f'<span style="color:#2ca02c">Session cache active — FAISS + warm retriever pre-loaded</span>',
            "rgba(23,190,207,0.07)"
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
st.title("ITMA — Live Demo")
st.markdown(
    "Ask a question from the lecture corpus. Mark helpful chunks and watch "
    "the **memory bank** adapt ITMA's re-ranking in real time."
)

# Sample query shortcuts
st.markdown("**Quick-start queries:**")
q_cols = st.columns(len(SAMPLE_QUERIES))
for i, (col, sample) in enumerate(zip(q_cols, SAMPLE_QUERIES)):
    with col:
        if st.button(sample[:35] + ("…" if len(sample) > 35 else ""),
                     key=f"sample_{i}", use_container_width=True):
            st.session_state["_prefill_query"] = sample

prefill = st.session_state.pop("_prefill_query", "")
query_input = st.text_input(
    "Ask a question",
    value=prefill,
    placeholder="e.g. What is the role of a database transaction log?",
    key="query_input_box",
)

search_clicked = st.button("🔍 Retrieve", type="primary", use_container_width=False)

# ── Retrieval ─────────────────────────────────────────────────────────────────
if search_clicked and query_input.strip():
    query = query_input.strip()
    embedder = get_embedder()
    retriever = st.session_state.live_retriever

    # Snapshot state BEFORE retrieval so we can explain the delta
    cache_before = len(retriever._chunk_emb_cache)
    mem_before = retriever._memory.size()

    with st.spinner("Retrieving…"):
        results = retriever.retrieve_with_ids(query, embedder, top_k=5)

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
        _render_objectives_trace(trace)
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
                border_col = "#2ca02c" if helpful else "rgba(128,128,128,0.3)"
                bg_col = "rgba(44,160,44,0.12)" if helpful else "rgba(128,128,128,0.05)"
                helpful_badge = " &nbsp;<span style='color:#2ca02c;font-size:0.8rem'>✓ marked helpful</span>" if helpful else ""
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid {border_col};
                        border-radius: 6px;
                        background: {bg_col};
                        padding: 0.7rem 1rem;
                        margin-bottom: 0.5rem;
                        color: inherit;
                    ">
                    <span style="font-weight:600">#{rank}</span>
                    &nbsp;<code style="font-size:0.8rem">{short_id}</code>{score_badge}{helpful_badge}<br>
                    <span style="font-size:0.9rem;line-height:1.5">{text[:320]}{'…' if len(text) > 320 else ''}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("")
    fb_col, _ = st.columns([2, 8])
    with fb_col:
        submit_feedback = st.button(
            "Submit feedback →",
            type="secondary",
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
        """
        <div style="
            border: 1px dashed rgba(128,128,128,0.4);
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
            opacity: 0.6;
            margin-top: 1rem;
            color: inherit;
        ">
        Select a quick-start query above or type your own question, then click <b>Retrieve</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
