"""ITMA — Upload-and-Query video page.

Examiners upload any video file; the system transcribes it, builds a FAISS
index, and exposes a query + feedback UI identical in spirit to the static
Live Demo page (`pages/2_Live_Demo.py`) — but pointed at the user's own
content rather than the pretrained NPTEL corpus.

State is per browser session via st.session_state (keys prefixed `upload_`).
"""

from __future__ import annotations

import os
import pickle
import subprocess
import time
from datetime import datetime
from typing import Optional

import streamlit as st

from src.demo_utils import (
    CHECKPOINT,
    ORANGE,
    ORANGE_DARK,
    get_embedder,
    inject_global_css,
)

# ---------------------------------------------------------------------------
# Page config + global styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ITMA — Upload Video",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()


UPLOADS_ROOT = "data/user_uploads"
ACCEPTED_EXTS = ["mp4", "mkv", "mov", "webm", "m4v"]


def _fmt_ts(seconds: float) -> str:
    """Format seconds as MM:SS for the per-result timestamp badge."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Six-step trace card (adapted from pages/2_Live_Demo.py for uploaded content)
# ---------------------------------------------------------------------------

def _step_card(num: str, icon: str, title: str, items: list[str], stat_html: str,
               bg: str, anim_delay: str = "", src_paths: str = "") -> str:
    items_html = "".join(
        f"<div style='margin-bottom:0.25rem;color:#44403C'>"
        f"<span style='color:#F97316;font-weight:700'>•</span> {s}</div>"
        for s in items
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
      <div style="font-size:0.84rem;line-height:1.7">{items_html}</div>
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


def _render_steps_trace(trace: dict, animate: bool = False) -> None:
    """2x3 card grid showing all six pipeline steps exercised by this retrieval,
    contextualised for the user's uploaded video.

    Cards are laid out in reading order (STEP 1 → 2 → 3 on row 1, then 4 → 5 → 6
    on row 2). Animation pulses also flow in that order so the directional arrows
    between cards match the temporal flow.
    """
    cache_hits = trace.get("cache_hits", 0)
    cache_new = trace.get("cache_new", 0)
    mem_entries = trace.get("mem_entries", 0)
    boosted = trace.get("boosted_ids", [])
    n_feedback = trace.get("n_feedback", 0)
    n_chunks = trace.get("n_chunks", 0)
    video_name = trace.get("video_name", "your video")

    # D[i] is the delay for STEP (i+1), matching visual reading order
    D = ["0s", "0.45s", "0.9s", "1.35s", "1.8s", "2.25s"] if animate else [""] * 6

    st.markdown("##### How the six pipeline steps were exercised by this retrieval")

    # Row 1: STEP 1 (Cache) → STEP 2 (ASR/Corpus) → STEP 3 (ReAct)
    r1c1, r1a1, r1c2, r1a2, r1c3 = st.columns([6, 1, 6, 1, 6], gap="small")
    with r1a1:
        st.markdown(_ARROW_HTML, unsafe_allow_html=True)
    with r1a2:
        st.markdown(_ARROW_HTML, unsafe_allow_html=True)
    with r1c1:
        cache_total = cache_hits + cache_new
        if cache_hits > 0:
            cache_stat = (
                f'✅ <b>{cache_hits}</b> embedding(s) reused &nbsp;·&nbsp; '
                f'🆕 <b>{cache_new}</b> newly encoded<br>'
                f'<span style="color:#EA580C">Saved {cache_hits} encoder call(s)</span>'
            )
        else:
            cache_stat = (
                f'🔲 <b>0</b> reused &nbsp;·&nbsp; 🆕 <b>{cache_new}</b> newly encoded<br>'
                f'<span style="opacity:0.6">Cache empty — first query on this video</span>'
            )
        st.markdown(_step_card(
            "1", "⚡", "Efficient Caching",
            ["FAISS index built once at upload and held in memory",
             "Query embedding computed once and reused",
             "Chunk embeddings looked up in in-memory cache"],
            cache_stat,
            "#FFF7ED", D[0],
            src_paths="src/itma/integration.py · src/agent.py · src/demo_utils.py",
        ), unsafe_allow_html=True)

    with r1c2:
        st.markdown(_step_card(
            "2", "🎙️", "Optimised ASR for Edge Computing",
            ["faster-whisper INT8 quantisation: ~4× less memory vs FP32",
             "Built-in VAD filter skips silent segments (no hallucinations)",
             "Transcript chunked → embedded with all-MiniLM-L6-v2",
             f"FAISS index built on-device from <b>{n_chunks}</b> chunk(s) of {video_name}"],
            '✅ Corpus built from ASR output of your upload &nbsp;·&nbsp; '
            f'<b>{n_chunks} chunk(s)</b> indexed<br>'
            '<span style="opacity:0.7">This query searches the edge-transcribed corpus</span>',
            "#FFEDD5", D[1],
            src_paths="src/transcribe.py · src/build_vectorstore.py · src/video_utils.py · src/upload_pipeline.py",
        ), unsafe_allow_html=True)

    with r1c3:
        st.markdown(_step_card(
            "3", "🤖", "Agentic Reasoning Framework",
            ["Observe: FAISS retrieves top-20 candidate chunks",
             "Reason: scoring head evaluates (query, chunk, memory)",
             "Act: re-ranked top-5 returned as response",
             "Reflect: feedback updates memory → next query benefits"],
            '🔄 ReAct loop completed &nbsp;·&nbsp; '
            f'20 candidates → <b>5</b> returned after re-ranking',
            "#FFF7ED", D[2],
            src_paths="src/itma/integration.py:_rank() · src/itma/scoring_head.py · src/rag_chain.py",
        ), unsafe_allow_html=True)

    # Connector hint between row 1 (steps 1–3) and row 2 (steps 4–6)
    st.markdown(_DOWN_HINT_HTML, unsafe_allow_html=True)

    # Row 2: STEP 4 (Smart Memory) → STEP 5 (Performance Eval) → STEP 6 (Cost-aware caching)
    r2c1, r2a1, r2c2, r2a2, r2c3 = st.columns([6, 1, 6, 1, 6], gap="small")
    with r2a1:
        st.markdown(_ARROW_HTML, unsafe_allow_html=True)
    with r2a2:
        st.markdown(_ARROW_HTML, unsafe_allow_html=True)

    with r2c1:
        adapt_status = "🟢 Adapting" if mem_entries > 0 else "🟡 Cold start"
        boost_stat = (
            f'<span style="color:#EA580C"><b>{len(boosted)}</b> chunk(s) ID-boosted</span>'
            if boosted else
            '<span style="opacity:0.6">No boosts yet — mark helpful chunks below</span>'
        )
        st.markdown(_step_card(
            "4", "🧠", "Smart Memory",
            ["Memory bank attended → weighted summary m",
             "Frozen scoring head scores each candidate",
             "ID-boost re-ranks chunks marked helpful in earlier queries",
             "Counterfactual reweighting updates weights post-feedback"],
            f'{adapt_status} &nbsp;·&nbsp; bank: <b>{mem_entries}</b> entries &nbsp;·&nbsp; '
            f'N=<b>{n_feedback}</b><br>{boost_stat}',
            "#FFEDD5", D[3],
            src_paths="src/itma/memory_bank.py · src/itma/integration.py:record_feedback() · src/itma/scoring_head.py",
        ), unsafe_allow_html=True)

    with r2c2:
        if n_feedback >= 50:
            perf_stat = (
                '<span style="color:#EA580C">✅ N=50 — full cold-start trajectory '
                'covered on your uploaded video</span>'
            )
        elif n_feedback >= 10:
            perf_stat = (
                '<span style="color:#EA580C">✅ N≥10 — ITMA enters adaptation regime; '
                'expect further gain through N=50</span>'
            )
        else:
            perf_stat = (
                'Cold-start regime · scoring head + boost engaged · '
                'mark helpful chunks to populate memory'
            )
        st.markdown(_step_card(
            "5", "📊", "Performance Evaluation",
            ["Retrieval quality measurable via Hit@5, MRR@10, nDCG@10, R@10",
             "Cold-start trajectory replicates the protocol from §6 of the paper",
             f"Index has <b>{n_chunks}</b> chunk(s) — large enough for meaningful re-ranking",
             "Same architecture & checkpoint as the LectureRAG-75 evaluation"],
            f'Progress: <b>N={n_feedback}</b> / 50<br>{perf_stat}',
            "#FFF7ED", D[4],
            src_paths="scripts/cold_start_eval.py · scripts/sensitivity_eval.py · scripts/eval_retrieval_only.py · analysis/make_plots.py",
        ), unsafe_allow_html=True)

    with r2c3:
        saved_pct = round(100 * cache_hits / (cache_hits + cache_new)) if (cache_hits + cache_new) > 0 else 0
        st.markdown(_step_card(
            "6", "💾", "Cost-Aware Caching Logic",
            ["FAISS store written to per-session directory and reused across queries",
             "Retriever object cached in st.session_state for the browser session",
             "Chunk embedding cache avoids redundant encoder calls",
             "No LLM in the loop — every adaptation step is gradient-free"],
            f'Embedding cache: <b>{cache_hits}/{cache_hits + cache_new}</b> hits this query '
            f'({saved_pct}% reuse rate)<br>'
            f'<span style="color:#EA580C">Session-scoped FAISS index active</span>',
            "#FFEDD5", D[5],
            src_paths="src/answer_cache.py · src/upload_pipeline.py · src/demo_utils.py",
        ), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

_UPLOAD_KEYS = [
    "upload_session_dir",
    "upload_video_name",
    "upload_retriever",
    "upload_timestamp_by_id",
    "upload_chunk_count",
    "upload_n_feedback",
    "upload_history",
    "upload_last_results",
    "upload_last_query",
    "upload_last_trace",
    "upload_animate_steps",
    "upload_feedback_submitted",
    "upload_processing",
]


def _init_session() -> None:
    for k in _UPLOAD_KEYS:
        if k not in st.session_state:
            st.session_state[k] = None
    if st.session_state.upload_n_feedback is None:
        st.session_state.upload_n_feedback = 0
    if st.session_state.upload_history is None:
        st.session_state.upload_history = []
    if st.session_state.upload_processing is None:
        st.session_state.upload_processing = False
    if st.session_state.upload_feedback_submitted is None:
        st.session_state.upload_feedback_submitted = False


def _reset_upload_state() -> None:
    """Drop the retriever (and its memory bank) and clear all upload state.

    Files on disk under data/user_uploads/<ts>/ are left in place; manual
    cleanup if disk pressure matters.
    """
    for k in _UPLOAD_KEYS:
        if k in st.session_state:
            del st.session_state[k]
    _init_session()


_init_session()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ITMA memory bank")
    n = st.session_state.upload_n_feedback or 0
    st.metric("Feedback examples seen", n)

    if n > 0:
        st.progress(min(n / 50.0, 1.0), text=f"{n}/50 toward paper N=50 benchmark")

    st.divider()

    if st.session_state.upload_retriever is not None:
        if st.button("🗑️ Reset memory bank", use_container_width=True):
            st.session_state.upload_retriever.reset_memory()
            st.session_state.upload_n_feedback = 0
            st.session_state.upload_history = []
            st.session_state.upload_last_results = None
            st.session_state.upload_last_query = None
            st.session_state.upload_feedback_submitted = False
            st.rerun()

        if st.button("⬆️ Upload a different video", use_container_width=True):
            _reset_upload_state()
            st.rerun()

    if st.session_state.upload_history:
        st.divider()
        st.markdown("**Query history**")
        for entry in reversed(st.session_state.upload_history[-8:]):
            tick = "✅" if entry.get("marked_ids") else "·"
            short_q = entry["query"][:40] + ("…" if len(entry["query"]) > 40 else "")
            st.markdown(f"{tick} `{short_q}`")

    st.divider()
    st.caption(
        "**ITMA — Upload & Query**\n\n"
        "Whisper `small` · faster-whisper INT8+VAD · cpu-only\n\n"
        "Processing time ≈ video duration. Use clips under 10 min for the live demo."
    )


# ---------------------------------------------------------------------------
# Main — hero header
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div style="background:linear-gradient(135deg,{ORANGE_DARK} 0%,{ORANGE} 100%);
                border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;color:#fff">
        <h2 style="margin:0 0 0.3rem;color:#fff;font-weight:800">
            ITMA — Upload Your Own Lecture
        </h2>
        <p style="margin:0;opacity:0.92;font-size:0.95rem">
            Upload any lecture video. ITMA transcribes it locally, builds a fresh
            retrieval index, and lets you query and give feedback in real time —
            verifying that the pretrained head generalises to <strong style="color:#fff">unseen
            content</strong>, not just the corpus it was trained on.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Stage 1 — Upload + Process (only shown when no index has been built yet)
# ---------------------------------------------------------------------------

def _run_processing(video_path: str, session_dir: str) -> None:
    """Run the ingest pipeline inside an st.status block, with error handling."""
    from src.upload_pipeline import process_video
    from src.video_utils import ffmpeg_available

    if not ffmpeg_available():
        st.error(
            "**ffmpeg not found on PATH.**  Install ffmpeg "
            "(`winget install ffmpeg` on Windows, `brew install ffmpeg` on macOS, "
            "`apt install ffmpeg` on Debian/Ubuntu) and restart this Streamlit app."
        )
        st.session_state.upload_processing = False
        return

    status_obj = st.status("Processing video…", expanded=True)
    progress_line = status_obj.empty()
    elapsed_line = status_obj.empty()
    t_start = time.time()

    def cb(stage: str, frac: float) -> None:
        progress_line.markdown(f"**{stage}** — {int(frac * 100)}%")
        elapsed_line.caption(f"Elapsed: {time.time() - t_start:.1f}s")

    try:
        process_video(video_path, session_dir, progress_cb=cb)
    except FileNotFoundError as e:
        status_obj.update(label="ffmpeg missing", state="error")
        st.error(f"**ffmpeg not found.** {e}")
        st.session_state.upload_processing = False
        return
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="replace")[-500:]
        status_obj.update(label="Audio extraction failed", state="error")
        st.error(
            "**Could not decode video** (likely an unsupported codec or a corrupt "
            f"file).\n\n```\n{stderr.strip()}\n```"
        )
        st.session_state.upload_processing = False
        return
    except RuntimeError as e:
        status_obj.update(label="Empty transcript", state="error")
        st.warning(f"**{e}**  Try a video with clearly audible narration.")
        st.session_state.upload_processing = False
        return
    except OSError as e:
        status_obj.update(label="Disk error", state="error")
        st.error(
            f"**Could not write index** (disk full or permission denied).\n\n`{e}`"
        )
        st.session_state.upload_processing = False
        return
    except Exception as e:  # noqa: BLE001 — surface anything else to the UI
        status_obj.update(label=f"Failed: {type(e).__name__}", state="error")
        st.error(f"**Processing failed:** {type(e).__name__}: {e}")
        st.session_state.upload_processing = False
        return

    status_obj.update(label="Done", state="complete", expanded=False)

    # Build the retriever and prime the timestamp lookup table
    from src.agent import make_chunk_id
    from src.itma.integration import ITMARetriever

    with open(os.path.join(session_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    timestamps = meta.get("timestamps") or [None] * len(meta["chunks"])
    timestamp_by_id = {
        make_chunk_id(c): ts for c, ts in zip(meta["chunks"], timestamps)
    }

    retriever = ITMARetriever(
        store_path=session_dir,
        checkpoint=CHECKPOINT,
        memory_path=None,
    )

    st.session_state.upload_session_dir = session_dir
    st.session_state.upload_retriever = retriever
    st.session_state.upload_timestamp_by_id = timestamp_by_id
    st.session_state.upload_chunk_count = len(meta["chunks"])
    st.session_state.upload_n_feedback = 0
    st.session_state.upload_history = []
    st.session_state.upload_last_results = None
    st.session_state.upload_last_query = None
    st.session_state.upload_feedback_submitted = False
    st.session_state.upload_processing = False
    st.rerun()


if st.session_state.upload_session_dir is None:
    st.subheader("1. Upload a lecture video")
    uploaded = st.file_uploader(
        "Drop an MP4, MKV, MOV, WEBM or M4V file (≤ 200 MB)",
        type=ACCEPTED_EXTS,
        accept_multiple_files=False,
        disabled=st.session_state.upload_processing,
    )

    if uploaded is not None:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.caption(
                f"**{uploaded.name}** &mdash; {uploaded.size / 1_000_000:.1f} MB"
            )
        with col_b:
            process_clicked = st.button(
                "🚀 Process",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.upload_processing,
            )

        if process_clicked:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            session_dir = os.path.join(UPLOADS_ROOT, ts)
            os.makedirs(session_dir, exist_ok=True)
            ext = os.path.splitext(uploaded.name)[1].lower() or ".bin"
            video_path = os.path.join(session_dir, f"video{ext}")
            with open(video_path, "wb") as f:
                f.write(uploaded.getbuffer())

            st.session_state.upload_processing = True
            st.session_state.upload_video_name = uploaded.name
            _run_processing(video_path, session_dir)


# ---------------------------------------------------------------------------
# Stage 2 — Query + feedback (only when a retriever is loaded)
# ---------------------------------------------------------------------------

if st.session_state.upload_session_dir is not None and st.session_state.upload_retriever is not None:
    n_chunks = st.session_state.get("upload_chunk_count") or 0
    st.success(
        f"Index ready for **{st.session_state.upload_video_name}** — "
        f"**{n_chunks} chunk{'s' if n_chunks != 1 else ''}** indexed.",
        icon="✅",
    )

    if n_chunks <= 2:
        st.warning(
            f"Only {n_chunks} chunk{'s' if n_chunks != 1 else ''} were extracted "
            "from this video. Retrieval can't differentiate between queries when the "
            "index is this small — every query will return the same chunk(s). "
            "**Try uploading a longer clip (≥ 30 seconds of speech) for a meaningful demo.**",
            icon="⚠️",
        )

    query_input = st.text_input(
        "Ask a question about the uploaded lecture",
        placeholder="e.g. What was the main definition introduced in the first half?",
        key="upload_query_input",
    )
    search_clicked = st.button("🔍 Retrieve", type="primary")

    if search_clicked and query_input.strip():
        query = query_input.strip()
        embedder = get_embedder()
        retriever = st.session_state.upload_retriever

        # Snapshot BEFORE retrieval so we can explain the cache/memory delta
        cache_before = len(retriever._chunk_emb_cache)

        with st.spinner("Retrieving…"):
            results = retriever.retrieve_with_ids(query, embedder, top_k=5)

        # Snapshot AFTER retrieval
        cache_after = len(retriever._chunk_emb_cache)
        cache_hits = cache_before
        cache_new = cache_after - cache_before
        mem_entries = retriever._memory.size()
        attended_ids = list(getattr(retriever, "_last_attended_ids", []) or [])
        boosted_ids = [
            cid for cid in attended_ids
            if any(r[2] == cid for r in results)
        ]

        OFF_TOPIC_THRESHOLD = 0.30  # lower than Live Demo — user content varies
        top_score = results[0][1] if results else 0.0
        if top_score < OFF_TOPIC_THRESHOLD:
            st.warning(
                f"**No strongly relevant content found** (top similarity score: "
                f"{top_score:.3f} < {OFF_TOPIC_THRESHOLD}).  Try a question more "
                "directly related to the uploaded video's content."
            )
            st.session_state.upload_last_results = None
            st.session_state.upload_last_query = None
            st.session_state.upload_last_trace = None
        else:
            st.session_state.upload_last_results = results
            st.session_state.upload_last_query = query
            st.session_state.upload_last_trace = {
                "cache_hits": cache_hits,
                "cache_new": cache_new,
                "mem_entries": mem_entries,
                "boosted_ids": boosted_ids,
                "n_feedback": st.session_state.upload_n_feedback or 0,
                "n_chunks": st.session_state.get("upload_chunk_count") or 0,
                "video_name": st.session_state.upload_video_name or "your video",
            }
            st.session_state.upload_animate_steps = True
            st.session_state.upload_feedback_submitted = False

    # ── Render results ──────────────────────────────────────────────────────
    if (
        st.session_state.upload_last_results
        and not st.session_state.upload_feedback_submitted
    ):
        results = st.session_state.upload_last_results
        query = st.session_state.upload_last_query
        ts_by_id = st.session_state.upload_timestamp_by_id or {}
        trace = st.session_state.get("upload_last_trace") or {}

        st.divider()

        # Six-step trace grid (animate only on the first render after a new query)
        if trace:
            animate = bool(st.session_state.get("upload_animate_steps", False))
            st.session_state.upload_animate_steps = False
            _render_steps_trace(trace, animate=animate)
            st.divider()

        st.subheader(f"Top-{len(results)} results for: *{query}*")

        checked_ids: list[str] = []
        for rank, (text, score, chunk_id) in enumerate(results, start=1):
            short_id = chunk_id[:14] + "…"
            ts = ts_by_id.get(chunk_id)
            if ts is not None:
                start_s, end_s = ts
                ts_badge = (
                    f"<code style='font-size:0.75rem;color:#1C1917;background:#FED7AA;"
                    f"padding:1px 6px;border-radius:4px;margin-left:6px'>"
                    f"{_fmt_ts(start_s)} → {_fmt_ts(end_s)}</code>"
                )
            else:
                ts_badge = ""

            with st.container():
                col_check, col_card = st.columns([1, 11])
                with col_check:
                    helpful = st.checkbox(
                        "Helpful",
                        key=f"upload_helpful_{chunk_id}_{rank}",
                        label_visibility="collapsed",
                    )
                    if helpful:
                        checked_ids.append(chunk_id)
                with col_card:
                    border_col = "#F97316" if helpful else "#E7E5E4"
                    bg_col = "#FFF7ED" if helpful else "#FAFAF9"
                    rank_col = "#F97316" if helpful else "#78716C"
                    score_badge = (
                        f"&nbsp;<code style='font-size:0.8rem;opacity:0.75'>{score:.4f}</code>"
                    )
                    helpful_badge = (
                        "&nbsp;<span style='background:#F97316;color:#fff;font-size:0.72rem;"
                        "padding:1px 7px;border-radius:10px;font-weight:600'>✓ helpful</span>"
                    ) if helpful else ""
                    st.markdown(
                        f"""
                        <div style="border:1.5px solid {border_col};border-radius:10px;
                                    background:{bg_col};padding:0.8rem 1.1rem;
                                    margin-bottom:0.5rem;color:#1C1917;
                                    box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                        <span style="font-weight:700;font-size:1rem;color:{rank_col}">
                            #{rank}
                        </span>
                        &nbsp;<code style="font-size:0.78rem;color:#78716C;background:#F5F5F4;
                                           padding:1px 5px;border-radius:4px">{short_id}</code>
                        {score_badge}{ts_badge}{helpful_badge}<br>
                        <span style="font-size:0.9rem;line-height:1.6;color:#292524">
                            {text[:400]}{'…' if len(text) > 400 else ''}
                        </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.markdown("")
        if not checked_ids:
            st.caption("☑ Check one or more helpful chunks above, then submit feedback.")
        fb_col, _ = st.columns([2, 8])
        with fb_col:
            submit_feedback = st.button(
                "Submit feedback →",
                type="primary" if checked_ids else "secondary",
                use_container_width=True,
                disabled=not checked_ids,
            )

        if submit_feedback and checked_ids:
            retriever = st.session_state.upload_retriever
            retriever.record_feedback(helpful_chunk_ids=checked_ids, reward=1.0)
            st.session_state.upload_n_feedback += 1
            st.session_state.upload_history.append(
                {
                    "query": query,
                    "marked_ids": checked_ids,
                    "n": st.session_state.upload_n_feedback,
                }
            )
            st.session_state.upload_feedback_submitted = True
            st.rerun()

    elif st.session_state.upload_feedback_submitted:
        n = st.session_state.upload_n_feedback
        st.success(
            f"Feedback recorded. Memory bank now has **{n}** feedback example(s). "
            "Run the same or a related query to see ITMA re-rank.",
            icon="✅",
        )


