"""ITMA Streamlit Demo — entry point.

Run with:
    streamlit run demo.py
"""

import streamlit as st
from src.demo_utils import inject_global_css, ORANGE, ORANGE_DARK, ORANGE_BG, ORANGE_200

st.set_page_config(
    page_title="ITMA — Inference-Time Memory Adaptation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_global_css()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, {ORANGE_DARK} 0%, {ORANGE} 60%, #FB923C 100%);
        border-radius: 16px;
        padding: 3rem 2.5rem 2.5rem;
        margin-bottom: 2rem;
        color: #fff;
        box-shadow: 0 8px 32px rgba(249,115,22,0.25);
    ">
        <div style="font-size:0.85rem;font-weight:600;letter-spacing:0.12em;
                    opacity:0.85;margin-bottom:0.5rem;text-transform:uppercase">
            Journal Paper Demo
        </div>
        <h1 style="margin:0 0 0.6rem;font-size:2.1rem;font-weight:800;line-height:1.2;color:#fff">
            Inference-Time Memory Adaptation<br>for Cold-Start Educational RAG
        </h1>
        <div style="font-size:1.15rem;font-weight:600;opacity:0.9;margin-bottom:1.2rem">ITMA</div>
        <p style="font-size:1rem;line-height:1.7;opacity:0.92;max-width:720px;margin:0">
            A scoring head pretrained once and <strong style="color:#fff">frozen forever</strong>.
            All per-deployment adaptation flows through an <strong style="color:#fff">online memory bank</strong>
            updated with counterfactual-reweighted feedback — no retraining, no LLM calls in the loop.
            Starts within 2% of a strong dense baseline at N=0, and
            <strong style="color:#fff">exceeds CFRAG-lite by N=50</strong>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Nav cards ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        f"""
        <div style="
            border: 2px solid {ORANGE_200};
            border-radius: 12px;
            padding: 1.8rem;
            background: {ORANGE_BG};
            height: 100%;
            box-shadow: 0 2px 12px rgba(249,115,22,0.08);
        ">
            <div style="font-size:2rem;margin-bottom:0.6rem">📖</div>
            <h3 style="margin:0 0 0.5rem;color:{ORANGE_DARK};font-weight:700">Simulation</h3>
            <p style="margin:0 0 1.2rem;color:#44403C;line-height:1.6">
                Stepped walkthrough of the ITMA story.
                Same query at <strong>N=0</strong> vs <strong>N=50</strong> side-by-side,
                then the cold-start curve and Table 1 results.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Simulation.py", label="Open Simulation →", icon="📖")

with col2:
    st.markdown(
        f"""
        <div style="
            border: 2px solid {ORANGE_200};
            border-radius: 12px;
            padding: 1.8rem;
            background: {ORANGE_BG};
            height: 100%;
            box-shadow: 0 2px 12px rgba(249,115,22,0.08);
        ">
            <div style="font-size:2rem;margin-bottom:0.6rem">🔬</div>
            <h3 style="margin:0 0 0.5rem;color:{ORANGE_DARK};font-weight:700">Live Demo</h3>
            <p style="margin:0 0 1.2rem;color:#44403C;line-height:1.6">
                Interactive retriever. Ask any lecture question, mark helpful chunks,
                and watch ITMA's <strong>memory bank</strong> re-rank results in real time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/2_Live_Demo.py", label="Open Live Demo →", icon="🔬")

# ── Stats row ────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

s1, s2, s3, s4, s5 = st.columns(5)
stats = [
    ("291", "QA pairs"),
    ("5", "domains"),
    ("59", "test items"),
    ("N=50", "ITMA beats CFRAG-lite"),
    ("0", "retraining calls"),
]
for col, (val, label) in zip([s1, s2, s3, s4, s5], stats):
    with col:
        st.markdown(
            f"""
            <div style="text-align:center;padding:1rem 0.5rem;
                        border:1px solid {ORANGE_200};border-radius:10px;
                        background:#fff">
                <div style="font-size:1.6rem;font-weight:800;color:{ORANGE_DARK}">{val}</div>
                <div style="font-size:0.78rem;color:#78716C;margin-top:0.2rem">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    "<div style='margin-top:2rem;text-align:center;color:#A8A29E;font-size:0.8rem'>"
    "LectureRAG-75 · retrieval-only demo (no generation API required)"
    "</div>",
    unsafe_allow_html=True,
)
