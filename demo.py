"""ITMA Streamlit Demo — entry point.

Run with:
    streamlit run demo.py
"""

import streamlit as st

st.set_page_config(
    page_title="ITMA — Inference-Time Memory Adaptation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Inference-Time Memory Adaptation for Cold-Start Educational RAG")
st.caption("**ITMA** — journal paper demo")

st.markdown(
    """
    ITMA is a retrieval framework for educational lecture corpora that **adapts at inference time**
    without retraining. A scoring head is pretrained once and frozen permanently; all per-deployment
    adaptation happens through an **online memory bank** updated with counterfactual-reweighted
    feedback signals after each query.

    The key property: ITMA starts within 2% of a strong dense baseline at N=0 feedback examples,
    recovers fully by N=10, and **exceeds CFRAG-lite** (an offline fine-tuned prior art) by N=50 —
    with **no retraining at deployment**.
    """
)

st.divider()

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📖 Simulation walkthrough")
    st.markdown(
        "Stepped narrative showing how ITMA adapts. Runs the same query at **N=0** "
        "and **N=50** side-by-side, then presents the cold-start curve and Table 1."
    )
    st.page_link("pages/1_Simulation.py", label="Open simulation →", icon="📖")

with col2:
    st.subheader("🔬 Live demo")
    st.markdown(
        "Interactive retriever. Ask any question from the lecture corpus, mark helpful "
        "chunks, and watch ITMA re-rank in real time as its **memory bank** grows."
    )
    st.page_link("pages/2_Live_Demo.py", label="Open live demo →", icon="🔬")

st.divider()

st.caption(
    "LectureRAG-75 benchmark · 291 QA pairs · 5 domains "
    "(computer networks, database systems, generative AI, machine learning, operating systems) · "
    "retrieval-only (no generation API required)"
)
