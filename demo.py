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
    initial_sidebar_state="expanded",
)
inject_global_css()

st.title("Inference-Time Memory Adaptation for Cold-Start Educational RAG")

# ── What is ITMA? ───────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="background:linear-gradient(135deg,{ORANGE_DARK} 0%,{ORANGE} 100%);
                border-radius:12px;padding:1.4rem 1.8rem;margin:0.4rem 0 1.2rem 0;color:#fff">
        <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
                    text-transform:uppercase;opacity:0.92;margin-bottom:0.4rem">
            What is ITMA?
        </div>
        <div style="font-size:1.02rem;line-height:1.6">
            <b>ITMA (Inference-Time Memory Adaptation)</b> is a retrieval framework for
            educational lecture corpora that improves at deployment <b>without any retraining</b>.
            A lightweight scoring head is pretrained once on held-out domains and then frozen
            permanently. All per-deployment adaptation flows through an
            <b>online memory bank</b> that updates after each query via a
            counterfactual-reweighted feedback rule. No gradient computation, no LLM in the
            adaptation loop, no offline interaction log required at deployment time.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Professional summary ─────────────────────────────────────────────────────
st.markdown("### The problem ITMA solves")
st.markdown(
    """
    Existing adaptive RAG systems (CFRAG, R3, RankRAG, DynamicRAG) need one of: gradient-based
    fine-tuning on the target domain, an LLM judge in the adaptation loop, or a pre-populated
    interaction log. None of these are available the moment an instructor uploads a new
    lecture and students start asking questions. This is the **cold-start regime** — zero
    history, zero feedback, zero domain signal — and it is the normal operating condition
    for any freshly deployed educational RAG.

    ITMA is designed specifically for this regime. Cold-start safety is a *learned* property
    (the scoring head's gate is initialised near-closed, so retrieval at N=0 stays within
    range of standard dense baselines), and online adaptation is *gradient-free* (the
    memory bank is updated by a closed-form counterfactual reweighting rule that runs in
    milliseconds on CPU).
    """
)

# ── Simple example panel ────────────────────────────────────────────────────
st.markdown("### A simple way to think about it")
st.markdown(
    f"""
    <div style="background:{ORANGE_BG};border:1.5px solid {ORANGE_200};
                border-radius:10px;padding:1.2rem 1.5rem;margin:0.4rem 0 0.8rem 0;
                color:#1C1917">
        <div style="font-weight:700;color:{ORANGE_DARK};margin-bottom:0.5rem">
            🎓 Imagine a brand-new teaching assistant on their first day
        </div>
        <div style="font-size:0.96rem;line-height:1.65">
            The TA has read every lecture transcript but has never met a student. A student
            walks up and asks: <i>“When was IPv4 first specified?”</i>
            The TA pulls out a few candidate pages, reads them, and picks the most relevant
            one. The student says <b>“yes, that's the one”</b> and marks the chosen page helpful.
            <br><br>
            The next day a different student asks something similar — say,
            <i>“When did IPv4 get standardised?”</i>
            The TA remembers: <em>“Last time, a student asked something close to this and
            said the IPv4 history page was the right answer.”</em> They surface that same page
            faster and more confidently — even though no one has retrained them, no one
            updated their textbook, and they never opened a “study guide”.
            <br><br>
            <b>That’s ITMA.</b> The TA's “brain” (the pretrained scoring head) is frozen.
            The TA's “notes from yesterday's students” (the online memory bank) is what gets
            updated, query by query, in milliseconds, on CPU. After roughly fifty student
            interactions the TA's retrieval quality has measurably improved on every
            metric — and the improvement traces back to specific named (question, page,
            confidence) triples that a human reviewer can audit.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── How ITMA differs from other AI approaches ──────────────────────────────
st.markdown("### How ITMA differs from other AI tools")
st.markdown(
    """
    People often ask: *“can’t I just use ChatGPT, or a regular search tool over the
    lectures, or one of the newer adaptive systems?”* Each of those works for some things
    and breaks down for others. Here’s a plain comparison.
    """
)

st.markdown(
    f"""
    <div style="overflow-x:auto;margin:0.4rem 0 0.8rem 0">
    <table style="width:100%;border-collapse:collapse;font-size:0.88rem;
                  line-height:1.55;color:#1C1917;
                  border:1.5px solid {ORANGE_200};border-radius:10px;overflow:hidden">
      <thead>
        <tr style="background:{ORANGE_BG};color:#1C1917;text-align:left">
          <th style="padding:0.55rem 0.8rem;border-bottom:1.5px solid {ORANGE_200}">Question</th>
          <th style="padding:0.55rem 0.8rem;border-bottom:1.5px solid {ORANGE_200};
                     text-align:center;width:18%">
            ChatGPT / Gemini<br>
            <span style="font-weight:400;font-size:0.78rem;color:#78716C">(just an AI chatbot)</span>
          </th>
          <th style="padding:0.55rem 0.8rem;border-bottom:1.5px solid {ORANGE_200};
                     text-align:center;width:18%">
            Regular search over lectures<br>
            <span style="font-weight:400;font-size:0.78rem;color:#78716C">(plain RAG)</span>
          </th>
          <th style="padding:0.55rem 0.8rem;border-bottom:1.5px solid {ORANGE_200};
                     text-align:center;width:18%">
            Other learning systems<br>
            <span style="font-weight:400;font-size:0.78rem;color:#78716C">(CFRAG, R3, etc.)</span>
          </th>
          <th style="padding:0.55rem 0.8rem;border-bottom:1.5px solid {ORANGE_200};
                     text-align:center;width:18%;background:#FFEDD5">
            <b style="color:{ORANGE_DARK}">ITMA</b>
          </th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="padding:0.5rem 0.8rem;font-weight:600">Does it actually read <i>your</i> lecture?</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#B91C1C">No — guesses from general knowledge</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#15803D">Yes</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#15803D">Yes</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#15803D;background:#FFFBF5">Yes</td>
        </tr>
        <tr style="background:#FFFBF5">
          <td style="padding:0.5rem 0.8rem;font-weight:600">Does it get smarter when students say “this was helpful”?</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#B91C1C">No</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#B91C1C">No — same for everyone, forever</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#15803D">Yes</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#15803D;background:#FFEDD5">Yes</td>
        </tr>
        <tr>
          <td style="padding:0.5rem 0.8rem;font-weight:600">Does it work on day 1 (before any student has used it)?</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#78716C">N/A</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#15803D">Yes</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#B91C1C">No — needs old data to learn from first</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#15803D;background:#FFFBF5">Yes</td>
        </tr>
        <tr style="background:#FFFBF5">
          <td style="padding:0.5rem 0.8rem;font-weight:600">Does it need a re-training run to improve?</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#78716C">N/A</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#15803D">No (it doesn’t improve at all)</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#B91C1C">Yes — needs a GPU</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#15803D;background:#FFEDD5">No — updates instantly</td>
        </tr>
        <tr>
          <td style="padding:0.5rem 0.8rem;font-weight:600">Can you see <i>why</i> it picked an answer?</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#B91C1C">No — black box</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#78716C">N/A — no learning</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#B91C1C">No — hidden inside weights</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;color:#15803D;background:#FFFBF5">Yes — traces to a past student’s feedback</td>
        </tr>
        <tr style="background:#FFFBF5">
          <td style="padding:0.5rem 0.8rem;font-weight:600">What hardware does each query need?</td>
          <td style="padding:0.5rem 0.8rem;text-align:center">GPU or paid AI call</td>
          <td style="padding:0.5rem 0.8rem;text-align:center">Just CPU</td>
          <td style="padding:0.5rem 0.8rem;text-align:center">GPU for periodic re-training</td>
          <td style="padding:0.5rem 0.8rem;text-align:center;background:#FFEDD5"><b>Just CPU</b></td>
        </tr>
      </tbody>
    </table>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    **In plain English:**

    - **ChatGPT / Gemini** are AI chatbots. They have never read your specific lecture, so
      they often make things up that *sound* right but aren’t actually in your course. And
      a student saying “that was wrong” doesn’t change anything for the next student.
    - **Regular lecture search** (plain RAG) does read your lecture and gives the right
      page, but it never gets better. Whether the 1st or the 100th student uses it,
      retrieval quality is exactly the same.
    - **Other learning systems** like CFRAG do get better with feedback, but they need a
      pile of old feedback first, a separate GPU re-training step, or another AI in the
      loop. None of that is around the moment a new lecture goes live.
    - **ITMA** keeps the AI model fixed and instead just remembers which lecture pages
      have been marked helpful for which questions. That remembering is instant, runs on
      a normal laptop CPU, works on day 1, and you can always see *why* it ranked a page
      higher — because a real past student said it helped.
    """
)

# ── Can ITMA be used in other domains? ─────────────────────────────────────
st.markdown("### Can ITMA be used outside lectures — for healthcare, software, legal, etc.?")
st.markdown(
    """
    Short answer: **yes, in any domain whose content is video, audio, or text — as long
    as the corpus is reasonably stable and users can signal whether an answer was
    helpful**. The Upload page on this demo transcribes any video locally with
    faster-whisper before indexing, so video lectures, training recordings, podcasts and
    case-recording audio all plug in the same way as pre-existing text. Below is an
    honest domain-by-domain verdict.
    """
)


def _domain_card(icon: str, title: str, verdict: str, verdict_color: str,
                 reason: str, caveat: str = "") -> str:
    caveat_html = (
        f"<div style='margin-top:0.45rem;font-size:0.78rem;color:#78716C;"
        f"border-top:1px dashed #FED7AA;padding-top:0.4rem'>"
        f"<b>Caveat:</b> {caveat}</div>"
        if caveat else ""
    )
    return f"""
    <div style="border:1.5px solid #FED7AA;border-radius:10px;
                padding:0.95rem 1.05rem;background:#FFF7ED;color:#1C1917;height:100%;
                box-shadow:0 1px 4px rgba(249,115,22,0.07)">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:0.45rem">
        <div style="font-weight:700;font-size:0.98rem">{icon} {title}</div>
        <span style="background:{verdict_color};color:#fff;font-size:0.7rem;
                     padding:2px 8px;border-radius:10px;font-weight:700;
                     letter-spacing:0.04em">{verdict}</span>
      </div>
      <div style="font-size:0.86rem;line-height:1.6;color:#292524">{reason}</div>
      {caveat_html}
    </div>
    """


col_a, col_b, col_c = st.columns(3, gap="small")
with col_a:
    st.markdown(_domain_card(
        "🩺", "Healthcare / Medical", "FITS WITH CARE", "#D97706",
        "<b>Inputs:</b> surgical training videos, ward-rounds recordings, patient-education "
        "videos, clinical-guideline PDFs. The Upload page already transcribes any video "
        "with faster-whisper locally, so a hospital can drop in a CME recording and start "
        "asking questions the same day. Nurses/doctors can say “this protocol applied” — "
        "exactly the feedback ITMA expects. Full audit trail is a real strength for "
        "HIPAA-style review.",
        "Swap MiniLM for a clinical encoder (BioBERT, ClinicalBERT, or PubMedBERT) so "
        "medical vocabulary doesn’t get embedded as gibberish. Whisper’s medical-term "
        "accuracy is good but not perfect — proofread transcripts before deploying in "
        "diagnostic-critical contexts.",
    ), unsafe_allow_html=True)

with col_b:
    st.markdown(_domain_card(
        "💻", "Software docs & customer support", "STRONG FIT", "#15803D",
        "<b>Inputs:</b> conference-talk videos (e.g. recorded engineering all-hands), "
        "screencast tutorials, support-call recordings, plus written API docs and "
        "ticket archives. The “was this article helpful?” thumbs already exists in "
        "most support tools — ITMA turns those clicks into better future retrieval, "
        "on day 1, with no GPU.",
        "Searching <i>raw source code</i> (not screencasts <i>about</i> code) needs a "
        "code-trained encoder like CodeBERT or StarCoder embeddings instead of MiniLM.",
    ), unsafe_allow_html=True)

with col_c:
    st.markdown(_domain_card(
        "⚖️", "Legal & regulatory", "STRONG FIT", "#15803D",
        "<b>Inputs:</b> deposition videos, court-hearing recordings, regulator briefings, "
        "plus written case law and contracts. Lawyers and paralegals iteratively refine "
        "queries and mark cases relevant — same shape as student feedback on a lecture "
        "page. The audit trail is a hard requirement in regulated environments, and "
        "ITMA provides it natively.",
    ), unsafe_allow_html=True)

st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

col_d, col_e, col_f = st.columns(3, gap="small")
with col_d:
    st.markdown(_domain_card(
        "🏢", "Corporate training & HR", "DIRECT TRANSFER", "#15803D",
        "<b>Inputs:</b> onboarding videos, leadership-training recordings, recorded "
        "town halls, written policy docs. The deployment shape is identical to the "
        "Upload page on this demo — drop a video in, transcript and index build "
        "locally, employees ask questions on day 1, the system gets sharper across the "
        "week as people mark answers helpful. No retraining cycle needed.",
    ), unsafe_allow_html=True)

with col_e:
    st.markdown(_domain_card(
        "🔬", "Research literature search", "STRONG FIT", "#15803D",
        "<b>Inputs:</b> conference-talk recordings, lab seminars, plus written papers "
        "and pre-prints. A researcher refines a query several times and marks "
        "papers/talks useful. ITMA’s cross-query memory means a paper marked helpful "
        "for query A will also surface for a semantically similar query B later — "
        "the natural literature-review workflow.",
    ), unsafe_allow_html=True)

with col_f:
    st.markdown(_domain_card(
        "❌", "Code search · silent images · live feeds", "POOR FIT", "#B91C1C",
        "<b>Raw code search</b> needs code-trained embeddings (MiniLM treats code "
        "as malformed English). <b>Pure image retrieval with no spoken audio</b> "
        "(diagrams, photographs, scans) needs multimodal encoders ITMA doesn’t use — "
        "but a video <i>with</i> narration over those images works fine, because "
        "Whisper transcribes the narration. <b>Live news feeds and social streams</b> "
        "break the ID-boost assumption that chunk IDs are stable. These need "
        "different architectures.",
    ), unsafe_allow_html=True)

st.markdown(
    """
    **What a new domain needs for ITMA to plug in cleanly:**

    1. A **corpus that can be reduced to text + stable chunk IDs.** Video and audio are
       fine — the Upload page in this demo already handles that path
       (`ffmpeg` → `faster-whisper` → chunked timestamped transcript → FAISS). Pure
       images, raw binaries, and live streams don’t fit because there is nothing to
       transcribe.
    2. **An embedding model that understands the vocabulary.** MiniLM is fine for general
       English; medical / legal / multilingual domains should swap it for a domain encoder.
       The rest of ITMA (memory bank, scoring head, boost) is encoder-agnostic.
    3. **A user feedback signal** — even a single thumbs-up checkbox is enough. ITMA does
       not need explicit ratings or labelled data.
    4. **A relatively stable corpus** — if chunks are added or rewritten constantly, the
       memory bank’s chunk-ID-based boost loses its grip. A weekly or monthly content
       refresh is fine; a live news stream is not.

    If all four hold, the same code in this repo runs on the new domain with only an
    encoder swap. If only one or two hold, expect ~a day of adapter work. If multiple
    fail (raw code, silent images, real-time data), ITMA is the wrong tool — a different
    architecture is needed.
    """
)

# ── Headline numbers ────────────────────────────────────────────────────────
st.markdown("### What the paper shows")
mc1, mc2, mc3 = st.columns(3)
with mc1:
    st.metric(
        "Hit@5 growth (N=0 → N=50)",
        "+2.25 pp",
        help="ITMA is the only retriever in the benchmark with positive growth on every metric",
    )
with mc2:
    st.metric(
        "Recall@10 at N=50",
        "0.923",
        delta="surpasses CFRAG-lite (0.888)",
        help="ITMA at N=50 is the strongest system on Recall@10 — the metric most relevant to the LLM context window",
    )
with mc3:
    st.metric(
        "Retraining cost at deployment",
        "0 gradient updates",
        help="No fine-tuning, no LLM judge, no replay buffer",
    )

st.divider()

# ── Three demo pages ────────────────────────────────────────────────────────
st.markdown("### Try it three ways")

col1, col2, col3 = st.columns(3, gap="large")

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
        "Interactive retriever on the pretrained lecture corpus. Ask any question, mark "
        "helpful chunks, and watch ITMA re-rank in real time as its **memory bank** grows."
    )
    st.page_link("pages/2_Live_Demo.py", label="Open live demo →", icon="🔬")

with col3:
    st.subheader("🎬 Upload your own video")
    st.markdown(
        "Upload **any lecture video** — the system transcribes, chunks, embeds, and "
        "indexes it locally, then lets you query and give feedback in real time. "
        "Verifies that ITMA generalises to unseen content."
    )
    st.page_link("pages/3_Upload_Video.py", label="Open upload page →", icon="🎬")

st.divider()

st.caption(
    "LectureRAG-75 benchmark · 442 QA pairs · 5 domains "
    "(computer networks, database systems, generative AI, machine learning, operating systems) · "
    "265/88/89 train/dev/test · retrieval-only (no generation API required)"
)
