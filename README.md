# Agentic-VideoRAG: An Enhanced and Cost-Aware Framework for Scalable Video Retrieval

[![Journal](https://img.shields.io/badge/Journal-EAAI--2026-blue)](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Agentic-VideoRAG** is a high-performance, cost-aware multimodal Retrieval-Augmented Generation (RAG) system designed for edge-deployed engineering environments. It intelligently orchestrates audio transcripts and visual frames using an agentic decision-making layer and a **Reinforcement Learning from Human Feedback (RLHF)** active learning memory loop.

---

## 📄 Note on Journal Submission
This repository contains the official implementation for the paper:  
> **"Agentic-VideoRAG: A Cost-Aware and Explainable Framework for Video Retrieval in Edge Engineering Environments"**  
> *Submitted to Elsevier: Engineering Applications of Artificial Intelligence (EAAI), 2026.*

---

## Architecture

### Indexing Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                      INPUT: data/lecture.mp4                     │
└──────────────────────────┬───────────────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
     [ AUDIO STREAM ]           [ VIDEO FRAMES ]
      transcribe.py              extract_frames.py
      Whisper (small)            OpenCV @ 1 frame / 5s
             │                           │
             ▼                           ▼
   data/lecture_transcript.txt    data/frames/*.jpg
             │                  + data/frames_metadata.json
             │                           │
             ▼                           ▼
    build_vectorstore.py    generate_visual_embeddings.py
    chunk_text()                  CLIP (clip-ViT-B-32)
    size=1000, overlap=100        model.encode(images)
    SentenceTransformer                   │
    (all-MiniLM-L6-v2)                    │
             │                            │
             ▼                            ▼
    data/vector_store.pkl    data/visual_vector_store.pkl
    { chunks[], embeddings[] } { metadata[], embeddings[] }
```

### Query & Answer Flow

```
                         USER QUERY
                              │
               ┌──────────────┴───────────────┐
               │                              │
               ▼                              ▼
      [ TEXT-ONLY RAG ]           [ MULTIMODAL RAG ]
        rag_chain.py               multimodal_rag.py
               │                              │
               │                    ┌─────────┴─────────┐
               │                    │                   │
               ▼                    ▼                   ▼
         embed (MiniLM)       embed (MiniLM)      embed (CLIP)
         cosine_sim()         cosine_sim()        cosine_sim()
         top-3 chunks         top-2 chunks        top-1 frame
               │                    │            + memory boost
               │                    └─────────┬─────────┘
               │                              │
               ▼                              ▼
   ┌───────────────────┐         ┌────────────────────────┐
   │  Groq LLM         │         │  Groq Vision LLM        │
   │  llama-3.1-8b     │         │  llama-4-scout-17b      │
   │  (text prompt)    │         │  (text + base64 images) │
   └─────────┬─────────┘         └───────────┬────────────┘
             │                               │
             └───────────────┬───────────────┘
                             ▼
                          ANSWER
                       + source citations
                       + confidence score
                             │
                    ┌────────┴─────────┐
                    │  USER FEEDBACK   │
                    │    (y / n)       │
                    └────────┬─────────┘
                             │
                             ▼
                   data/rl_feedback.json
                             │
                             ▼
                   RetrievalMemory
                   batch-encodes past queries
                   boosts matching sources (+0.2)
                   on the next similar query
```

### Active Learning / Memory Loop

```
  New query arrives
        │
        ▼
  RetrievalMemory.get_verified_contexts(query, threshold=0.85)
        │
        ├─ encode current query (MiniLM)
        ├─ cosine_sim vs all pre-encoded past queries
        │
        ├─ sim > 0.85 ──YES──► return context_ids from that entry
        │                               │
        │                               ▼
        │                   VisualRetriever: sims[i] += 0.2
        │                   for each matching frame path
        │
        └─ sim ≤ 0.85 ──────────► normal retrieval, no boost
```

---

## Error Handling

### API Rate Limits (429) — both `rag_chain.py` and `multimodal_rag.py`

```
POST /chat/completions
        │
        ├─ 200 OK ──────────────────► return answer
        │
        ├─ 429 Rate Limited
        │       │
        │       ├─ print warning with last 4 chars of key
        │       ├─ wait: 20s × attempt  (text)
        │       │         30s × attempt  (multimodal)
        │       ├─ KeyManager.rotate_key() → next key in pool
        │       └─ retry (max 3 attempts total)
        │               │
        │               └─ exhausted → raise RuntimeError
        │
        ├─ other HTTP error ─────────► raise RuntimeError (text)
        │                              return error string (multimodal)
        │
        └─ network exception ────────► print, decrement retries
```

### Key Manager

```
.env loaded → GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY
        │
        ├─ all empty? → "Warning: No API keys found"
        │               get_current_key() returns None
        │               groq_generate() raises RuntimeError
        │
        └─ only 1 key? → rotate_key() warns, stays on same key
```

### Other Failure Points

| Location | Failure | Behaviour |
|----------|---------|-----------|
| `extract_frames.py` | `data/lecture.mp4` missing | Prints error, returns |
| `generate_visual_embeddings.py` | `frames_metadata.json` missing | Prints error, returns |
| `evaluate.py` / `evaluate_multimodal.py` | `eval_set.json` missing | Prints error, returns |
| `interactive_demo.py` | `.pkl` store missing | Caught by `except`, prints `Initialization Failed`, exits |
| `retrieval_memory.py` | Corrupt line in `rl_feedback.json` | `except: continue` — line skipped silently |
| `generate_eval_set.py` | LLM returns non-JSON | `except` catches `json.loads` error, skips that chunk |

---

## Project Structure

```
lecture_rag_agent/
├── .env                              # API keys — never commit this
├── .env.example                      # Template
├── requirements.txt
├── README.md
├── data/
│   ├── lecture.mp4                   # Your input video (not tracked)
│   ├── lecture_transcript.txt        # Output of transcribe.py
│   ├── frames/                       # Output of extract_frames.py
│   ├── frames_metadata.json          # Frame index + timestamps
│   ├── vector_store.pkl              # Text embeddings
│   ├── visual_vector_store.pkl       # Image embeddings
│   ├── eval_set.json                 # QA evaluation pairs
│   └── rl_feedback.json              # RLHF feedback log
└── src/
    ├── transcribe.py                 # Whisper: audio → transcript
    ├── build_vectorstore.py          # Transcript → text embeddings
    ├── extract_frames.py             # Video → frame images
    ├── generate_visual_embeddings.py # Frames → CLIP embeddings
    ├── agent.py                      # SimpleRetriever (cosine sim)
    ├── key_manager.py                # Groq API key pool + rotation
    ├── rag_chain.py                  # AgenticRAG (text-only)
    ├── multimodal_rag.py             # MultimodalRAG (text + vision)
    ├── retrieval_memory.py           # RLHF memory + source boosting
    ├── interactive_demo.py           # CLI entry point
    ├── generate_eval_set.py          # Auto-generate text QA pairs
    ├── generate_visual_eval_set.py   # Auto-generate visual QA pairs
    ├── evaluate.py                   # Evaluate text-only RAG
    ├── evaluate_multimodal.py        # Evaluate multimodal RAG
    └── eval_utils.py                 # Shared: load_eval_set, calculate_similarity
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.10+
- `ffmpeg` on PATH — required by Whisper ([download](https://ffmpeg.org/download.html))
- A Groq API key — free at [console.groq.com](https://console.groq.com)

### 2. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd lecture_rag_agent

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install PyTorch first

Visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and pick the right command for your OS/GPU. Example for CPU-only:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY_1=gsk_your_primary_key_here
GROQ_API_KEY_2=gsk_your_backup_key_here    # optional — used on rate limit
```

### 6. Add your lecture video

```bash
mkdir data
# copy your video to:
# data/lecture.mp4
```

---

## Running the Pipeline

Run these steps in order the first time. Once `.pkl` stores exist, skip straight to Step 5.

### Step 1 — Transcribe audio

```bash
python -m src.transcribe
```

Or from a script:

```python
from src.transcribe import transcribe_audio
transcribe_audio('data/lecture.mp4', 'data/lecture_transcript.txt')
```

Output: `data/lecture_transcript.txt`

### Step 2 — Build text vector store

```bash
python -c "from src.build_vectorstore import create_embeddings; create_embeddings('data/lecture_transcript.txt', 'data/vector_store.pkl')"
```

Output: `data/vector_store.pkl`

### Step 3 — Extract video frames

```bash
python -m src.extract_frames
```

Output: `data/frames/*.jpg`, `data/frames_metadata.json`

### Step 4 — Generate visual embeddings

```bash
python -m src.generate_visual_embeddings
```

Output: `data/visual_vector_store.pkl`

### Step 5 — Launch interactive demo

```bash
python -m src.interactive_demo
```

You will be prompted to:
1. Select dataset (original or second video if indexed)
2. Choose mode: **Text-Only RAG** or **Multimodal RAG**
3. Ask a question about the lecture
4. Rate the answer (`y` / `n`) — this trains the memory

---

## Evaluation

### Generate QA pairs

```bash
# From audio transcript (10 pairs)
python -m src.generate_eval_set

# From visual frames (5 pairs, appended to same file)
python -m src.generate_visual_eval_set
```

### Run benchmarks

```bash
# Text-only RAG
python -m src.evaluate

# Multimodal RAG
python -m src.evaluate_multimodal
```

Metrics:

| Metric | Description |
|--------|-------------|
| Average Cosine Similarity | Semantic closeness of generated vs ground truth answer |
| Exact Match Score | Strict lowercase string match rate |

---

## Models Used

| Component | Model | Notes |
|-----------|-------|-------|
| Transcription | `whisper small` | Upgrade to `medium` for better accuracy on technical content |
| Text embedding | `all-MiniLM-L6-v2` | Fast, 384-dim, good for sentence-level retrieval |
| Image embedding | `clip-ViT-B-32` | Cross-modal: text queries can retrieve images |
| Text LLM | `llama-3.1-8b-instant` via Groq | Fast, low quota usage |
| Vision LLM | `meta-llama/llama-4-scout-17b-16e-instruct` via Groq | Accepts base64 images inline |

---

## Adding a Second Video

```bash
# 1. Transcribe
python -c "from src.transcribe import transcribe_audio; transcribe_audio('data/7.mp4', 'data/transcript_7.txt')"

# 2. Build text store
python -c "from src.build_vectorstore import create_embeddings; create_embeddings('data/transcript_7.txt', 'data/vector_store_7.pkl')"

# 3. Edit VIDEO_PATH in src/extract_frames.py to 'data/7.mp4'
#    Edit OUTPUT_PATH in src/generate_visual_embeddings.py to 'data/visual_vector_store_7.pkl'

# 4. Extract frames + visual embeddings
python -m src.extract_frames
python -m src.generate_visual_embeddings
```

The interactive demo auto-detects `data/vector_store_7.pkl` and offers it as option 2.
