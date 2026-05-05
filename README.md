# Agentic Lecture RAG

Text-only retrieval-augmented generation system for lecture understanding. The project is aligned to these six objectives:

1. To develop an agentic reasoning framework for text-based lecture understanding.
2. To optimize automatic speech recognition for edge computing environments.
3. To convert lecture audio into structured and searchable textual knowledge.
4. To implement a cost-aware caching mechanism for efficient inference.
5. To incorporate feedback-driven memory for improved text retrieval and response quality.
6. To evaluate the framework in terms of accuracy, latency, and cost efficiency.

## Overview

The repository implements a transcript-centric pipeline:

1. `src/transcribe.py`
   Converts lecture audio or video into text using Whisper with edge-oriented defaults.
2. `src/build_vectorstore.py`
   Chunks the transcript and creates sentence embeddings for retrieval.
3. `src/rag_chain.py`
   Retrieves top transcript chunks, applies memory-based boosting, and generates grounded answers.
4. `src/answer_cache.py`
   Stores previous query-context-answer results to reduce repeated LLM calls.
5. `src/retrieval_memory.py`
   Reuses successful historical context IDs for similar future queries.
6. `src/evaluate.py`
   Measures generated-answer similarity against a reference QA set.

## Project Structure

```text
lecture_rag_agent/
├── data/
│   ├── lecture_transcript.txt
│   ├── vector_store.pkl
│   ├── eval_set.json
│   ├── rl_feedback.json
│   └── answer_cache.json
├── src/
│   ├── agent.py
│   ├── answer_cache.py
│   ├── build_vectorstore.py
│   ├── evaluate.py
│   ├── generate_eval_set.py
│   ├── interactive_demo.py
│   ├── key_manager.py
│   ├── rag_chain.py
│   ├── retrieval_memory.py
│   └── transcribe.py
└── requirements.txt
```

## Setup

### 1. Requirements

- Python 3.10+
- `ffmpeg` on `PATH`
- Groq API key in `.env`

Example `.env`:

```env
GROQ_API_KEY=your_key_here
```

Optional ASR tuning for edge deployment:

```env
WHISPER_MODEL=base
WHISPER_DEVICE=cpu
```

### 2. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline

### Step 1. Transcribe lecture audio

```bash
python -c "from src.transcribe import transcribe_audio; transcribe_audio('data/lecture.mp4', 'data/lecture_transcript.txt')"
```

### Step 2. Build the text vector store

```bash
python -c "from src.build_vectorstore import create_embeddings; create_embeddings('data/lecture_transcript.txt', 'data/vector_store.pkl')"
```

### Step 3. Run the interactive CLI

After installation, you can launch the terminal assistant directly:

```bash
agentic-video-rag
```

Explicit chat mode:

```bash
agentic-video-rag chat
```

Single-question mode:

```bash
agentic-video-rag ask "What is retrieval augmented generation?"
```

Legacy module entrypoint still works:

```bash
python -m src.interactive_demo
```

### Step 4. Generate an evaluation set

```bash
python -m src.generate_eval_set
```

### Step 5. Evaluate the system

```bash
python -m src.evaluate
```

## Implementation Notes

- Agentic reasoning:
  `src/rag_chain.py` grounds answers strictly in retrieved transcript chunks and instructs the model to avoid unsupported guesses.
- Edge-oriented ASR:
  `src/transcribe.py` uses configurable Whisper settings with CPU-safe defaults.
- Searchable text knowledge:
  `src/build_vectorstore.py` creates chunked sentence embeddings for transcript retrieval.
- Cost-aware inference:
  `src/answer_cache.py` caches answers by normalized query and retrieved context signature.
- Feedback-driven memory:
  `src/retrieval_memory.py` boosts transcript chunks that were previously validated for similar queries.
- Evaluation:
  `src/evaluate.py` reports cosine similarity and exact match over the QA set.

## Notes

- The repository supports transcript-based lecture retrieval and answering.
- Cache and feedback files are generated locally under `data/`.
