"""Orchestration glue for the upload-and-query demo page.

Wires together video_utils.extract_audio, transcribe.transcribe_audio, and
build_vectorstore.build_faiss_store, calling a progress callback between
stages so the Streamlit page can render a live status block.

No Streamlit imports — callable from the CLI for smoke testing.
"""

from __future__ import annotations

import os
from typing import Callable

ProgressCallback = Callable[[str, float], None]


def _noop(stage: str, frac: float) -> None:
    pass


def process_video(
    video_path: str,
    session_dir: str,
    progress_cb: ProgressCallback = _noop,
    whisper_model: str = "small",
    chunk_size: int = 200,
    overlap: int = 40,
) -> str:
    """Run the full ingest pipeline for an uploaded video.

    Stages: extract_audio → transcribe → build_faiss_store.

    Writes the following files into session_dir (caller is responsible for
    placing video.<ext> there beforehand):
        audio.wav
        transcript.txt
        index.faiss
        meta.pkl

    Returns session_dir on success. Re-raises any exception from the
    underlying stages so the caller can map them to UI messages.
    """
    # Make sure transcribe.py picks up the right Whisper size.  Must be set
    # before src.transcribe is imported (which reads the env var at module
    # load time).
    os.environ["WHISPER_MODEL"] = whisper_model

    audio_path = os.path.join(session_dir, "audio.wav")
    transcript_path = os.path.join(session_dir, "transcript.txt")

    progress_cb("Extracting audio (ffmpeg)", 0.0)
    from src.video_utils import extract_audio
    extract_audio(video_path, audio_path)
    progress_cb("Extracting audio (ffmpeg)", 1.0)

    progress_cb(f"Transcribing (faster-whisper {whisper_model}, INT8+VAD)", 0.0)
    from src.transcribe import transcribe_audio
    transcribe_audio(audio_path, transcript_path)
    if os.path.getsize(transcript_path) == 0:
        raise RuntimeError(
            "Transcription produced an empty transcript. The video likely has "
            "no audible speech."
        )
    progress_cb(f"Transcribing (faster-whisper {whisper_model}, INT8+VAD)", 1.0)

    progress_cb("Chunking, embedding, FAISS index", 0.0)
    from src.build_vectorstore import build_faiss_store
    build_faiss_store(
        transcript_path=transcript_path,
        output_dir=session_dir,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    progress_cb("Chunking, embedding, FAISS index", 1.0)

    return session_dir
