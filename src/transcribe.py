"""ASR transcription — faster-whisper backend with INT8 quantization and VAD.

Step 2 (pipeline-flow ordering): Optimize ASR for Edge Computing
  - INT8 quantization (compute_type="int8") reduces model memory ~4× vs FP32
  - Voice Activity Detection (vad_filter=True) skips silent segments,
    reducing hallucinations and speeding up transcription on CPU
  - Falls back to openai-whisper if faster-whisper is unavailable
"""

import os

DEFAULT_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
DEFAULT_DEVICE     = os.getenv("WHISPER_DEVICE", "cpu")
# INT8 on CPU; use "float16" if running on CUDA
DEFAULT_COMPUTE    = os.getenv("WHISPER_COMPUTE_TYPE", "int8")


def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:06.3f}"


def transcribe_audio(audio_path: str, transcript_path: str) -> None:
    """Transcribe audio to a timestamped transcript file.

    Uses faster-whisper (INT8, VAD) for edge-optimised inference.
    Falls back to openai-whisper if faster-whisper is not importable.
    """
    try:
        _transcribe_faster_whisper(audio_path, transcript_path)
    except ImportError:
        _transcribe_openai_whisper(audio_path, transcript_path)


def _transcribe_faster_whisper(audio_path: str, transcript_path: str) -> None:
    """INT8-quantized transcription with Voice Activity Detection."""
    from faster_whisper import WhisperModel

    model = WhisperModel(
        DEFAULT_MODEL_NAME,
        device=DEFAULT_DEVICE,
        compute_type=DEFAULT_COMPUTE,   # INT8 on CPU — ~4× less memory than FP32
    )

    segments, info = model.transcribe(
        audio_path,
        language="en",
        vad_filter=True,                # skip silent segments (VAD)
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        },
        temperature=0,
        condition_on_previous_text=False,
        beam_size=5,
    )

    os.makedirs(os.path.dirname(transcript_path) if os.path.dirname(transcript_path) else ".", exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as f:
        for seg in segments:
            start = _fmt_ts(seg.start)
            end   = _fmt_ts(seg.end)
            f.write(f"[{start} --> {end}] {seg.text.strip()}\n")

    print(
        f"Transcription saved to {transcript_path} "
        f"(model={DEFAULT_MODEL_NAME}, device={DEFAULT_DEVICE}, "
        f"compute_type={DEFAULT_COMPUTE}, vad=True, "
        f"detected_language={info.language}, duration={info.duration:.1f}s)"
    )


def _transcribe_openai_whisper(audio_path: str, transcript_path: str) -> None:
    """Fallback: openai-whisper (FP32, no VAD)."""
    import whisper
    model = whisper.load_model(DEFAULT_MODEL_NAME, device=DEFAULT_DEVICE)
    result = model.transcribe(
        audio_path,
        fp16=False,
        language="en",
        verbose=False,
        condition_on_previous_text=False,
        temperature=0,
    )
    os.makedirs(os.path.dirname(transcript_path) if os.path.dirname(transcript_path) else ".", exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            start = _fmt_ts(seg["start"])
            end   = _fmt_ts(seg["end"])
            f.write(f"[{start} --> {end}] {seg['text'].strip()}\n")
    print(f"Transcription saved (fallback openai-whisper, model={DEFAULT_MODEL_NAME})")
