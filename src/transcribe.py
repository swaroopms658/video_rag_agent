import whisper
import os


DEFAULT_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
DEFAULT_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")


def _fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:06.3f}"


def transcribe_audio(audio_path, transcript_path):
    model = whisper.load_model(DEFAULT_MODEL_NAME, device=DEFAULT_DEVICE)
    result = model.transcribe(
        audio_path,
        fp16=False,
        language="en",
        verbose=False,
        condition_on_previous_text=False,
        temperature=0,
    )
    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            start = _fmt_ts(seg["start"])
            end = _fmt_ts(seg["end"])
            f.write(f"[{start} --> {end}] {seg['text'].strip()}\n")
    print(
        f"Transcription saved to {transcript_path} "
        f"(model={DEFAULT_MODEL_NAME}, device={DEFAULT_DEVICE})"
    )
