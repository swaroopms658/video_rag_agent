import whisper
import os

def transcribe_audio(audio_path, transcript_path):
    model = whisper.load_model("small")
    result = model.transcribe(audio_path)
    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"Transcription saved to {transcript_path}")
