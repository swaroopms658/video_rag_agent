"""ffmpeg subprocess wrappers for the upload-and-query demo page.

No Streamlit imports — keeps this module testable from the CLI.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Optional


def ffmpeg_available() -> bool:
    """Return True if ffmpeg is on PATH."""
    return shutil.which("ffmpeg") is not None


def extract_audio(video_path: str, audio_path: str, sample_rate: int = 16000) -> None:
    """Extract mono PCM-WAV audio from a video file.

    Raises:
        FileNotFoundError: ffmpeg is not on PATH.
        subprocess.CalledProcessError: ffmpeg failed (unsupported codec, corrupt file, etc.).
            The stderr is captured on the exception and can be shown to the user.
    """
    if not ffmpeg_available():
        raise FileNotFoundError("ffmpeg not found on PATH")

    cmd = [
        "ffmpeg",
        "-y",                # overwrite output without prompting
        "-i", video_path,
        "-vn",               # drop video stream
        "-ac", "1",          # mono
        "-ar", str(sample_rate),
        "-f", "wav",
        audio_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def probe_duration(video_path: str) -> Optional[float]:
    """Return duration in seconds via ffprobe, or None if probing fails."""
    if not shutil.which("ffprobe"):
        return None
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
        payload = json.loads(out.stdout)
        return float(payload["format"]["duration"])
    except (subprocess.CalledProcessError, KeyError, ValueError):
        return None
