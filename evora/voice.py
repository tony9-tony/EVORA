"""Local speech-to-text (English + Swahili) using faster-whisper when installed."""

from __future__ import annotations

import os
import tempfile
import threading
from typing import Optional


class VoiceUnavailable(RuntimeError):
    pass


_model = None
_lock = threading.Lock()


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def _load():
    global _model
    with _lock:
        if _model is None:
            try:
                from faster_whisper import WhisperModel
            except Exception as e:
                raise VoiceUnavailable(
                    "faster-whisper is not installed. Run: pip install faster-whisper") from e
            size = os.environ.get("EVORA_WHISPER_MODEL", "base")
            _model = WhisperModel(size, device="cpu", compute_type="int8")
        return _model


def transcribe(audio: bytes, language: Optional[str] = None, suffix: str = ".webm") -> dict:
    """language: 'en', 'sw' or None for auto-detect."""
    model = _load()
    lang = language if language in ("en", "sw") else None
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        fh.write(audio)
        path = fh.name
    try:
        segments, info = model.transcribe(path, language=lang, beam_size=1, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        return {"text": text, "language": info.language}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
