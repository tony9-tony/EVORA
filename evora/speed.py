"""Speed helpers for small local Ollama models on modest hardware."""

from __future__ import annotations

import re
from typing import Optional

import httpx

CODE_HINTS = re.compile(
    r"\b(code|bug|fix|error|function|class|file|folder|test|run|create|write|edit|refactor|git|commit|install|"
    r"script|api|server|html|css|python|javascript|sql|tengeneza|andika|rekebisha|faili|msimbo|kosa|jaribu|endesha)\b",
    re.IGNORECASE,
)


def native_base(openai_base_url: str) -> str:
    """http://127.0.0.1:11434/v1 -> http://127.0.0.1:11434"""
    return openai_base_url.rstrip("/").removesuffix("/v1")


def list_models(openai_base_url: str, timeout: float = 5.0) -> list[str]:
    try:
        r = httpx.get(native_base(openai_base_url) + "/api/tags", timeout=timeout)
        if r.status_code == 200:
            return [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
    except Exception:
        pass
    return []


def pick_models(available: list[str]) -> dict[str, Optional[str]]:
    """Pick a coding model (qwen/coder) and a fast chat model (gemma / smallest) from what is installed."""
    def first(*needles: str) -> Optional[str]:
        for needle in needles:
            for name in available:
                if needle in name.lower():
                    return name
        return None

    code = first("qwen2.5-coder", "coder", "qwen")
    chat = first("gemma", "llama3.2", "phi") or code or (available[0] if available else None)
    return {"code": code or chat, "chat": chat}


def route(message: str, models: dict[str, Optional[str]]) -> Optional[str]:
    """Code/action-like requests go to the coder model; plain talk goes to the fast model."""
    if CODE_HINTS.search(message or ""):
        return models.get("code")
    return models.get("chat")


def warm_up(openai_base_url: str, model: str, keep_alive: str = "30m", timeout: float = 120.0) -> bool:
    """Load the model into memory and keep it there so the first answer is not slow."""
    try:
        r = httpx.post(native_base(openai_base_url) + "/api/generate",
                       json={"model": model, "prompt": "", "keep_alive": keep_alive, "stream": False},
                       timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False
