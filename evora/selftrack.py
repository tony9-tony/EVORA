"""Self-tracking: EVORA keeps a ledger of where it fails so the owner can decide what to fix."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path


class WeaknessTracker:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    @staticmethod
    def signature(kind: str, tool: str, detail: str) -> str:
        norm = re.sub(r"[0-9a-f]{8,}|\d+", "#", (detail or "").lower())
        norm = re.sub(r"[/\\][^\s:]+", "<path>", norm)
        return f"{kind}|{tool}|{norm[:90]}"

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def record(self, kind: str, tool: str, detail: str) -> str:
        sig = self.signature(kind, tool, detail)
        with self._lock:
            data = self._load()
            now = time.time()
            item = data.get(sig) or {"kind": kind, "tool": tool, "count": 0, "first": now,
                                     "status": "open", "example": (detail or "")[:300]}
            item["count"] += 1
            item["last"] = now
            if item["status"] == "fixed":
                item["status"] = "open"  # it came back
            data[sig] = item
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data), encoding="utf-8")
        return sig

    def top(self, limit: int = 20, include_closed: bool = False) -> list[dict]:
        data = self._load()
        items = [{"id": sig, **v} for sig, v in data.items() if include_closed or v.get("status") == "open"]
        items.sort(key=lambda i: (-i["count"], -i.get("last", 0)))
        return items[:limit]

    def mark(self, sig: str, status: str) -> bool:
        if status not in ("open", "fixed", "ignored"):
            return False
        with self._lock:
            data = self._load()
            if sig not in data:
                return False
            data[sig]["status"] = status
            self.path.write_text(json.dumps(data), encoding="utf-8")
        return True
