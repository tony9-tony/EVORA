"""
Owner governance for EVORA.

The owner (CEO) is the final authority. EVORA is never *refused* anything the
owner wants; instead every sensitive step is put in front of the owner as an
Accept / Deny card, and the owner chooses how much freedom EVORA gets through
an autonomy level that only the owner can change.

Pieces:
    AutonomyLevel     0 observer .. 3 trusted (owner-set, persisted)
    classify()        turns a tool call into an Action (category + risk)
    decide()          auto-run or ask the owner, for a level + action
    AuditLog          append-only, hash-chained JSONL record of everything
    Trash             "delete" moves to a recoverable trash folder
    ApprovalBroker    pending approvals shared between the agent loop and the web UI
    Governance        facade used by the agent loop and the web server
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional

from evora.security import PermissionLevel, PermissionManager


class AutonomyLevel(IntEnum):
    OBSERVER = 0      # reads freely; everything else asks
    ASSISTANT = 1     # + edits files inside the workspace
    COLLABORATOR = 2  # + runs safe commands
    TRUSTED = 3       # + installs, commits; only irreversible/outgoing things still ask


LEVEL_INFO = {
    AutonomyLevel.OBSERVER: ("Observer", "Reads and researches. Asks before anything else."),
    AutonomyLevel.ASSISTANT: ("Assistant", "Edits files in the workspace. Asks before running commands."),
    AutonomyLevel.COLLABORATOR: ("Collaborator", "Also runs safe commands. Asks for installs and commits."),
    AutonomyLevel.TRUSTED: ("Trusted", "Also installs and commits. Always asks for deletes, sends and self-changes."),
}

# ---------------------------------------------------------------- classification

READ_TOOLS = {
    "read_file", "list_dir", "search_files", "search_content", "git_status", "git_diff",
    "git_log", "analyze_project", "analyze_code", "self_analyze",
}
NETWORK_READ_TOOLS = {"web_search", "web_fetch", "browser_search", "browser_read"}
WRITE_TOOLS = {"write_file", "edit_file", "create_dir"}
EXEC_TOOLS = {"execute_command", "run_tests"}
GIT_WRITE_TOOLS = {"git_commit", "git_branch"}
SEND_TOOLS = {"email_send", "email_reply", "github_write", "github_push", "browser_act", "login"}
DELETE_TOOLS = {"delete_path"}
SELF_TOOLS = {"self_improve"}

# Commands that touch the owner's controls or send things out: always shown to the owner.
SENSITIVE_COMMAND = re.compile(
    r"(web_token|audit\.jsonl|governance\.json|vault|/api/approve|/api/autonomy|"
    r"\bgit\s+push\b|\bsendmail\b|\bsmtp|\bcurl\b.*\s-X\s*(POST|PUT|DELETE)|\bscp\b|\bssh\b)",
    re.IGNORECASE,
)

SECRET_KEY = re.compile(r"(pass(word)?|token|secret|api[_-]?key|authorization|cookie)", re.IGNORECASE)


@dataclass
class Action:
    tool: str
    args: dict
    category: str          # read | network_read | write | exec | git_write | send | delete | self | core | outside
    risk: str              # safe | ask | dangerous
    summary: str
    warning: str = ""

    def key(self) -> str:
        """Key used for 'allow for this session' grants."""
        if self.category == "exec":
            cmd = str(self.args.get("command", "")).strip().split()
            return f"exec:{cmd[0].lower() if cmd else ''}"
        return f"{self.category}:{self.tool}"


def redact(value: Any, depth: int = 0) -> Any:
    """Hide secrets and shorten long values before they reach logs or the UI."""
    if depth > 4:
        return "..."
    if isinstance(value, dict):
        return {k: ("***" if SECRET_KEY.search(str(k)) else redact(v, depth + 1)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v, depth + 1) for v in list(value)[:20]]
    if isinstance(value, str) and len(value) > 600:
        return value[:600] + f"... [{len(value) - 600} more chars]"
    return value


def _evora_home() -> Path:
    return Path(os.environ.get("EVORA_HOME") or (Path.home() / ".evora"))


class Classifier:
    """Decides what kind of action a tool call is."""

    def __init__(self, security: PermissionManager, protected: Optional[list[Path]] = None):
        self.security = security
        self.protected = [Path(p).resolve() for p in (protected or [])]

    def _path_arg(self, args: dict) -> Optional[str]:
        for k in ("path", "file_path", "directory", "dir", "target"):
            if args.get(k):
                return str(args[k])
        return None

    def _is_core(self, p: Path) -> bool:
        for prot in self.protected:
            try:
                p.relative_to(prot)
                return True
            except ValueError:
                if p == prot:
                    return True
        return False

    def _outside_workspace(self, p: Path) -> bool:
        try:
            p.relative_to(self.security.workspace_dir)
            return False
        except ValueError:
            return True

    def classify(self, tool: str, args: dict) -> Action:
        args = args or {}
        path_str = self._path_arg(args)
        resolved: Optional[Path] = None
        if path_str:
            try:
                base = Path(path_str)
                if not base.is_absolute():
                    base = self.security.workspace_dir / base
                resolved = base.resolve()
            except Exception:
                resolved = None

        # Anything touching EVORA's own controls is "core" and always asks.
        if resolved is not None and self._is_core(resolved):
            return Action(tool, args, "core", "dangerous",
                          f"{tool} on EVORA core/control file: {resolved}",
                          "This changes EVORA's own rules, logs or keys.")

        if tool in DELETE_TOOLS:
            return Action(tool, args, "delete", "dangerous",
                          f"Delete (move to trash): {path_str}",
                          "Goes to the trash first so it can be restored.")
        if tool in SELF_TOOLS:
            return Action(tool, args, "self", "dangerous", "EVORA wants to change its own code",
                          "Self-modification always needs the owner.")
        if tool in SEND_TOOLS:
            return Action(tool, args, "send", "dangerous", f"{tool}: " + json.dumps(redact(args))[:300],
                          "This leaves the computer (message, push, login or browser action).")

        if tool in READ_TOOLS:
            if resolved is not None and self._outside_workspace(resolved):
                return Action(tool, args, "outside", "ask", f"Read outside workspace: {resolved}")
            return Action(tool, args, "read", "safe", f"{tool} {path_str or ''}".strip())
        if tool in NETWORK_READ_TOOLS:
            return Action(tool, args, "network_read", "safe", f"{tool}: " + json.dumps(redact(args))[:200])

        if tool in WRITE_TOOLS:
            if resolved is not None and self._outside_workspace(resolved):
                return Action(tool, args, "outside", "ask", f"Write outside workspace: {resolved}",
                              "This path is outside the project workspace.")
            return Action(tool, args, "write", "ask", f"{tool} {path_str or ''}".strip())

        if tool in GIT_WRITE_TOOLS:
            return Action(tool, args, "git_write", "ask", f"{tool}: " + json.dumps(redact(args))[:200])

        if tool in EXEC_TOOLS:
            command = str(args.get("command", "")) if tool == "execute_command" else "run tests"
            if tool == "execute_command" and SENSITIVE_COMMAND.search(command):
                return Action(tool, args, "send", "dangerous", f"Run: {command}",
                              "Touches EVORA controls or sends data out.")
            risk = self.security.check_command_safety(command).value if tool == "execute_command" else "safe"
            return Action(tool, args, "exec", risk, f"Run: {command}",
                          "Potentially destructive command." if risk == "dangerous" else "")

        return Action(tool, args, "exec", "ask", f"Unknown tool {tool}: " + json.dumps(redact(args))[:200])


def decide(level: AutonomyLevel, action: Action) -> str:
    """Return 'auto' or 'ask'. Nothing is ever 'deny': the owner always has the last word."""
    cat, risk = action.category, action.risk
    if cat in ("core", "delete", "send", "self", "outside") or risk == "dangerous":
        return "ask"
    if cat in ("read", "network_read"):
        return "auto"
    if level == AutonomyLevel.OBSERVER:
        return "ask"
    if cat == "write":
        return "auto"
    if cat == "exec":
        if risk == "safe":
            return "auto" if level >= AutonomyLevel.COLLABORATOR else "ask"
        return "auto" if level >= AutonomyLevel.TRUSTED else "ask"
    if cat == "git_write":
        return "auto" if level >= AutonomyLevel.TRUSTED else "ask"
    return "ask"


# ---------------------------------------------------------------- audit log

class AuditLog:
    """Append-only JSONL; every line carries the hash of the previous line."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last = self._read_last_hash()

    def _read_last_hash(self) -> str:
        if not self.path.exists():
            return "0" * 64
        last = "0" * 64
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        last = json.loads(line).get("hash", last)
                    except json.JSONDecodeError:
                        pass
        return last

    @staticmethod
    def _digest(prev: str, body: dict) -> str:
        payload = prev + json.dumps(body, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def record(self, event: str, **fields: Any) -> dict:
        with self._lock:
            body = {"ts": round(time.time(), 3), "event": event, **{k: redact(v) for k, v in fields.items()}}
            body["prev"] = self._last
            body["hash"] = self._digest(self._last, {k: v for k, v in body.items() if k not in ("hash",)})
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(body, ensure_ascii=False) + "\n")
            self._last = body["hash"]
            return body

    def tail(self, limit: int = 100) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        out = []
        for ln in lines[-limit:]:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                out.append({"event": "corrupt-line"})
        return out

    def verify(self) -> tuple[bool, int]:
        """Return (intact, lines_checked). Any edit, removal or reorder breaks the chain."""
        prev = "0" * 64
        count = 0
        if not self.path.exists():
            return True, 0
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    return False, count
                stored = rec.pop("hash", None)
                if rec.get("prev") != prev or stored != self._digest(prev, rec):
                    return False, count
                prev = stored
                count += 1
        return True, count


# ---------------------------------------------------------------- trash

class Trash:
    """Deleting never destroys data: files go to a dated trash folder with a manifest."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def soft_delete(self, path: Path) -> dict:
        src = Path(path).resolve()
        if not src.exists():
            raise FileNotFoundError(str(src))
        trash_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        dest_dir = self.root / trash_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        shutil.move(str(src), str(dest))
        manifest = {"id": trash_id, "original": str(src), "stored": str(dest), "ts": time.time()}
        (dest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def list(self) -> list[dict]:
        out = []
        if self.root.exists():
            for d in sorted(self.root.iterdir()):
                m = d / "manifest.json"
                if m.exists():
                    try:
                        out.append(json.loads(m.read_text(encoding="utf-8")))
                    except json.JSONDecodeError:
                        pass
        return out

    def restore(self, trash_id: str) -> str:
        d = self.root / trash_id
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        original = Path(manifest["original"])
        if original.exists():
            raise FileExistsError(str(original))
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(manifest["stored"], str(original))
        shutil.rmtree(d, ignore_errors=True)
        return str(original)


# ---------------------------------------------------------------- approvals

@dataclass
class Pending:
    id: str
    action: Action
    created: float
    loop: Optional[asyncio.AbstractEventLoop] = None
    future: Optional[asyncio.Future] = None
    decision: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tool": self.action.tool,
            "category": self.action.category,
            "risk": self.action.risk,
            "summary": self.action.summary,
            "warning": self.action.warning,
            "args": redact(self.action.args),
            "created": self.created,
            "can_allow_session": self.action.category in ("write", "exec", "git_write")
                                 and self.action.risk != "dangerous",
        }


class ApprovalBroker:
    """Thread-safe: the agent awaits on its event loop while HTTP threads resolve."""

    def __init__(self, audit: AuditLog, timeout: float = 900.0):
        self.audit = audit
        self.timeout = timeout
        self._pending: dict[str, Pending] = {}
        self._lock = threading.Lock()

    def create(self, action: Action) -> Pending:
        loop = asyncio.get_running_loop()
        p = Pending(id=uuid.uuid4().hex[:10], action=action, created=time.time(), loop=loop,
                    future=loop.create_future())
        with self._lock:
            self._pending[p.id] = p
        self.audit.record("approval_requested", id=p.id, tool=action.tool, category=action.category,
                          risk=action.risk, summary=action.summary)
        return p

    async def wait(self, p: Pending) -> str:
        try:
            decision = await asyncio.wait_for(p.future, timeout=self.timeout)
        except asyncio.TimeoutError:
            decision = "deny"
            self.audit.record("approval_timeout", id=p.id)
        finally:
            with self._lock:
                self._pending.pop(p.id, None)
        return decision

    def resolve(self, approval_id: str, decision: str, by: str = "owner") -> bool:
        if decision not in ("allow", "allow_session", "deny"):
            return False
        with self._lock:
            p = self._pending.get(approval_id)
        if p is None or p.future is None or p.future.done():
            return False
        if decision == "allow_session" and not p.to_dict()["can_allow_session"]:
            decision = "allow"
        p.decision = decision
        self.audit.record("approval_resolved", id=approval_id, decision=decision, by=by, tool=p.action.tool)
        p.loop.call_soon_threadsafe(lambda: (not p.future.done()) and p.future.set_result(decision))
        return True

    def list_pending(self) -> list[dict]:
        with self._lock:
            return [p.to_dict() for p in self._pending.values() if p.future is not None and not p.future.done()]


# ---------------------------------------------------------------- facade

class Governance:
    """What the agent loop and web server talk to."""

    def __init__(self, security: PermissionManager, home: Optional[Path] = None,
                 level: Optional[AutonomyLevel] = None):
        self.security = security
        self.home = Path(home) if home else _evora_home()
        self.home.mkdir(parents=True, exist_ok=True)
        self.audit = AuditLog(self.home / "audit.jsonl")
        self.trash = Trash(self.home / "trash")
        self.broker = ApprovalBroker(self.audit)
        self.classifier = Classifier(security, protected=[
            self.home, Path(__file__).resolve(), Path(__file__).with_name("security.py").resolve(),
            Path(__file__).with_name("approval.py").resolve(),
        ])
        self._level_file = self.home / "governance.json"
        self._grants: set[str] = set()
        self._level = level if level is not None else self._load_level()

    # level is owner-controlled; the agent has no tool that can call set_level
    def _load_level(self) -> AutonomyLevel:
        try:
            data = json.loads(self._level_file.read_text(encoding="utf-8"))
            return AutonomyLevel(int(data.get("level", 1)))
        except Exception:
            return AutonomyLevel.ASSISTANT

    @property
    def level(self) -> AutonomyLevel:
        return self._level

    def set_level(self, level: int, by: str = "owner") -> AutonomyLevel:
        new = AutonomyLevel(int(level))
        old = self._level
        self._level = new
        self._grants.clear()
        self._level_file.write_text(json.dumps({"level": int(new)}), encoding="utf-8")
        self.audit.record("autonomy_changed", old=int(old), new=int(new), by=by)
        return new

    def describe(self) -> dict:
        return {
            "level": int(self._level),
            "name": LEVEL_INFO[self._level][0],
            "description": LEVEL_INFO[self._level][1],
            "levels": [{"level": int(l), "name": n, "description": d} for l, (n, d) in LEVEL_INFO.items()],
        }

    def assess(self, tool: str, args: dict) -> tuple[Action, str]:
        action = self.classifier.classify(tool, args)
        mode = decide(self._level, action)
        if mode == "ask" and action.key() in self._grants and action.risk != "dangerous":
            mode = "auto"
        return action, mode

    def remember_grant(self, action: Action) -> None:
        if action.risk != "dangerous" and action.category in ("write", "exec", "git_write"):
            self._grants.add(action.key())

    def log_tool(self, action: Action, decision: str, success: Optional[bool] = None, detail: str = "") -> None:
        self.audit.record("tool", tool=action.tool, category=action.category, risk=action.risk,
                          decision=decision, success=success, args=action.args, detail=detail[:300])
