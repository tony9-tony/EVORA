"""End-to-end tests of the owner console over real HTTP (fake model, real tools and governance)."""

import http.client
import json
import socket
import threading
import time

import pytest

import evora.chat_server as srv
from evora.chat import ChatSession
from evora.config import Config
from evora.governance import AutonomyLevel
from evora.model import ModelProvider, ModelResponse, ToolCall


class ScriptedProvider(ModelProvider):
    def __init__(self):
        self.turns = []  # list of lists of steps

    def name(self):
        return "fake"

    def model(self):
        return "fake-coder"

    async def chat(self, request):
        raise NotImplementedError

    async def chat_stream(self, request):
        if not self.turns:
            yield ModelResponse(content="hello from fake")
            return
        step = self.turns[0].pop(0)
        if not self.turns[0]:
            self.turns.pop(0)
        yield ModelResponse(content=step.get("content", ""), tool_calls=step.get("tool_calls", []))


@pytest.fixture
def console(tmp_path, monkeypatch):
    monkeypatch.setenv("EVORA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("EVORA_TOKEN", "test-token-123")
    monkeypatch.setenv("EVORA_WARMUP", "0")
    (tmp_path / "memory").mkdir()
    (tmp_path / "identity").mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()
    config = Config(workspace_dir=str(ws), log_level="ERROR", memory_dir=str(tmp_path / "memory"),
                    identity_dir=str(tmp_path / "identity"))
    session = ChatSession(config=config)
    provider = ScriptedProvider()
    session.manager.active  # ensure manager exists
    session.manager._providers["fake"] = provider
    session.manager._active = "fake"
    session.agent.manager = session.manager
    session.governance.set_level(AutonomyLevel.OBSERVER)
    srv.chat_session = session
    srv._event_loop = None
    srv._failed.clear()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = srv.ThreadedHTTPServer(("127.0.0.1", port), srv.ChatHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield port, session, provider, ws
    server.shutdown()


def req(port, path, method="GET", body=None, headers=None, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    h = dict(headers or {})
    data = raw
    if body is not None:
        data = json.dumps(body)
        h["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=h)
    r = conn.getresponse()
    payload = r.read()
    try:
        return r.status, json.loads(payload)
    except Exception:
        return r.status, payload


OWNER = {"X-Evora-Token": "test-token-123"}


def test_local_reads_need_no_token(console):
    port, *_ = console
    status, data = req(port, "/api/status")
    assert status == 200 and data["autonomy"]["level"] == 0


def test_remote_host_needs_token(console):
    port, *_ = console
    status, _ = req(port, "/api/status", headers={"Host": "abc.ngrok-free.dev"})
    assert status == 401
    status, data = req(port, "/api/status", headers={"Host": "abc.ngrok-free.dev", **OWNER})
    assert status == 200


def test_forwarded_requests_count_as_remote(console):
    port, *_ = console
    status, _ = req(port, "/api/status", headers={"X-Forwarded-For": "1.2.3.4"})
    assert status == 401


def test_owner_actions_need_token_even_locally(console):
    port, session, *_ = console
    assert req(port, "/api/autonomy", "POST", {"level": 3})[0] == 401
    assert session.governance.level == AutonomyLevel.OBSERVER
    assert req(port, "/api/approve", "POST", {"id": "x", "decision": "allow"})[0] == 401
    status, data = req(port, "/api/autonomy", "POST", {"level": 2}, headers=OWNER)
    assert status == 200 and data["level"] == 2
    assert req(port, "/api/autonomy", "POST", {"level": 9}, headers=OWNER)[0] == 400


def test_audit_requires_owner_and_is_intact(console):
    port, session, *_ = console
    assert req(port, "/api/audit")[0] == 401
    status, data = req(port, "/api/audit", headers=OWNER)
    assert status == 200 and data["intact"] is True


def test_wrong_token_locks_out_after_repeated_failures(console):
    port, *_ = console
    for _ in range(10):
        req(port, "/api/autonomy", "POST", {"level": 1}, headers={"X-Evora-Token": "wrong"})
    status, _ = req(port, "/api/autonomy", "POST", {"level": 1}, headers=OWNER)
    assert status == 401  # locked out for a while, even with the right token


def test_login_cookie_flow(console):
    port, *_ = console
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", "/?token=test-token-123")
    r = conn.getresponse()
    r.read()
    assert r.status == 302 and "evora_token=test-token-123" in r.getheader("Set-Cookie")
    status, _ = req(port, "/api/audit", headers={"Cookie": "evora_token=test-token-123"})
    assert status == 200


def test_index_served_and_remote_gets_login_page(console):
    port, *_ = console
    status, body = req(port, "/")
    assert status == 200 and b"EVORA" in body and b"approval" in body
    status, body = req(port, "/", headers={"Host": "evil.example"})
    assert status == 401 and b"Owner token" in body


def stream(port, message, on_event, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    from urllib.parse import quote
    conn.request("GET", "/api/chat/stream?message=" + quote(message), headers=headers or {})
    r = conn.getresponse()
    event = None
    events = []
    buf = b""
    while True:
        chunk = r.readline()
        if not chunk:
            break
        line = chunk.decode().strip()
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data = json.loads(line[5:])
            events.append((event, data))
            on_event(event, data)
            if event in ("done", "error"):
                break
    conn.close()
    return events


def test_full_approval_flow_over_http(console):
    port, session, provider, ws = console
    provider.turns.append([
        {"tool_calls": [ToolCall(id="t1", name="write_file", arguments={"path": "hello.txt", "content": "hi"})]},
        {"content": "written"},
    ])

    def on_event(name, data):
        if name == "approval_request":
            # while the agent is waiting, a pending approval is visible to the owner
            _, pending = req(port, "/api/approvals")
            assert pending["pending"][0]["id"] == data["approval"]["id"]
            # the agent (no token) cannot approve itself
            assert req(port, "/api/approve", "POST", {"id": data["approval"]["id"], "decision": "allow"})[0] == 401
            assert not (ws / "hello.txt").exists()
            # the owner approves
            req(port, "/api/approve", "POST", {"id": data["approval"]["id"], "decision": "allow"}, headers=OWNER)

    events = stream(port, "write a file", on_event)
    names = [e[0] for e in events]
    assert "approval_request" in names and names[-1] == "done"
    assert (ws / "hello.txt").read_text() == "hi"


def test_owner_deny_over_http(console):
    port, session, provider, ws = console
    provider.turns.append([
        {"tool_calls": [ToolCall(id="t1", name="write_file", arguments={"path": "no.txt", "content": "x"})]},
        {"content": "ok"},
    ])

    def on_event(name, data):
        if name == "approval_request":
            req(port, "/api/approve", "POST", {"id": data["approval"]["id"], "decision": "deny"}, headers=OWNER)

    events = stream(port, "write", on_event)
    assert not (ws / "no.txt").exists()
    result = [d for n, d in events if n == "tool_result"][0]
    assert result["success"] is False


def test_transcribe_falls_back_when_whisper_missing(console, monkeypatch):
    port, *_ = console
    monkeypatch.setattr("evora.voice._load", lambda: (_ for _ in ()).throw(
        srv.voice_mod.VoiceUnavailable("faster-whisper is not installed")))
    status, data = req(port, "/api/transcribe?lang=sw", "POST", raw=b"RIFFxxxx",
                       headers={"Content-Type": "audio/webm"})
    assert status == 501 and data["fallback"] == "browser"


def test_plain_stream_mode_still_works(console):
    port, session, provider, ws = console
    events = stream(port, "hi", lambda n, d: None)
    assert events[-1][0] == "done"
