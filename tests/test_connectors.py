import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from evora.connectors.email_tools import EmailReadTool, EmailSendTool
from evora.connectors.github import GithubReadTool, GithubWriteTool
from evora.security import PermissionManager
from evora.selftrack import WeaknessTracker
from evora.vault import Vault, VaultLocked


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def sec(tmp_path):
    return PermissionManager(workspace_dir=str(tmp_path))


def test_vault_roundtrip(tmp_path):
    v = Vault(tmp_path / "v.json")
    with pytest.raises(VaultLocked):
        v.set("a", "b")
    assert v.unlock("correct horse")
    v.set("github_token", "ghp_secret")
    assert v.get("github_token") == "ghp_secret"
    assert "ghp_secret" not in (tmp_path / "v.json").read_text()
    v.lock()
    w = Vault(tmp_path / "v.json")
    assert not w.unlock("wrong passphrase")
    assert w.unlock("correct horse")
    assert w.names() == ["github_token"]
    assert w.delete("github_token")


def test_tracker_groups_and_marks(tmp_path):
    t = WeaknessTracker(tmp_path / "w.json")
    t.record("tool_failure", "web_fetch", "timeout after 30 s")
    t.record("tool_failure", "web_fetch", "timeout after 45 s")
    top = t.top()
    assert len(top) == 1 and top[0]["count"] == 2
    assert t.mark(top[0]["id"], "fixed")
    assert t.top() == []
    t.record("tool_failure", "web_fetch", "timeout after 5 s")
    assert len(t.top()) == 1  # came back


def test_github_read_and_write(sec):
    def handler(request: httpx.Request):
        if request.method == "POST":
            assert request.headers["authorization"] == "Bearer tok"
            return httpx.Response(201, json={"html_url": "https://github.com/o/r/issues/1"})
        if request.url.path == "/repos/o/r":
            return httpx.Response(200, json={"full_name": "o/r", "description": "d", "default_branch": "main",
                                             "stargazers_count": 3, "open_issues_count": 1, "private": False})
        return httpx.Response(404, text="nope")

    class V:
        def get(self, name):
            return "tok" if name == "github_token" else None

    r = GithubReadTool(sec, vault=V())
    r.transport = httpx.MockTransport(handler)
    out = run(r.execute(kind="repo", repo="o/r"))
    assert out.success and "default branch: main" in out.output
    bad = run(r.execute(kind="file", repo="o/r", path="x"))
    assert not bad.success and "404" in bad.error
    w = GithubWriteTool(sec, vault=V())
    w.transport = httpx.MockTransport(handler)
    ok = run(w.execute(kind="create_issue", repo="o/r", title="t", body="b"))
    assert ok.success and "issues/1" in ok.output


def test_github_write_needs_token(sec, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    res = run(GithubWriteTool(sec, vault=None).execute(kind="comment", repo="o/r", body="x", number=1))
    assert not res.success and "token" in res.error.lower()


class FakeVault:
    data = {"email_address": "me@example.com", "email_app_password": "pw"}

    def get(self, name):
        return self.data.get(name)


def test_email_send_uses_smtp(sec, monkeypatch):
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=0):
            sent["host"] = host

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, u, p):
            sent["login"] = (u, p)

        def send_message(self, msg):
            sent["msg"] = msg

    monkeypatch.setattr("evora.connectors.email_tools.smtplib.SMTP_SSL", FakeSMTP)
    tool = EmailSendTool(sec, vault=FakeVault())
    res = run(tool.execute(to="a@b.com", subject="Hi", body="Hello", in_reply_to="<id1>"))
    assert res.success
    assert sent["login"] == ("me@example.com", "pw")
    assert sent["msg"]["In-Reply-To"] == "<id1>" and sent["msg"]["To"] == "a@b.com"
    assert not run(tool.execute(to="nobody", subject="s", body="b")).success


def test_email_read_without_credentials(sec):
    class Empty:
        def get(self, n):
            return None
    res = run(EmailReadTool(sec, vault=Empty()).execute(action="list"))
    assert not res.success and "vault" in res.error.lower()


def test_email_read_lists(sec, monkeypatch):
    raw = b"From: x@y.com\r\nSubject: Hello\r\nDate: Mon, 1 Jan 2026 10:00:00 +0000\r\n\r\n"

    class FakeIMAP:
        def __init__(self, host):
            pass

        def login(self, u, p):
            pass

        def select(self, *a, **k):
            return "OK", [b"1"]

        def search(self, *a):
            return "OK", [b"1 2"]

        def fetch(self, i, spec):
            return "OK", [(b"1", raw)]

        def logout(self):
            pass

    monkeypatch.setattr("evora.connectors.email_tools.imaplib.IMAP4_SSL", FakeIMAP)
    res = run(EmailReadTool(sec, vault=FakeVault()).execute(action="list", limit=5))
    assert res.success and "Hello" in res.output and "x@y.com" in res.output


class _Page(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"<html><title>T1</title><body><h1>Hello EVORA</h1></body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_browser_read_and_act(sec, monkeypatch):
    pytest.importorskip("playwright")
    monkeypatch.setenv("EVORA_BROWSER_HEADLESS", "1")
    monkeypatch.setenv("EVORA_BROWSER_CHANNEL", "")
    from evora.connectors.browser import BrowserActTool, BrowserReadTool, BrowserSession
    srv = HTTPServer(("127.0.0.1", 0), _Page)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/"

    async def go():
        session = BrowserSession()
        try:
            read = await BrowserReadTool(sec, session=session).execute(url=url)
            act = await BrowserActTool(sec, session=session).execute(steps=[{"action": "goto", "url": url}])
            bad = await BrowserReadTool(sec, session=session).execute(url="file:///etc/passwd")
            return read, act, bad
        finally:
            await session.close()

    try:
        try:
            read, act, bad = run(go())
        except Exception as e:  # no browser in this environment
            pytest.skip(f"browser unavailable: {e}")
    finally:
        srv.shutdown()
    if not read.success and "launch" in (read.error or "").lower():
        pytest.skip(read.error)
    assert read.success and "Hello EVORA" in read.output and read.data["screenshot"]
    assert act.success
    assert not bad.success
