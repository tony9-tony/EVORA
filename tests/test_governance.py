"""Tests for owner governance: levels, classification, audit chain, trash, approvals."""

import asyncio
import json
import threading

import pytest

from evora.governance import (
    Action, ApprovalBroker, AuditLog, AutonomyLevel, Classifier, Governance, Trash, decide, redact,
)
from evora.security import PermissionManager


@pytest.fixture
def gov(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    sec = PermissionManager(str(ws))
    return Governance(sec, home=tmp_path / "home", level=AutonomyLevel.ASSISTANT), ws


def classify(gov_ws, tool, **args):
    g, ws = gov_ws
    return g.classifier.classify(tool, args)


class TestClassification:
    def test_reads_are_safe(self, gov):
        a = classify(gov, "read_file", path="a.py")
        assert (a.category, a.risk) == ("read", "safe")

    def test_read_outside_workspace_asks(self, gov, tmp_path):
        a = classify(gov, "read_file", path=str(tmp_path / "elsewhere.txt"))
        assert a.category == "outside"

    def test_write_in_workspace(self, gov):
        assert classify(gov, "write_file", path="x.py", content="1").category == "write"

    def test_delete_is_dangerous(self, gov):
        a = classify(gov, "delete_path", path="x.py")
        assert (a.category, a.risk) == ("delete", "dangerous")

    def test_dangerous_command(self, gov):
        a = classify(gov, "execute_command", command="rm -rf /")
        assert a.risk == "dangerous"

    def test_push_and_control_endpoints_are_sensitive(self, gov):
        assert classify(gov, "execute_command", command="git push origin main").category == "send"
        assert classify(gov, "execute_command", command="curl -X POST localhost:8080/api/autonomy").category == "send"
        assert classify(gov, "execute_command", command="type web_token").category == "send"

    def test_core_files_are_protected(self, gov):
        g, ws = gov
        a = classify(gov, "write_file", path=str(g.home / "governance.json"), content="{}")
        assert a.category == "core"

    def test_self_modification_asks(self, gov):
        assert classify(gov, "self_improve").category == "self"


class TestPolicy:
    def act(self, category, risk="ask"):
        return Action("t", {}, category, risk, "s")

    def test_owner_always_has_last_word(self):
        for level in AutonomyLevel:
            for cat in ("core", "delete", "send", "self", "outside"):
                assert decide(level, self.act(cat)) == "ask"
            assert decide(level, self.act("exec", "dangerous")) == "ask"

    def test_levels(self):
        assert decide(AutonomyLevel.OBSERVER, self.act("read", "safe")) == "auto"
        assert decide(AutonomyLevel.OBSERVER, self.act("write")) == "ask"
        assert decide(AutonomyLevel.ASSISTANT, self.act("write")) == "auto"
        assert decide(AutonomyLevel.ASSISTANT, self.act("exec", "safe")) == "ask"
        assert decide(AutonomyLevel.COLLABORATOR, self.act("exec", "safe")) == "auto"
        assert decide(AutonomyLevel.COLLABORATOR, self.act("exec", "ask")) == "ask"
        assert decide(AutonomyLevel.TRUSTED, self.act("exec", "ask")) == "auto"
        assert decide(AutonomyLevel.TRUSTED, self.act("git_write")) == "auto"

    def test_session_grant_never_covers_dangerous(self, gov):
        g, ws = gov
        action, mode = g.assess("execute_command", {"command": "pip install x"})
        assert mode == "ask"
        g.remember_grant(action)
        assert g.assess("execute_command", {"command": "pip install y"})[1] == "auto"
        danger, _ = g.assess("execute_command", {"command": "rm -rf build"})
        g.remember_grant(danger)
        assert g.assess("execute_command", {"command": "rm -rf other"})[1] == "ask"

    def test_level_is_persisted(self, gov, tmp_path):
        g, ws = gov
        g.set_level(3)
        again = Governance(PermissionManager(str(ws)), home=g.home)
        assert again.level == AutonomyLevel.TRUSTED


class TestAudit:
    def test_chain_verifies_and_detects_tampering(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        for i in range(5):
            log.record("tool", n=i)
        ok, count = log.verify()
        assert ok and count == 5
        lines = (tmp_path / "audit.jsonl").read_text().splitlines()
        tampered = json.loads(lines[2])
        tampered["n"] = 99
        lines[2] = json.dumps(tampered)
        (tmp_path / "audit.jsonl").write_text("\n".join(lines) + "\n")
        assert AuditLog(tmp_path / "audit.jsonl").verify()[0] is False

    def test_removed_line_breaks_chain(self, tmp_path):
        log = AuditLog(tmp_path / "a.jsonl")
        for i in range(4):
            log.record("x", n=i)
        lines = (tmp_path / "a.jsonl").read_text().splitlines()
        del lines[1]
        (tmp_path / "a.jsonl").write_text("\n".join(lines) + "\n")
        assert AuditLog(tmp_path / "a.jsonl").verify()[0] is False

    def test_secrets_are_redacted(self, tmp_path):
        log = AuditLog(tmp_path / "a.jsonl")
        rec = log.record("login", args={"user": "me", "password": "hunter2", "api_key": "k"})
        assert rec["args"]["password"] == "***" and rec["args"]["api_key"] == "***"
        assert "hunter2" not in (tmp_path / "a.jsonl").read_text()

    def test_resumes_chain_after_restart(self, tmp_path):
        AuditLog(tmp_path / "a.jsonl").record("one")
        log2 = AuditLog(tmp_path / "a.jsonl")
        log2.record("two")
        assert log2.verify() == (True, 2)


class TestTrash:
    def test_delete_is_recoverable(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("precious")
        trash = Trash(tmp_path / "trash")
        m = trash.soft_delete(f)
        assert not f.exists()
        assert trash.list()[0]["id"] == m["id"]
        trash.restore(m["id"])
        assert f.read_text() == "precious"

    def test_folder_delete(self, tmp_path):
        d = tmp_path / "proj"
        (d / "sub").mkdir(parents=True)
        (d / "sub" / "a.txt").write_text("1")
        trash = Trash(tmp_path / "trash")
        m = trash.soft_delete(d)
        assert not d.exists()
        trash.restore(m["id"])
        assert (d / "sub" / "a.txt").exists()


class TestBroker:
    def test_resolve_from_another_thread(self, tmp_path):
        broker = ApprovalBroker(AuditLog(tmp_path / "a.jsonl"), timeout=5)

        async def scenario():
            p = broker.create(Action("write_file", {"path": "x"}, "write", "ask", "write x"))
            assert broker.list_pending()[0]["id"] == p.id
            threading.Timer(0.1, lambda: broker.resolve(p.id, "allow")).start()
            return await broker.wait(p)

        assert asyncio.run(scenario()) == "allow"

    def test_timeout_denies(self, tmp_path):
        broker = ApprovalBroker(AuditLog(tmp_path / "a.jsonl"), timeout=0.2)

        async def scenario():
            p = broker.create(Action("write_file", {}, "write", "ask", "w"))
            return await broker.wait(p)

        assert asyncio.run(scenario()) == "deny"

    def test_allow_session_downgrades_for_dangerous(self, tmp_path):
        broker = ApprovalBroker(AuditLog(tmp_path / "a.jsonl"), timeout=5)

        async def scenario():
            p = broker.create(Action("delete_path", {}, "delete", "dangerous", "d"))
            threading.Timer(0.05, lambda: broker.resolve(p.id, "allow_session")).start()
            return await broker.wait(p)

        assert asyncio.run(scenario()) == "allow"

    def test_unknown_id_and_bad_decision(self, tmp_path):
        broker = ApprovalBroker(AuditLog(tmp_path / "a.jsonl"))
        assert broker.resolve("nope", "allow") is False
        assert broker.resolve("nope", "maybe") is False


def test_redact_shortens_long_values():
    assert len(redact("x" * 5000)) < 700
