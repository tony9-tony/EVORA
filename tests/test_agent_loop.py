"""Tests for the tool-using chat agent: governance gates every call."""

import asyncio

import pytest

from evora.agent_loop import ChatAgent
from evora.governance import AutonomyLevel, Governance
from evora.model import Message, ModelProvider, ModelResponse, Role, ToolCall
from evora.owner_tools import register_owner_tools
from evora.security import PermissionManager
from evora.tools import ToolRegistry


class FakeProvider(ModelProvider):
    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self._model = "fake-coder"

    def name(self):
        return "fake"

    def model(self):
        return self._model

    async def chat(self, request):
        raise NotImplementedError

    async def chat_stream(self, request):
        self.requests.append(request)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        yield ModelResponse(content=step.get("content", ""), tool_calls=step.get("tool_calls", []))


class FakeManager:
    def __init__(self, provider):
        self.active = provider


def build(tmp_path, script, level):
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    sec = PermissionManager(str(ws))
    gov = Governance(sec, home=tmp_path / "home", level=level)
    reg = ToolRegistry(sec)
    register_owner_tools(reg, gov)
    provider = FakeProvider(script)
    agent = ChatAgent(FakeManager(provider), reg, gov, workspace=ws)
    return agent, gov, ws, provider


async def drain(agent, text, gov=None, decide=None):
    """Run a turn. `decide` answers approval cards like the web UI would."""
    events = []
    messages = [Message(role=Role.SYSTEM, content="sys")]
    async for ev in agent.run(messages, text):
        events.append(ev)
        if ev["type"] == "approval_request" and decide:
            gov.broker.resolve(ev["approval"]["id"], decide)
    return events, messages


def kinds(events):
    return [e["type"] for e in events]


def call(name, **args):
    return {"tool_calls": [ToolCall(id=f"c-{name}", name=name, arguments=args)]}


def test_read_runs_without_asking(tmp_path):
    agent, gov, ws, _ = build(tmp_path, [call("read_file", path="a.txt"), {"content": "done"}], AutonomyLevel.OBSERVER)
    (ws / "a.txt").write_text("hello")
    events, messages = asyncio.run(drain(agent, "read it", gov))
    assert "approval_request" not in kinds(events)
    result = [e for e in events if e["type"] == "tool_result"][0]
    assert result["success"] and "hello" in result["output"]
    assert kinds(events)[-1] == "done"


def test_write_asks_at_observer_and_deny_blocks(tmp_path):
    agent, gov, ws, provider = build(
        tmp_path, [call("write_file", path="n.txt", content="x"), {"content": "ok denied"}], AutonomyLevel.OBSERVER)
    events, messages = asyncio.run(drain(agent, "write", gov, decide="deny"))
    assert "approval_request" in kinds(events)
    assert not (ws / "n.txt").exists()
    tool_msg = [m for m in messages if m.role == Role.TOOL][0]
    assert "denied" in tool_msg.content.lower()


def test_write_allowed_by_owner(tmp_path):
    agent, gov, ws, _ = build(
        tmp_path, [call("write_file", path="n.txt", content="x"), {"content": "done"}], AutonomyLevel.OBSERVER)
    asyncio.run(drain(agent, "write", gov, decide="allow"))
    assert (ws / "n.txt").read_text() == "x"


def test_write_is_automatic_at_assistant_level(tmp_path):
    agent, gov, ws, _ = build(
        tmp_path, [call("write_file", path="n.txt", content="y"), {"content": "done"}], AutonomyLevel.ASSISTANT)
    events, _ = asyncio.run(drain(agent, "write", gov))
    assert "approval_request" not in kinds(events)
    assert (ws / "n.txt").read_text() == "y"


def test_delete_always_asks_and_goes_to_trash(tmp_path):
    agent, gov, ws, _ = build(
        tmp_path, [call("delete_path", path="old.txt"), {"content": "done"}], AutonomyLevel.TRUSTED)
    (ws / "old.txt").write_text("keep me")
    events, _ = asyncio.run(drain(agent, "delete", gov, decide="allow"))
    assert "approval_request" in kinds(events)
    assert not (ws / "old.txt").exists()
    restored = gov.trash.restore(gov.trash.list()[0]["id"])
    assert (ws / "old.txt").read_text() == "keep me"


def test_dangerous_command_runs_only_with_owner_yes(tmp_path):
    script = [call("execute_command", command="echo shutdown-test"), {"content": "done"}]
    agent, gov, ws, _ = build(tmp_path, list(script), AutonomyLevel.TRUSTED)
    events, _ = asyncio.run(drain(agent, "go", gov, decide="allow"))
    req = [e for e in events if e["type"] == "approval_request"][0]
    assert req["approval"]["risk"] == "dangerous"
    res = [e for e in events if e["type"] == "tool_result"][0]
    assert res["success"] and "shutdown-test" in res["output"]

    agent2, gov2, _, _ = build(tmp_path / "second", list(script), AutonomyLevel.TRUSTED)
    events2, _ = asyncio.run(drain(agent2, "go", gov2, decide="deny"))
    res2 = [e for e in events2 if e["type"] == "tool_result"][0]
    assert not res2["success"]


def test_owner_grant_is_single_use(tmp_path):
    agent, gov, ws, _ = build(tmp_path, [call("execute_command", command="echo shutdown"), {"content": "d"}],
                              AutonomyLevel.TRUSTED)
    asyncio.run(drain(agent, "go", gov, decide="allow"))
    assert not gov.security.owner_approved_command("echo shutdown")


def test_core_files_always_need_owner(tmp_path):
    agent, gov, ws, _ = build(
        tmp_path, [call("write_file", path=str(tmp_path / "home" / "governance.json"), content='{"level":3}'),
                   {"content": "d"}], AutonomyLevel.TRUSTED)
    events, _ = asyncio.run(drain(agent, "raise my level", gov, decide="deny"))
    req = [e for e in events if e["type"] == "approval_request"][0]
    assert req["approval"]["category"] == "core"
    assert gov.level == AutonomyLevel.TRUSTED  # unchanged by the agent
    assert not (tmp_path / "home" / "governance.json").exists()


def test_outside_workspace_needs_owner_then_works(tmp_path):
    outside = tmp_path / "other.txt"
    outside.write_text("secret-ish")
    agent, gov, ws, _ = build(tmp_path, [call("read_file", path=str(outside)), {"content": "d"}], AutonomyLevel.TRUSTED)
    events, _ = asyncio.run(drain(agent, "read", gov, decide="allow"))
    assert [e for e in events if e["type"] == "approval_request"][0]["approval"]["category"] == "outside"
    assert "secret-ish" in [e for e in events if e["type"] == "tool_result"][0]["output"]


def test_model_without_tool_support_falls_back_to_text(tmp_path):
    agent, gov, ws, provider = build(
        tmp_path, [RuntimeError("400: model does not support tools"), {"content": "plain answer"}],
        AutonomyLevel.ASSISTANT)
    events, _ = asyncio.run(drain(agent, "hi", gov))
    assert "note" in kinds(events)
    assert [e for e in events if e["type"] == "content"][0]["content"] == "plain answer"
    assert provider.requests[1].tools == []


def test_unknown_tool_does_not_crash(tmp_path):
    agent, gov, ws, _ = build(tmp_path, [call("launch_missiles"), {"content": "d"}], AutonomyLevel.TRUSTED)
    events, _ = asyncio.run(drain(agent, "go", gov, decide="allow"))
    res = [e for e in events if e["type"] == "tool_result"][0]
    assert not res["success"]


def test_audit_trail_records_everything(tmp_path):
    agent, gov, ws, _ = build(
        tmp_path, [call("write_file", path="n.txt", content="x"), {"content": "d"}], AutonomyLevel.OBSERVER)
    asyncio.run(drain(agent, "w", gov, decide="allow"))
    events = [r["event"] for r in gov.audit.tail()]
    assert "approval_requested" in events and "approval_resolved" in events and "tool" in events
    assert gov.audit.verify()[0]


def test_step_limit(tmp_path):
    script = [call("list_dir", path=".") for _ in range(3)]
    agent, gov, ws, _ = build(tmp_path, script, AutonomyLevel.ASSISTANT)
    agent.max_steps = 3
    events, _ = asyncio.run(drain(agent, "loop", gov))
    assert any(e["type"] == "note" and "Stopped" in e["text"] for e in events)
