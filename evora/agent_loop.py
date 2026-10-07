"""
Chat agent loop: the model can call tools, every call goes through governance.

Events yielded (all plain dicts, ready for Server-Sent Events):
    content           {"content": "text chunk"}
    tool_start        {"id", "name", "args"}
    approval_request  {"approval": {...pending dict...}}
    approval_result   {"id", "decision"}
    tool_result       {"id", "name", "success", "output", "error"}
    note              {"text": "..."}
    done              {"response", "model", "response_time", "steps"}
    error             {"error": "..."}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from evora.governance import Governance, redact
from evora.model import ChatRequest, Message, Role, ToolCall, ToolResult as ModelToolResult, ToolSpec
from evora.tools import ToolRegistry

PATH_KEYS = ("path", "file_path", "directory", "dir", "target", "cwd")
MAX_TOOL_OUTPUT = 6000
MAX_HISTORY = 24


def tool_specs(registry: ToolRegistry) -> list[ToolSpec]:
    specs = []
    for name in registry.list():
        tool = registry.get(name)
        specs.append(ToolSpec(
            name=tool.name,
            description=tool.description,
            parameters={
                "type": "object",
                "properties": {k: {kk: vv for kk, vv in v.items() if kk != "required"} for k, v in tool.parameters.items()},
                "required": [k for k, v in tool.parameters.items() if v.get("required")],
            },
        ))
    return specs


GROUPS = {
    "web": (r"search|google|browse|website|web|tafuta|internet|http|news|habari za", {"browser_search", "browser_read", "web_search", "web_fetch"}),
    "github": (r"github|repo|issue|pull request|\bpr\b|commit", {"github_read", "github_write"}),
    "email": (r"e-?mail|barua|inbox|mail|reply|jibu", {"email_read", "email_send"}),
    "login": (r"log ?in|sign ?in|password|ingia|click|fill|form", {"browser_act", "browser_read"}),
    "delete": (r"delete|remove|futa|ondoa", {"delete_path"}),
}
CORE = {"read_file", "list_files", "list_directory", "write_file", "edit_file", "execute_command", "run_tests", "search_files", "grep"}


def select_specs(specs: list[ToolSpec], message: str, smart: bool = True) -> list[ToolSpec]:
    """Small local models are slow because every tool description is re-read each request.
    Send only the tools the message could plausibly need; plain chat gets none."""
    import re
    if not smart:
        return specs
    text = message or ""
    wanted: set[str] = set()
    for pattern, names in GROUPS.values():
        if re.search(pattern, text, re.I):
            wanted |= names
    from evora.speed import CODE_HINTS
    if CODE_HINTS.search(text) or re.search(r"file|folder|faili|read|soma|edit|run|test|fix|create|andika|tengeneza|list|taja|workspace", text, re.I):
        wanted |= CORE
    return [sp for sp in specs if sp.name in wanted]


def _coerce_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _trim(messages: list[Message], keep: int = MAX_HISTORY) -> list[Message]:
    """Keep the system prompt and the most recent turns so small local models stay fast."""
    if len(messages) <= keep + 1:
        return messages
    head = [m for m in messages[:1] if m.role == Role.SYSTEM]
    return head + messages[-keep:]


class ChatAgent:
    def __init__(self, manager, registry: ToolRegistry, governance: Governance, workspace: Path,
                 logger=None, max_steps: int = 8):
        self.manager = manager
        self.registry = registry
        self.governance = governance
        self.workspace = Path(workspace).resolve()
        self.logger = logger
        self.max_steps = max_steps
        self._no_tools_models: set[str] = set()

    def _absolute(self, args: dict) -> dict:
        fixed = dict(args)
        for key in PATH_KEYS:
            val = fixed.get(key)
            if isinstance(val, str) and val and not Path(val).is_absolute():
                fixed[key] = str((self.workspace / val).resolve())
        return fixed

    async def _run_tool(self, call: ToolCall):
        """Gate, (maybe) ask the owner, execute. Yields events; last item is ('result', ToolResult)."""
        args = self._absolute(_coerce_args(call.arguments))
        action, mode = self.governance.assess(call.name, args)
        yield {"type": "tool_start", "id": call.id, "name": call.name, "args": redact(args),
               "category": action.category, "risk": action.risk, "summary": action.summary}

        decision = "auto"
        if mode == "ask":
            pending = self.governance.broker.create(action)
            yield {"type": "approval_request", "approval": pending.to_dict()}
            decision = await self.governance.broker.wait(pending)
            yield {"type": "approval_result", "id": pending.id, "decision": decision}
            if decision == "deny":
                self.governance.log_tool(action, "denied")
                yield ("result", ModelToolResult(id=call.id, output="", error="The owner denied this action."))
                return
            if decision == "allow_session":
                self.governance.remember_grant(action)

        security = self.governance.security
        command = args.get("command") if call.name == "execute_command" else None
        path = None
        for key in PATH_KEYS:
            if isinstance(args.get(key), str):
                path = args[key]
                break
        # An owner "yes" is a single-use grant for exactly this command / path.
        owner_said_yes = mode == "ask" and decision in ("allow", "allow_session")
        if owner_said_yes:
            security.grant_owner(command=command, path=path)
        try:
            result = await self.registry.execute(call.name, **args)
        except TypeError as e:
            from evora.tools import ToolResult as ToolsResult
            result = ToolsResult(success=False, error=f"Bad arguments for {call.name}: {e}")
        except Exception as e:  # tool bugs must not kill the chat
            from evora.tools import ToolResult as ToolsResult
            result = ToolsResult(success=False, error=f"{call.name} failed: {e}")
        finally:
            if owner_said_yes:
                security.revoke_owner(command=command, path=path)

        self.last_screenshot = (getattr(result, "data", None) or {}).get("screenshot", "") if isinstance(getattr(result, "data", None), dict) else ""
        if not result.success:
            try:
                self.governance.tracker.record("tool_failure", call.name, result.error or "")
            except Exception:
                pass
        self.governance.log_tool(action, decision, success=result.success,
                                 detail=(result.error or result.output or "")[:200])
        output = (result.output or "")[:MAX_TOOL_OUTPUT]
        yield ("result", ModelToolResult(id=call.id, output=output, error=result.error or None))

    async def run(self, messages: list[Message], user_input: str, max_tokens: int = 1024,
                  temperature: float = 0.4) -> AsyncGenerator[dict, None]:
        start = time.time()
        provider = self.manager.active
        if provider is None:
            yield {"type": "error", "error": "No model provider is active."}
            return
        messages.append(Message(role=Role.USER, content=user_input))
        all_specs = tool_specs(self.registry)
        specs = select_specs(all_specs, user_input, smart=getattr(self, "smart_tools", True))
        full_text = ""
        steps = 0

        for step in range(self.max_steps):
            steps = step + 1
            model_name = provider.model()
            use_tools = model_name not in self._no_tools_models
            request = ChatRequest(messages=_trim(messages), tools=specs if use_tools else [],
                                  max_tokens=max_tokens, temperature=temperature, stream=True)
            content = ""
            calls: list[ToolCall] = []
            try:
                async for chunk in provider.chat_stream(request):
                    if chunk.content:
                        content += chunk.content
                        yield {"type": "content", "content": chunk.content}
                    for tc in chunk.tool_calls or []:
                        if all(tc.id != c.id for c in calls):
                            calls.append(tc)
            except Exception as e:
                text = str(e).lower()
                if use_tools and ("tool" in text) and ("support" in text or "400" in text):
                    self._no_tools_models.add(model_name)
                    yield {"type": "note", "text": f"{model_name} cannot call tools; answering in text only. "
                                                   "Switch to a coding model (e.g. qwen) for actions."}
                    continue
                try:
                    self.governance.tracker.record("model_error", model_name, str(e))
                except Exception:
                    pass
                yield {"type": "error", "error": str(e)}
                return

            if not calls:
                full_text += content
                messages.append(Message(role=Role.ASSISTANT, content=content))
                break

            for call in calls:
                messages.append(Message(role=Role.ASSISTANT, content=content, tool_call=call))
                content = ""  # only attach the text to the first call
                result = None
                async for ev in self._run_tool(call):
                    if isinstance(ev, tuple):
                        result = ev[1]
                    else:
                        yield ev
                ev_out = {"type": "tool_result", "id": call.id, "name": call.name,
                          "success": not result.error, "output": result.output[:2000], "error": result.error}
                if getattr(self, "last_screenshot", ""):
                    ev_out["screenshot"] = self.last_screenshot
                    self.last_screenshot = ""
                yield ev_out
                result.output = result.output or result.error or ""
                messages.append(Message(role=Role.TOOL, content=result.output, tool_result=result, name=call.name))
        else:
            yield {"type": "note", "text": f"Stopped after {self.max_steps} steps. Tell me to continue."}

        yield {"type": "done", "response": full_text, "model": provider.model(),
               "response_time": round(time.time() - start, 2), "steps": steps}
