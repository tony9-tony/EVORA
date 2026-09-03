"""
Tests for the EVORA CLI provider selection and chat server.
"""

import asyncio
import json
import socket
import subprocess
import sys
import threading
import urllib.request
import urllib.error
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evora.config import Config, ProviderConfig
from evora.cli import _build_model_manager, _chat_turn, async_run
from evora.identity import AuthorityLevel, Identity
from evora.logger import Logger
from evora.memory import LongTermMemoryEntry, RetrievalResult
from evora.model import ModelResponse, _has_openai, _has_anthropic, _has_httpx, ChatRequest, Message, Role


@pytest.fixture
def logger():
    return Logger("test", "ERROR")


def _config(**overrides):
    providers = {
        "openai": ProviderConfig(name="openai", model="gpt-4o", base_url="https://api.openai.com/v1"),
        "ollama": ProviderConfig(name="ollama", model="qwen2.5-coder:latest", base_url="http://127.0.0.1:11434/v1", timeout=180.0),
        "anthropic": ProviderConfig(name="anthropic", model="claude-3-5-sonnet-20241022", base_url="https://api.anthropic.com"),
    }
    if "providers" in overrides:
        providers = overrides.pop("providers")
    return Config(
        api_key=overrides.get("api_key", ""),
        provider=overrides.get("provider", ""),
        model=overrides.get("model", "gpt-4o"),
        base_url=overrides.get("base_url", "https://api.openai.com/v1"),
        workspace_dir=overrides.get("workspace_dir", "."),
        providers=providers,
    )


class TestBuildModelManager:

    def test_no_keys_selects_ollama(self, logger):
        cfg = _config(api_key="")
        mgr = _build_model_manager(cfg, logger)
        assert mgr.active is not None
        assert mgr.active.name() == "ollama"

    def test_explicit_ollama_selection(self, logger):
        cfg = _config(provider="ollama")
        mgr = _build_model_manager(cfg, logger)
        assert mgr.active is not None
        assert mgr.active.name() == "ollama"

    def test_openai_key_selects_openai(self, logger):
        if not _has_openai:
            pytest.skip("openai package not installed")
        cfg = _config(api_key="sk-test-123", provider="")
        mgr = _build_model_manager(cfg, logger)
        assert mgr.active is not None
        assert mgr.active.name() == "openai"

    def test_explicit_openai_without_key_falls_back(self, logger):
        cfg = _config(api_key="", provider="openai")
        mgr = _build_model_manager(cfg, logger)
        available = mgr.list_providers()
        assert "openai" not in available

    def test_provider_override_arg(self, logger):
        cfg = _config(api_key="", provider="")
        mgr = _build_model_manager(cfg, logger, provider_override="ollama")
        assert mgr.active is not None
        assert mgr.active.name() == "ollama"

    def test_ollama_custom_model_and_url(self, logger):
        cfg = _config(provider="ollama", providers={
            "ollama": ProviderConfig(name="ollama", model="custom-model", base_url="http://localhost:11434/v1", timeout=10.0),
        })
        mgr = _build_model_manager(cfg, logger)
        assert mgr.active.name() == "ollama"
        assert mgr.active.model() == "custom-model"

    def test_no_httpx_falls_back_to_mock(self, logger, monkeypatch):
        monkeypatch.setattr("evora.model._has_httpx", False)
        cfg = _config(api_key="", provider="")
        mgr = _build_model_manager(cfg, logger)
        assert mgr.active is not None
        assert mgr.active.name() == "mock"


class TestChatTurn:

    def test_turn_appends_messages(self):
        manager = MagicMock()
        manager.chat = AsyncMock(return_value=ModelResponse(content="hi", provider="mock", model="m"))
        messages = [Message(role=Role.SYSTEM, content="sys")]
        memory_service = MagicMock()
        memory_service.retrieve_relevant = MagicMock(return_value=[])
        logger = MagicMock()

        response = asyncio.run(_chat_turn(manager, messages, "hello", memory_service, "proj", logger))

        assert len(messages) == 3
        assert messages[1] == Message(role=Role.USER, content="hello")
        assert messages[2] == Message(role=Role.ASSISTANT, content="hi")
        assert response.content == "hi"

    def test_turn_injects_memory_context(self):
        manager = MagicMock()
        manager.chat = AsyncMock(return_value=ModelResponse(content="hi", provider="mock", model="m"))
        messages = [Message(role=Role.SYSTEM, content="sys")]

        entry = LongTermMemoryEntry(memory_type="preference", content="user likes cats", importance=0.5)
        result = RetrievalResult(entry=entry, score=0.9)
        memory_service = MagicMock()
        memory_service.retrieve_relevant = MagicMock(return_value=[result])
        logger = MagicMock()

        asyncio.run(_chat_turn(manager, messages, "hello", memory_service, "proj", logger))

        assert len(messages) == 3  # persistent history stays clean
        assert messages[1] == Message(role=Role.USER, content="hello")
        assert messages[2].role == Role.ASSISTANT
        # Verify the model request included memory context
        called_request = manager.chat.call_args[0][0]
        assert any(m.role == Role.SYSTEM and "user likes cats" in m.content for m in called_request.messages)

    def test_turn_handles_memory_failure(self):
        manager = MagicMock()
        manager.chat = AsyncMock(return_value=ModelResponse(content="hi", provider="mock", model="m"))
        messages = [Message(role=Role.SYSTEM, content="sys")]
        memory_service = MagicMock()
        memory_service.retrieve_relevant = MagicMock(side_effect=RuntimeError("db down"))
        logger = MagicMock()

        response = asyncio.run(_chat_turn(manager, messages, "hello", memory_service, "proj", logger))
        assert response.content == "hi"
        assert len(messages) == 3  # no context injected

    def test_turn_handles_model_failure(self):
        manager = MagicMock()
        manager.chat = AsyncMock(side_effect=ConnectionError("ollama down"))
        messages = [Message(role=Role.SYSTEM, content="sys")]
        memory_service = MagicMock()
        memory_service.retrieve_relevant = MagicMock(return_value=[])
        logger = MagicMock()

        with pytest.raises(ConnectionError):
            asyncio.run(_chat_turn(manager, messages, "hello", memory_service, "proj", logger))


class TestChatServer:

    def _free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _start_server(self, tmp_path, port):
        import evora.chat_server as chat_server_mod
        from evora.chat import ChatSession
        from evora.chat_server import ChatHandler, ThreadedHTTPServer

        config = Config(
            api_key="",
            provider="",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            workspace_dir=str(tmp_path),
            log_level="ERROR",
            log_file="",
            memory_dir=str(tmp_path / "memory"),
            identity_dir=str(tmp_path / "identity"),
            providers={
                "ollama": ProviderConfig(name="ollama", model="qwen2.5-coder:latest", base_url="http://127.0.0.1:11434/v1", timeout=180.0),
            },
        )
        (tmp_path / "memory").mkdir(exist_ok=True)
        (tmp_path / "identity").mkdir(exist_ok=True)

        session = ChatSession(config=config)
        chat_server_mod.chat_session = session
        # Reset event loop state
        chat_server_mod._event_loop = None
        chat_server_mod._event_loop_thread = None

        server = ThreadedHTTPServer(("127.0.0.1", port), ChatHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        return server, session, t

    def _request(self, port, path, method="GET", body=None, parse_json=True):
        url = f"http://127.0.0.1:{port}{path}"
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read()
                if parse_json:
                    return resp.status, json.loads(raw)
                return resp.status, raw
        except urllib.error.HTTPError as e:
            raw = e.read()
            if parse_json:
                try:
                    return e.code, json.loads(raw)
                except Exception:
                    return e.code, {"error": raw.decode()[:200]}
            return e.code, raw

    def test_server_serves_html(self, tmp_path):
        port = self._free_port()
        server, session, t = self._start_server(tmp_path, port)
        try:
            status, data = self._request(port, "/", parse_json=False)
            assert status == 200
            assert b"<!DOCTYPE html>" in data
        finally:
            session.close()
            server.shutdown()

    def test_server_status(self, tmp_path):
        port = self._free_port()
        server, session, t = self._start_server(tmp_path, port)
        try:
            status, data = self._request(port, "/api/status")
            assert status == 200
            assert data["provider"] == "ollama"
            assert data["model"] == "qwen2.5-coder:latest"
        finally:
            session.close()
            server.shutdown()

    def test_server_chat_turn(self, tmp_path):
        port = self._free_port()
        server, session, t = self._start_server(tmp_path, port)
        try:
            # Patch the model manager to avoid real Ollama call
            mock_mgr = MagicMock()
            mock_mgr.active = MagicMock()
            mock_mgr.active.name.return_value = "ollama"
            mock_mgr.active.model.return_value = "qwen2.5-coder:latest"

            async def fake_chat(request):
                return ModelResponse(content="test response", provider="ollama", model="qwen2.5-coder:latest")
            mock_mgr.chat = fake_chat
            session.manager = mock_mgr

            status, data = self._request(port, "/api/chat", method="POST", body={"message": "hi"})
            assert status == 200
            assert data["response"] == "test response"
            assert data["provider"] == "ollama"
            assert data["identity"] == "guest"
        finally:
            session.close()
            server.shutdown()

    def test_server_chat_empty_message(self, tmp_path):
        port = self._free_port()
        server, session, t = self._start_server(tmp_path, port)
        try:
            status, data = self._request(port, "/api/chat", method="POST", body={"message": ""})
            assert status == 200
            assert "error" in data
        finally:
            session.close()
            server.shutdown()

    def test_server_clear(self, tmp_path):
        port = self._free_port()
        server, session, t = self._start_server(tmp_path, port)
        try:
            status, data = self._request(port, "/api/clear", method="POST")
            assert status == 200
            assert data.get("status") == "ok"
        finally:
            session.close()
            server.shutdown()

    def test_server_repeated_chat_no_loop_closed(self, tmp_path):
        """Verify multiple chat requests work without 'Event loop is closed' error."""
        port = self._free_port()
        server, session, t = self._start_server(tmp_path, port)
        try:
            mock_mgr = MagicMock()
            mock_mgr.active = MagicMock()
            mock_mgr.active.name.return_value = "ollama"
            mock_mgr.active.model.return_value = "qwen2.5-coder:latest"

            async def fake_chat(request):
                return ModelResponse(content="response", provider="ollama", model="qwen2.5-coder:latest")
            mock_mgr.chat = fake_chat
            session.manager = mock_mgr

            # Send multiple messages in sequence
            for i in range(5):
                status, data = self._request(port, "/api/chat", method="POST", body={"message": f"msg {i}"})
                assert status == 200
                assert data["response"] == "response"
                assert "error" not in data

            # Verify event loop is still alive
            import evora.chat_server as chat_server_mod
            assert chat_server_mod._event_loop is not None
            assert not chat_server_mod._event_loop.is_closed()
        finally:
            session.close()
            server.shutdown()
            import evora.chat_server as chat_server_mod
            chat_server_mod._event_loop = None
            chat_server_mod._event_loop_thread = None

    def test_chat_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "evora", "chat", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "Interactive chat" in result.stdout or "chat" in result.stdout


class TestAsyncRunConstructionOrder:
    """Regression tests for the async_run construction-order bug.

    Previously, identity_service was referenced before assignment in
    async_run, causing UnboundLocalError. These tests verify the
    function reaches the execution layer instead of crashing.
    """

    def _make_args(self, tmp_path):
        """Create a minimal args namespace for async_run."""
        from argparse import Namespace
        return Namespace(
            request="Create a test file",
            workspace=str(tmp_path),
            auto_approve=True,
            provider=None,
            timeout=60,
            max_retries=1,
        )

    @pytest.fixture
    def patched_run_env(self, tmp_path, monkeypatch):
        """Patch external dependencies so async_run can be tested in isolation."""
        from evora.identity import IdentityService
        from evora.memory import Memory
        from evora.security import PermissionManager
        from evora.approval import ApprovalSystem
        from evora.tools import ToolRegistry
        from evora.analyzer import ProjectAnalyzer
        from evora.planner import Planner

        identity_dir = str(tmp_path / "identity")
        memory_dir = str(tmp_path / "memory")
        Path(identity_dir).mkdir(parents=True, exist_ok=True)
        Path(memory_dir).mkdir(parents=True, exist_ok=True)

        def fake_load_config():
            return Config(
                api_key="",
                provider="",
                model="gpt-4o",
                base_url="https://api.openai.com/v1",
                workspace_dir=str(tmp_path),
                log_level="ERROR",
                log_file="",
                memory_dir=memory_dir,
                identity_dir=identity_dir,
                providers={
                    "ollama": ProviderConfig(name="ollama", model="test", base_url="http://127.0.0.1:11434/v1"),
                },
            )

        def fake_build_model_manager(config, logger, provider_override=None):
            from evora.model import ModelManager
            mgr = ModelManager(logger)
            from evora.cli import MockModelProvider
            mgr.register("mock", MockModelProvider())
            mgr.set_active("mock")
            return mgr

        # Bootstrap creator for identity
        def bootstrap():
            svc = IdentityService(identity_dir=identity_dir)
            if svc.get_creator() is None:
                svc.bootstrap_creator_with_profile(name="test_creator", display_name="Test Creator")
            return svc

        bootstrap()

        monkeypatch.setattr("evora.cli.load_config", fake_load_config)
        monkeypatch.setattr("evora.cli._build_model_manager", fake_build_model_manager)

        return tmp_path

    def test_async_run_does_not_raise_unbound_local(self, patched_run_env):
        """async_run must not raise UnboundLocalError for identity_service."""
        args = self._make_args(patched_run_env)
        try:
            result = asyncio.run(async_run(args))
            assert isinstance(result, int)
        except UnboundLocalError as e:
            pytest.fail(f"async_run raised UnboundLocalError: {e}")

    def test_async_run_reaches_execution_layer(self, patched_run_env):
        """async_run must construct all services and reach agent.run."""
        args = self._make_args(patched_run_env)
        with patch.object(__import__("evora.autonomous", fromlist=["AutonomousAgent"]).AutonomousAgent, "run",
                          new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Mock report: task completed"
            try:
                result = asyncio.run(async_run(args))
                assert isinstance(result, int)
                assert mock_run.called
            except UnboundLocalError as e:
                pytest.fail(f"async_run raised UnboundLocalError: {e}")

    def test_async_run_identity_service_available(self, patched_run_env, monkeypatch):
        """Verify identity_service is created before use in async_run."""
        from evora.cli import async_run
        args = self._make_args(patched_run_env)

        captured = {}

        original_agent_init = __import__("evora.autonomous", fromlist=["AutonomousAgent"]).AutonomousAgent.__init__

        def spy_init(self, *a, **kwargs):
            if "identity_service" in kwargs:
                captured["identity_service"] = kwargs["identity_service"]
            return original_agent_init(self, *a, **kwargs)

        monkeypatch.setattr(
            "evora.autonomous.AutonomousAgent.__init__", spy_init
        )

        with patch.object(__import__("evora.autonomous", fromlist=["AutonomousAgent"]).AutonomousAgent, "run",
                          new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "OK"
            try:
                asyncio.run(async_run(args))
            except UnboundLocalError:
                pytest.fail("UnboundLocalError - identity_service referenced before assignment")

        assert "identity_service" in captured
        from evora.identity import IdentityService
        assert isinstance(captured["identity_service"], IdentityService)
