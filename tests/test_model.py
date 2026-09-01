"""
Tests for the EVORA model abstraction layer.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from evora.model import _has_anthropic
from evora.model import (
    ModelManager,
    OpenAIProvider,
    AnthropicProvider,
    Message,
    Role,
    ChatRequest,
    ModelResponse,
    Usage,
    ToolSpec,
    ToolCall,
)
from evora.logger import Logger


class TestModelTypes:

    def test_role_enum(self):
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.TOOL == "tool"

    def test_message_creation(self):
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_message_to_dict(self):
        msg = Message(role=Role.USER, content="Hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"

    def test_tool_spec_creation(self):
        ts = ToolSpec(
            name="read_file",
            description="Read a file",
            parameters={"path": {"type": "string"}}
        )
        assert ts.name == "read_file"

    def test_usage_defaults(self):
        u = Usage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0

    def test_model_response_defaults(self):
        r = ModelResponse()
        assert r.content == ""
        assert r.tool_calls == []


class TestModelManager:

    def test_manager_creation(self):
        logger = Logger("test", "error")
        mgr = ModelManager(logger)
        assert mgr is not None

    def test_register_and_list(self):
        logger = Logger("test", "error")
        mgr = ModelManager(logger)
        mock_provider = MagicMock()
        mock_provider.name = lambda: "mock"
        mock_provider.model = lambda: "test-model"
        mgr.register("mock", mock_provider)
        assert "mock" in mgr.list_providers()

    def test_no_active_provider(self):
        logger = Logger("test", "error")
        mgr = ModelManager(logger)
        assert mgr.active is None


class TestOpenAIProvider:

    def test_provider_name(self):
        provider = OpenAIProvider(api_key="test", model="gpt-4o")
        assert provider.name() == "openai"

    def test_provider_model(self):
        provider = OpenAIProvider(api_key="test", model="gpt-4o")
        assert provider.model() == "gpt-4o"

    def test_provider_close(self):
        provider = OpenAIProvider(api_key="test", model="gpt-4o")
        assert provider.close() is None


class TestAnthropicProvider:

    @pytest.mark.skipif(
        not _has_anthropic,
        reason="anthropic package not installed"
    )
    def test_provider_name(self):
        provider = AnthropicProvider(api_key="test", model="claude-3")
        assert provider.name() == "anthropic"

    @pytest.mark.skipif(
        not _has_anthropic,
        reason="anthropic package not installed"
    )
    def test_provider_model(self):
        provider = AnthropicProvider(api_key="test", model="claude-3")
        assert provider.model() == "claude-3"
