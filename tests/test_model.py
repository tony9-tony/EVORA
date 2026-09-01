"""
Tests for the EVORA model abstraction layer.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from evora.model import _has_anthropic
from evora.model import _has_openai
from evora.model import (
    ModelManager,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
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

    @pytest.mark.skipif(
        not _has_openai,
        reason="openai package not installed"
    )
    def test_provider_name(self):
        provider = OpenAIProvider(api_key="test", model="gpt-4o")
        assert provider.name() == "openai"

    @pytest.mark.skipif(
        not _has_openai,
        reason="openai package not installed"
    )
    def test_provider_model(self):
        provider = OpenAIProvider(api_key="test", model="gpt-4o")
        assert provider.model() == "gpt-4o"

    @pytest.mark.skipif(
        not _has_openai,
        reason="openai package not installed"
    )
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


class TestOllamaProvider:

    def test_provider_name(self):
        provider = OllamaProvider()
        assert provider.name() == "ollama"

    def test_provider_model_default(self):
        provider = OllamaProvider()
        assert provider.model() == "qwen2.5-coder:latest"

    def test_provider_custom_model(self):
        provider = OllamaProvider(model="llama3")
        assert provider.model() == "llama3"

    def test_provider_default_base_url(self):
        provider = OllamaProvider()
        assert provider._base_url == "http://127.0.0.1:11434/v1"

    def test_provider_close_no_client(self):
        provider = OllamaProvider()
        assert provider.close() is None

    def test_build_payload_no_tools(self):
        provider = OllamaProvider()
        req = ChatRequest(
            messages=[Message(role=Role.USER, content="hi")],
            max_tokens=64,
            temperature=0.5,
        )
        payload = provider._build_payload(req)
        assert payload["model"] == "qwen2.5-coder:latest"
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 64
        assert payload["messages"] == [{"role": "user", "content": "hi"}]

    def test_build_payload_with_tools(self):
        provider = OllamaProvider()
        req = ChatRequest(
            messages=[Message(role=Role.USER, content="hi")],
            tools=[ToolSpec(name="read_file", description="r", parameters={"path": {"type": "string"}})],
        )
        payload = provider._build_payload(req)
        assert payload["tools"][0]["type"] == "function"
        assert payload["tools"][0]["function"]["name"] == "read_file"

    def test_build_payload_stream_flag(self):
        provider = OllamaProvider()
        req = ChatRequest(messages=[Message(role=Role.USER, content="hi")])
        assert "stream" not in provider._build_payload(req)
        assert provider._build_payload(req, stream=True)["stream"] is True

    def test_chat_connection_error(self):
        provider = OllamaProvider(base_url="http://127.0.0.1:19999/v1")
        req = ChatRequest(messages=[Message(role=Role.USER, content="hi")])
        with pytest.raises(ConnectionError) as exc_info:
            import asyncio
            asyncio.run(provider.chat(req))
        msg = str(exc_info.value)
        assert "Cannot connect to Ollama" in msg
        assert "http://127.0.0.1:19999/v1" in msg
        assert "ollama serve" in msg

    def test_chat_stream_connection_error(self):
        provider = OllamaProvider(base_url="http://127.0.0.1:19999/v1")
        req = ChatRequest(messages=[Message(role=Role.USER, content="hi")])

        async def run_stream():
            async for _ in provider.chat_stream(req):
                pass

        with pytest.raises(ConnectionError) as exc_info:
            import asyncio
            asyncio.run(run_stream())
        msg = str(exc_info.value)
        assert "Cannot connect to Ollama" in msg

    def test_switch_model(self):
        provider = OllamaProvider(model="qwen2.5-coder:latest")
        assert provider.model() == "qwen2.5-coder:latest"
        provider.switch_model("llama3:latest")
        assert provider.model() == "llama3:latest"

    def test_max_retries_attribute(self):
        provider = OllamaProvider(max_retries=5)
        assert provider._max_retries == 5

    def test_max_retries_default(self):
        provider = OllamaProvider()
        assert provider._max_retries == 2
