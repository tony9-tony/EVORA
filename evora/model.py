"""
Model abstraction layer for EVORA.

Provides a unified interface for interacting with different AI model providers.
New providers can be added by implementing the ModelProvider interface.

Architecture:
    EVORA -> ModelInterface -> AIModelProvider -> [OpenAI, Anthropic, Ollama, ...]
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Optional

try:
    from openai import AsyncOpenAI
    _has_openai = True
except ImportError:
    _has_openai = False

try:
    import anthropic
    _has_anthropic = True
except ImportError:
    _has_anthropic = False


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    id: str
    output: str
    error: Optional[str] = None


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role.value}
        if self.content:
            d["content"] = self.content
        if self.name:
            d["name"] = self.name
        if self.tool_call:
            d["tool_calls"] = [{
                "id": self.tool_call.id,
                "type": "function",
                "function": {
                    "name": self.tool_call.name,
                    "arguments": self.tool_call.arguments,
                }
            }]
        if self.tool_result:
            d["role"] = "tool"
            d["tool_call_id"] = self.tool_result.id
            d["content"] = self.tool_result.output
        return d


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ModelResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    provider: str = ""
    model: str = ""
    raw: Optional[dict[str, Any]] = None


@dataclass
class ChatRequest:
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    max_tokens: int = 4096
    temperature: float = 0.7
    stream: bool = False


class ModelProvider(ABC):
    """Abstract base class for AI model providers."""

    @abstractmethod
    def name(self) -> str:
        """Return the provider name (e.g. 'openai')."""

    @abstractmethod
    def model(self) -> str:
        """Return the model identifier."""

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ModelResponse:
        """Send a chat request and return the response."""

    async def chat_stream(self, request: ChatRequest) -> AsyncGenerator[ModelResponse, None]:
        """Stream responses. Override for streaming support."""
        resp = await self.chat(request)
        yield resp

    def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass


class OpenAIProvider(ModelProvider):
    """OpenAI-compatible model provider."""

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = "https://api.openai.com/v1", timeout: float = 30.0):
        if not _has_openai:
            raise ImportError("openai package not installed. Run: pip install openai")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout = timeout
        self._client: Optional[AsyncOpenAI] = None

    def name(self) -> str:
        return "openai"

    def model(self) -> str:
        return self._model

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    async def chat(self, request: ChatRequest) -> ModelResponse:
        client = self._get_client()

        messages = [m.to_dict() for m in request.messages]

        tools = None
        if request.tools:
            tools = []
            for t in request.tools:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        if tools:
            kwargs["tools"] = tools

        response = await client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = {}
                if tc.function.arguments:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {"raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        usage = Usage()
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return ModelResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            usage=usage,
            provider=self.name(),
            model=self._model,
            raw=response.model_dump() if hasattr(response, 'model_dump') else None,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncGenerator[ModelResponse, None]:
        client = self._get_client()

        messages = [m.to_dict() for m in request.messages]

        tools = None
        if request.tools:
            tools = []
            for t in request.tools:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": True,
        }
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        if tools:
            kwargs["tools"] = tools

        stream = await client.chat.completions.create(**kwargs)
        collected_content = ""
        collected_tool_calls: dict[str, ToolCall] = {}
        usage = Usage()

        async for chunk in stream:
            if chunk.usage:
                usage = Usage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                collected_content += delta.content
                yield ModelResponse(
                    content=delta.content,
                    provider=self.name(),
                    model=self._model,
                )

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    tc_id = tc.id or ""
                    if tc_id not in collected_tool_calls:
                        args = {}
                        if tc.function.arguments:
                            try:
                                args = json.loads(tc.function.arguments)
                            except json.JSONDecodeError:
                                args = {"raw": tc.function.arguments}
                        collected_tool_calls[tc_id] = ToolCall(
                            id=tc_id,
                            name=tc.function.name,
                            arguments=args,
                        )

        if collected_tool_calls:
            yield ModelResponse(
                content=collected_content,
                tool_calls=list(collected_tool_calls.values()),
                usage=usage,
                provider=self.name(),
                model=self._model,
            )

    def close(self) -> None:
        if self._client:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._client.close())
                else:
                    loop.run_until_complete(self._client.close())
            except Exception:
                pass


class AnthropicProvider(ModelProvider):
    """Anthropic Claude model provider."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022", base_url: str = "https://api.anthropic.com", timeout: float = 30.0):
        if not _has_anthropic:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout = timeout
        self._client = None

    def name(self) -> str:
        return "anthropic"

    def model(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is None:
            import anthropic as anthropic_mod
            self._client = anthropic_mod.AsyncAnthropic(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    def _convert_message(self, msg: Message) -> dict[str, Any]:
        d: dict[str, Any] = {"role": msg.role.value}
        if msg.content:
            d["content"] = msg.content
        if msg.tool_result:
            d["content"] = [{
                "type": "tool_result",
                "tool_use": {
                    "input": {},
                    "name": "",
                },
                "content": msg.tool_result.output,
            }]
        return d

    async def chat(self, request: ChatRequest) -> ModelResponse:
        client = self._get_client()

        messages = [self._convert_message(m) for m in request.messages]

        tools_list = []
        for t in request.tools:
            tools_list.append({
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            })

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if tools_list:
            kwargs["tools"] = tools_list

        response = await client.messages.create(**kwargs)

        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                args = block.input or {}
                tool_calls.append(ToolCall(
                    id=block.id or "",
                    name=block.name,
                    arguments=args,
                ))

        usage = Usage()
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            provider=self.name(),
            model=self._model,
        )

    def close(self) -> None:
        if self._client:
            self._client = None


class ModelManager:
    """Registry and facade for model providers."""

    def __init__(self, logger=None):
        self._providers: dict[str, ModelProvider] = {}
        self._active: Optional[str] = None
        self._logger = logger

    def register(self, name: str, provider: ModelProvider) -> None:
        self._providers[name] = provider
        if self._active is None:
            self._active = name
        if self._logger:
            self._logger.debug(f"registered model provider: {name}")

    def set_active(self, name: str) -> None:
        if name not in self._providers:
            raise ValueError(f"unknown provider: {name}")
        self._active = name
        if self._logger:
            self._logger.debug(f"active model provider set to: {name}")

    @property
    def active(self) -> Optional[ModelProvider]:
        if self._active:
            return self._providers.get(self._active)
        return None

    def get(self, name: str) -> Optional[ModelProvider]:
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    async def chat(self, request: ChatRequest) -> ModelResponse:
        provider = self.active
        if provider is None:
            raise RuntimeError("no active model provider")
        return await provider.chat(request)

    async def chat_stream(self, request: ChatRequest) -> AsyncGenerator[ModelResponse, None]:
        provider = self.active
        if provider is None:
            raise RuntimeError("no active model provider")
        async for chunk in provider.chat_stream(request):
            yield chunk

    def close(self) -> None:
        for name, provider in self._providers.items():
            provider.close()
            if self._logger:
                self._logger.debug(f"closed model provider: {name}")
