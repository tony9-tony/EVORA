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


try:
    import httpx
    _has_httpx = True
except ImportError:
    _has_httpx = False


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
        payload = self._build_payload(request, stream=True)
        collected_content = ""
        collected_tool_calls: dict[str, ToolCall] = {}
        usage = Usage()

        async def _stream_once():
            nonlocal collected_content, collected_tool_calls, usage
            async with client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    raise ConnectionError(
                        f"Ollama stream request failed (HTTP {resp.status_code})"
                    )
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choice = chunk.get("choices", [{}])[0]
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        collected_content += delta["content"]
                        yield ModelResponse(
                            content=delta["content"],
                            provider=self.name(),
                            model=self._model,
                        )
                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            tc_id = tc.get("id") or ""
                            if tc_id not in collected_tool_calls:
                                args = {}
                                raw_args = tc.get("function", {}).get("arguments", "")
                                if raw_args:
                                    try:
                                        args = json.loads(raw_args)
                                    except json.JSONDecodeError:
                                        args = {"raw": raw_args}
                                collected_tool_calls[tc_id] = ToolCall(
                                    id=tc_id,
                                    name=tc.get("function", {}).get("name", ""),
                                    arguments=args,
                                )
                    if chunk.get("usage"):
                        usage = Usage(
                            prompt_tokens=chunk["usage"].get("prompt_tokens", 0),
                            completion_tokens=chunk["usage"].get("completion_tokens", 0),
                            total_tokens=chunk["usage"].get("total_tokens", 0),
                        )

        import asyncio as _asyncio
        for attempt in range(self._max_retries + 1):
            try:
                async for item in _stream_once():
                    yield item
                if collected_tool_calls:
                    yield ModelResponse(
                        content=collected_content,
                        tool_calls=list(collected_tool_calls.values()),
                        usage=usage,
                        provider=self.name(),
                        model=self._model,
                    )
                return
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt < self._max_retries:
                    await _asyncio.sleep(1.0 * (attempt + 1))
                    continue
                raise self._wrap_connection_error(e)
            except httpx.HTTPStatusError as e:
                raise ConnectionError(
                    f"Ollama stream request failed (HTTP {e.response.status_code})"
                ) from e

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


class OllamaProvider(ModelProvider):
    """Ollama local model provider (OpenAI-compatible /v1 API via httpx)."""

    DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
    DEFAULT_MODEL = "qwen2.5-coder:latest"
    DEFAULT_TIMEOUT = 180.0

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        api_key: str = "ollama",
        max_retries: int = 2,
    ):
        if not _has_httpx:
            raise ImportError("httpx package not installed. Run: pip install httpx")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._api_key = api_key
        self._max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    def name(self) -> str:
        return "ollama"

    def model(self) -> str:
        return self._model

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        return self._client

    def switch_model(self, model: str) -> None:
        """Switch the model used by this provider without recreating the client."""
        self._model = model

    def _get_available_models(self) -> list[str]:
        """Fetch the list of locally available Ollama models."""
        import asyncio
        try:
            client = httpx.Client(base_url=self._base_url, timeout=10.0)
            resp = client.get("/models")
            client.close()
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("model", m.get("id", "")) for m in data.get("data", data.get("models", []))]
        except Exception:
            pass
        return []

    def _build_payload(self, request: ChatRequest, stream: bool = False) -> dict[str, Any]:
        messages = [m.to_dict() for m in request.messages]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        if request.tools:
            kwargs["tools"] = [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            } for t in request.tools]
        if stream:
            kwargs["stream"] = True
        return kwargs

    def _wrap_connection_error(self, original: Exception) -> ConnectionError:
        if isinstance(original, httpx.TimeoutException):
            return ConnectionError(
                f"Ollama at {self._base_url} timed out after {self._timeout}s. "
                f"The model '{self._model}' may be slow to respond on this hardware. "
                f"Try increasing the timeout in your EVORA config. "
                f"Original error: {original}"
            )
        return ConnectionError(
            f"Cannot connect to Ollama at {self._base_url}. "
            f"Ensure Ollama is running ('ollama serve') and the model "
            f"'{self._model}' is available "
            f"(run 'ollama pull {self._model}'). "
            f"Original error: {original}"
        )

    async def chat(self, request: ChatRequest) -> ModelResponse:
        client = self._get_client()
        payload = self._build_payload(request, stream=False)
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = await client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                break
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < self._max_retries:
                    import asyncio
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                raise self._wrap_connection_error(e)
            except httpx.HTTPStatusError as e:
                body = ""
                try:
                    body = (await e.response.aread()).decode()[:300]
                except Exception:
                    body = str(e)
                raise ConnectionError(
                    f"Ollama request failed (HTTP {e.response.status_code}): {body}"
                ) from e

        data = resp.json()
        choice = data["choices"][0]
        msg = choice.get("message") or choice.get("delta") or {}
        content = msg.get("content") or ""

        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            args = {}
            raw_args = tc.get("function", {}).get("arguments", "")
            if raw_args:
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {"raw": raw_args}
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=tc.get("function", {}).get("name", ""),
                arguments=args,
            ))

        usage = Usage()
        u = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
        )

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            provider=self.name(),
            model=self._model,
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncGenerator[ModelResponse, None]:
        client = self._get_client()
        payload = self._build_payload(request, stream=True)
        collected_content = ""
        collected_tool_calls: dict[str, ToolCall] = {}
        usage = Usage()

        async def _stream_once():
            nonlocal collected_content, collected_tool_calls, usage
            async with client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    raise ConnectionError(
                        f"Ollama stream request failed (HTTP {resp.status_code})"
                    )
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choice = chunk.get("choices", [{}])[0]
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        collected_content += delta["content"]
                        yield ModelResponse(
                            content=delta["content"],
                            provider=self.name(),
                            model=self._model,
                        )
                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            tc_id = tc.get("id") or ""
                            if tc_id not in collected_tool_calls:
                                args = {}
                                raw_args = tc.get("function", {}).get("arguments", "")
                                if raw_args:
                                    try:
                                        args = json.loads(raw_args)
                                    except json.JSONDecodeError:
                                        args = {"raw": raw_args}
                                collected_tool_calls[tc_id] = ToolCall(
                                    id=tc_id,
                                    name=tc.get("function", {}).get("name", ""),
                                    arguments=args,
                                )
                    if chunk.get("usage"):
                        usage = Usage(
                            prompt_tokens=chunk["usage"].get("prompt_tokens", 0),
                            completion_tokens=chunk["usage"].get("completion_tokens", 0),
                            total_tokens=chunk["usage"].get("total_tokens", 0),
                        )

        import asyncio as _asyncio
        for attempt in range(self._max_retries + 1):
            try:
                async for item in _stream_once():
                    yield item
                if collected_tool_calls:
                    yield ModelResponse(
                        content=collected_content,
                        tool_calls=list(collected_tool_calls.values()),
                        usage=usage,
                        provider=self.name(),
                        model=self._model,
                    )
                return
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt < self._max_retries:
                    await _asyncio.sleep(1.0 * (attempt + 1))
                    continue
                raise self._wrap_connection_error(e)
            except httpx.HTTPStatusError as e:
                raise ConnectionError(
                    f"Ollama stream request failed (HTTP {e.response.status_code})"
                ) from e

    def close(self) -> None:
        if self._client is not None:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._client.aclose())
                else:
                    loop.run_until_complete(self._client.aclose())
            except Exception:
                pass
            finally:
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
