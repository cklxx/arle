"""Async LLM client for pegainfer and OpenAI-compatible endpoints.

Supports two modes:
1. Direct pegainfer: /v1/completions with raw prompt (uses chat.format_prompt)
2. OpenAI-compatible: /v1/chat/completions with structured messages (via dynamo frontend)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    base_url: str = "http://localhost:8000"
    mode: str = "completions"  # "completions" (pegainfer direct) or "chat" (dynamo/openai)
    timeout: float = 300.0
    model: str = "Qwen3-4B"


class LLMClient:
    """Async HTTP client for LLM inference."""

    def __init__(self, config: ClientConfig):
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout, connect=10.0),
        )

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: list[str] | None = None,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Generate completion from raw prompt (pegainfer /v1/completions)."""
        request = {
            "model": self.config.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": stream,
        }
        if stop:
            request["stop"] = stop

        if stream:
            async for chunk in self._stream_completions(request):
                yield chunk
        else:
            response = await self._client.post("/v1/completions", json=request)
            response.raise_for_status()
            data = response.json()
            yield data["choices"][0].get("text", "")

    async def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        tools: list[dict] | None = None,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Chat completion via OpenAI-compatible /v1/chat/completions (dynamo frontend)."""
        request = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if tools:
            request["tools"] = tools

        if stream:
            async for chunk in self._stream_chat(request):
                yield chunk
        else:
            response = await self._client.post("/v1/chat/completions", json=request)
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            yield choice.get("message", {}).get("content", "")

    async def _stream_completions(self, request: dict) -> AsyncIterator[str]:
        async with self._client.stream("POST", "/v1/completions", json=request) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    text = data["choices"][0].get("text", "")
                    if text:
                        yield text
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    async def _stream_chat(self, request: dict) -> AsyncIterator[str]:
        async with self._client.stream("POST", "/v1/chat/completions", json=request) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    async def close(self):
        await self._client.aclose()
