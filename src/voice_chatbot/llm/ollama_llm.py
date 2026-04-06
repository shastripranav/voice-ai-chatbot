import json
from collections.abc import AsyncIterator

import httpx

from voice_chatbot.llm.base import BaseLLM


class OllamaLLM(BaseLLM):
    """Local Ollama LLM provider — no cloud API key required."""

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434"):
        self._model = model
        self._client = httpx.AsyncClient(base_url=base_url, timeout=120.0)

    async def generate(self, messages: list[dict]) -> str:
        payload = {"model": self._model, "messages": messages, "stream": False}
        resp = await self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    async def generate_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        payload = {"model": self._model, "messages": messages, "stream": True}
        # TODO: add retry logic for transient network failures
        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if content := chunk.get("message", {}).get("content"):
                    yield content

    async def close(self):
        await self._client.aclose()
