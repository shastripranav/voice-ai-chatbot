import json
import logging
from collections.abc import AsyncIterator

import httpx

from voice_chatbot.llm.base import BaseLLM

log = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI GPT chat completion provider."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self._model = model
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    async def generate(self, messages: list[dict]) -> str:
        payload = {"model": self._model, "messages": messages}
        try:
            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            log.error("OpenAI API error: %s", exc.response.text)
            raise RuntimeError(f"OpenAI generation failed: {exc.response.status_code}") from exc

    async def generate_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        payload = {"model": self._model, "messages": messages, "stream": True}
        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0].get("delta", {})
                if content := delta.get("content"):
                    yield content

    async def close(self):
        await self._client.aclose()
