import json
import logging
from collections.abc import AsyncIterator

import httpx

from voice_chatbot.llm.base import BaseLLM

log = logging.getLogger(__name__)


class AnthropicLLM(BaseLLM):
    """Anthropic Claude messages API provider."""

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self._model = model
        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com/v1",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    def _split_messages(self, messages: list[dict]) -> tuple[str, list[dict]]:
        # Anthropic separates system prompt from messages, unlike OpenAI
        system = ""
        chat_msgs = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_msgs.append(msg)
        return system, chat_msgs

    async def generate(self, messages: list[dict]) -> str:
        system, chat_msgs = self._split_messages(messages)
        payload: dict = {
            "model": self._model,
            "max_tokens": 1024,
            "messages": chat_msgs,
        }
        if system:
            payload["system"] = system

        try:
            resp = await self._client.post("/messages", json=payload)
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]
        except httpx.HTTPStatusError as exc:
            log.error("Anthropic API error: %s", exc.response.text)
            raise RuntimeError(
                f"Anthropic generation failed: {exc.response.status_code}"
            ) from exc

    async def generate_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        system, chat_msgs = self._split_messages(messages)
        payload: dict = {
            "model": self._model,
            "max_tokens": 1024,
            "messages": chat_msgs,
            "stream": True,
        }
        if system:
            payload["system"] = system

        async with self._client.stream("POST", "/messages", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("type") == "content_block_delta":
                    if text := event.get("delta", {}).get("text"):
                        yield text

    async def close(self):
        await self._client.aclose()
