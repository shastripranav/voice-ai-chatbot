import asyncio
from collections.abc import AsyncIterator

from voice_chatbot.llm.base import BaseLLM

MOCK_RESPONSES = [
    "I'm doing well, thanks for asking! How can I help you today?",
    "That's a great question. Let me think about that for a moment.",
    "I understand what you're looking for. Here's what I'd suggest.",
    "Interesting! Could you tell me more about what you have in mind?",
    "Sure, I can help with that. Let me walk you through it.",
]


class MockLLM(BaseLLM):
    """Mock LLM that cycles through canned responses. No API key needed."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or MOCK_RESPONSES
        self._call_count = 0

    async def generate(self, messages: list[dict]) -> str:
        resp = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return resp

    async def generate_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        response = await self.generate(messages)
        for word in response.split():
            yield word + " "
            await asyncio.sleep(0.02)
