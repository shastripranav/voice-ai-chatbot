from collections.abc import AsyncIterator

from voice_chatbot.tts.base import BaseTTS
from voice_chatbot.utils import create_silent_wav


class MockTTS(BaseTTS):
    """Mock TTS that returns a silent WAV file proportional to text length."""

    async def synthesize(self, text: str) -> bytes:
        duration = max(0.5, len(text) * 0.04)
        return create_silent_wav(duration)

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        audio = await self.synthesize(text)
        chunk_size = 4096
        for i in range(0, len(audio), chunk_size):
            yield audio[i : i + chunk_size]
