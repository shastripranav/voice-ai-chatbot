from voice_chatbot.stt.base import BaseSTT


class MockSTT(BaseSTT):
    """Mock STT provider for testing and demos without API keys."""

    def __init__(self, default_text: str = "Hello, this is a test transcription."):
        self._default_text = default_text

    async def transcribe(self, audio: bytes, format: str = "wav") -> str:
        return self._default_text
