from abc import ABC, abstractmethod


class BaseSTT(ABC):
    """Abstract interface for Speech-to-Text providers."""

    @abstractmethod
    async def transcribe(self, audio: bytes, format: str = "wav") -> str:
        """Convert audio bytes to text."""
        ...
