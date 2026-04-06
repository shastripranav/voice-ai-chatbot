from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class BaseTTS(ABC):
    """Abstract interface for Text-to-Speech providers."""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes."""
        ...

    @abstractmethod
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio bytes as they're generated."""
        ...
        yield  # pragma: no cover
