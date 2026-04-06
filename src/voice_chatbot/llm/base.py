from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class BaseLLM(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def generate(self, messages: list[dict]) -> str:
        """Generate a response from a list of conversation messages."""
        ...

    @abstractmethod
    async def generate_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream response tokens from a list of conversation messages."""
        ...
        # needed to make this a valid async generator
        yield  # pragma: no cover
