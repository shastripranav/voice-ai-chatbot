import logging
from collections.abc import AsyncIterator

from voice_chatbot.conversation import ConversationMemory
from voice_chatbot.llm.base import BaseLLM
from voice_chatbot.stt.base import BaseSTT
from voice_chatbot.tts.base import BaseTTS

log = logging.getLogger(__name__)


class VoicePipeline:
    """Orchestrates the full STT → LLM → TTS voice conversation pipeline."""

    def __init__(
        self,
        stt: BaseSTT,
        llm: BaseLLM,
        tts: BaseTTS,
        conversation: ConversationMemory,
    ):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.conversation = conversation

    async def process_voice(self, audio_bytes: bytes, audio_format: str = "wav") -> bytes:
        """Full pipeline: audio in → audio out."""
        text = await self.stt.transcribe(audio_bytes, audio_format)
        log.info("Transcribed: %s", text[:80])

        self.conversation.add_user_message(text)
        response = await self.llm.generate(self.conversation.get_messages())
        self.conversation.add_assistant_message(response)

        audio = await self.tts.synthesize(response)
        return audio

    async def process_text(self, text: str) -> str:
        """Text-only pipeline (skip STT/TTS)."""
        self.conversation.add_user_message(text)
        response = await self.llm.generate(self.conversation.get_messages())
        self.conversation.add_assistant_message(response)
        return response

    async def process_voice_to_text(self, audio_bytes: bytes, audio_format: str = "wav") -> str:
        """Voice in → text response (no TTS on the way out)."""
        text = await self.stt.transcribe(audio_bytes, audio_format)
        return await self.process_text(text)

    async def process_text_stream(self, text: str) -> AsyncIterator[str]:
        """Stream LLM response tokens for a text input."""
        self.conversation.add_user_message(text)
        full_response = []
        async for token in self.llm.generate_stream(self.conversation.get_messages()):
            full_response.append(token)
            yield token
        self.conversation.add_assistant_message("".join(full_response))

    def reset(self):
        self.conversation.clear()
