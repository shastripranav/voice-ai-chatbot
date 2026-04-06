import logging

import httpx

from voice_chatbot.stt.base import BaseSTT

log = logging.getLogger(__name__)


class WhisperSTT(BaseSTT):
    """OpenAI Whisper API implementation for speech-to-text."""

    def __init__(self, api_key: str, model: str = "whisper-1"):
        self._model = model
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

    async def transcribe(self, audio: bytes, format: str = "wav") -> str:
        mime = "audio/wav" if format == "wav" else f"audio/{format}"
        files = {"file": (f"audio.{format}", audio, mime)}
        data = {"model": self._model}

        try:
            resp = await self._client.post("/audio/transcriptions", files=files, data=data)
            resp.raise_for_status()
            return resp.json()["text"]
        except httpx.HTTPStatusError as exc:
            log.error("Whisper API error: %s", exc.response.text)
            raise RuntimeError(f"Whisper transcription failed: {exc.response.status_code}") from exc

    async def close(self):
        await self._client.aclose()
