import logging
from collections.abc import AsyncIterator

import httpx

from voice_chatbot.tts.base import BaseTTS

log = logging.getLogger(__name__)


class ElevenLabsTTS(BaseTTS):
    """ElevenLabs text-to-speech provider."""

    def __init__(
        self,
        api_key: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_monolingual_v1",
    ):
        self._voice_id = voice_id
        self._model_id = model_id
        self._client = httpx.AsyncClient(
            base_url="https://api.elevenlabs.io/v1",
            headers={"xi-api-key": api_key},
            timeout=30.0,
        )

    async def synthesize(self, text: str) -> bytes:
        payload = {
            "text": text,
            "model_id": self._model_id,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        try:
            resp = await self._client.post(
                f"/text-to-speech/{self._voice_id}",
                json=payload,
                headers={"Accept": "audio/mpeg"},
            )
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPStatusError as exc:
            log.error("ElevenLabs API error: %s", exc.response.text)
            raise RuntimeError(f"ElevenLabs synthesis failed: {exc.response.status_code}") from exc

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        payload = {
            "text": text,
            "model_id": self._model_id,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        async with self._client.stream(
            "POST",
            f"/text-to-speech/{self._voice_id}/stream",
            json=payload,
            headers={"Accept": "audio/mpeg"},
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes(chunk_size=4096):
                yield chunk

    async def get_voices(self) -> list[dict]:
        """Fetch available voices from ElevenLabs."""
        resp = await self._client.get("/voices")
        resp.raise_for_status()
        voices = resp.json().get("voices", [])
        return [
            {
                "voice_id": v["voice_id"],
                "name": v["name"],
                "preview_url": v.get("preview_url"),
            }
            for v in voices
        ]

    async def close(self):
        await self._client.aclose()
