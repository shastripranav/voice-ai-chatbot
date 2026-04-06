import logging
import uuid

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from voice_chatbot.config import VoiceChatConfig
from voice_chatbot.conversation import ConversationMemory
from voice_chatbot.llm.base import BaseLLM
from voice_chatbot.llm.mock import MockLLM
from voice_chatbot.pipeline import VoicePipeline
from voice_chatbot.stt.base import BaseSTT
from voice_chatbot.stt.mock import MockSTT
from voice_chatbot.tts.base import BaseTTS
from voice_chatbot.tts.mock import MockTTS

log = logging.getLogger(__name__)
router = APIRouter()

_config = VoiceChatConfig()

# TODO: implement proper session cleanup with TTL
_sessions: dict[str, VoicePipeline] = {}


def _build_stt(cfg: VoiceChatConfig) -> BaseSTT:
    if cfg.stt_provider == "mock":
        return MockSTT()
    from voice_chatbot.stt.whisper import WhisperSTT
    return WhisperSTT(api_key=cfg.openai_api_key)


def _build_llm(cfg: VoiceChatConfig) -> BaseLLM:
    if cfg.llm_provider == "mock":
        return MockLLM()
    if cfg.llm_provider == "openai":
        from voice_chatbot.llm.openai_llm import OpenAILLM
        return OpenAILLM(api_key=cfg.openai_api_key, model=cfg.openai_model)
    if cfg.llm_provider == "anthropic":
        from voice_chatbot.llm.anthropic_llm import AnthropicLLM
        return AnthropicLLM(api_key=cfg.anthropic_api_key, model=cfg.anthropic_model)
    if cfg.llm_provider == "ollama":
        from voice_chatbot.llm.ollama_llm import OllamaLLM
        return OllamaLLM(model=cfg.ollama_model, base_url=cfg.ollama_base_url)
    raise ValueError(f"Unknown LLM provider: {cfg.llm_provider}")


def _build_tts(cfg: VoiceChatConfig) -> BaseTTS:
    if cfg.tts_provider == "mock":
        return MockTTS()
    from voice_chatbot.tts.elevenlabs import ElevenLabsTTS
    return ElevenLabsTTS(
        api_key=cfg.elevenlabs_api_key,
        voice_id=cfg.elevenlabs_voice_id,
        model_id=cfg.elevenlabs_model_id,
    )


def _get_pipeline(session_id: str | None = None) -> tuple[str, VoicePipeline]:
    sid = session_id or str(uuid.uuid4())
    if sid not in _sessions:
        conversation = ConversationMemory(
            system_prompt=_config.system_prompt,
            max_turns=_config.max_conversation_turns,
        )
        pipeline = VoicePipeline(
            stt=_build_stt(_config),
            llm=_build_llm(_config),
            tts=_build_tts(_config),
            conversation=conversation,
        )
        _sessions[sid] = pipeline
    return sid, _sessions[sid]


class TextChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class TextChatResponse(BaseModel):
    response: str
    session_id: str


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "stt_provider": _config.stt_provider,
        "llm_provider": _config.llm_provider,
        "tts_provider": _config.tts_provider,
    }


@router.post("/chat/text", response_model=TextChatResponse)
async def chat_text(body: TextChatRequest):
    sid, pipeline = _get_pipeline(body.session_id)
    response = await pipeline.process_text(body.message)
    return TextChatResponse(response=response, session_id=sid)


@router.post("/chat/voice")
async def chat_voice(
    audio: UploadFile = File(...),
    session_id: str = Form(default=""),
):
    """Upload audio, get audio response. Full STT → LLM → TTS pipeline."""
    audio_bytes = await audio.read()
    fmt = audio.filename.rsplit(".", 1)[-1] if audio.filename and "." in audio.filename else "wav"

    sid, pipeline = _get_pipeline(session_id or None)
    response_audio = await pipeline.process_voice(audio_bytes, audio_format=fmt)

    from fastapi.responses import Response

    return Response(
        content=response_audio,
        media_type="audio/wav",
        headers={"X-Session-Id": sid},
    )


@router.get("/voices")
async def list_voices():
    """List available TTS voices (only works with ElevenLabs provider)."""
    if _config.tts_provider != "elevenlabs" or not _config.elevenlabs_api_key:
        return [{"voice_id": "mock", "name": "Mock Voice", "preview_url": None}]

    from voice_chatbot.tts.elevenlabs import ElevenLabsTTS

    tts = ElevenLabsTTS(api_key=_config.elevenlabs_api_key)
    try:
        return await tts.get_voices()
    except Exception:
        log.exception("Failed to fetch voices")
        return []
    finally:
        await tts.close()
