from pydantic_settings import BaseSettings


class VoiceChatConfig(BaseSettings):
    """Central configuration for the voice chatbot pipeline."""

    stt_provider: str = "whisper"
    llm_provider: str = "openai"
    tts_provider: str = "elevenlabs"

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    elevenlabs_api_key: str = ""

    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-haiku-20240307"
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"

    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel
    elevenlabs_model_id: str = "eleven_monolingual_v1"

    system_prompt: str = (
        "You are a helpful AI assistant. Keep responses concise and conversational."
    )
    max_conversation_turns: int = 20

    model_config = {"env_file": ".env", "extra": "ignore"}
