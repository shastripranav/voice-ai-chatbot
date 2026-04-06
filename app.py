"""Streamlit demo UI for the Voice AI Chatbot pipeline."""

import io
import base64

import httpx
import streamlit as st

API_BASE = "http://localhost:8000/api"

st.set_page_config(page_title="Voice AI Chatbot", page_icon="🎙️", layout="wide")


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = ""


def get_provider_config() -> dict:
    """Sidebar controls for provider selection."""
    with st.sidebar:
        st.header("Provider Settings")

        stt = st.selectbox("STT Provider", ["mock", "whisper"], index=0)
        llm = st.selectbox("LLM Provider", ["mock", "openai", "anthropic", "ollama"], index=0)
        tts = st.selectbox("TTS Provider", ["mock", "elevenlabs"], index=0)

        st.divider()

        st.subheader("API Keys")
        openai_key = st.text_input("OpenAI API Key", type="password")
        anthropic_key = st.text_input("Anthropic API Key", type="password")
        elevenlabs_key = st.text_input("ElevenLabs API Key", type="password")

        st.divider()

        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = ""
            st.rerun()

        return {
            "stt": stt,
            "llm": llm,
            "tts": tts,
            "openai_key": openai_key,
            "anthropic_key": anthropic_key,
            "elevenlabs_key": elevenlabs_key,
        }


def render_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("audio"):
                st.audio(msg["audio"], format="audio/wav")


def send_text_message(text: str):
    """Send a text message through the API."""
    st.session_state.messages.append({"role": "user", "content": text})

    try:
        resp = httpx.post(
            f"{API_BASE}/chat/text",
            json={
                "message": text,
                "session_id": st.session_state.session_id or None,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        st.session_state.session_id = data["session_id"]
        st.session_state.messages.append({"role": "assistant", "content": data["response"]})
    except httpx.HTTPError as exc:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Error: Could not reach the API. {exc}",
        })


def send_voice_message(audio_bytes: bytes):
    """Send an audio message through the voice API."""
    st.session_state.messages.append({"role": "user", "content": "🎤 [Voice message]"})

    try:
        resp = httpx.post(
            f"{API_BASE}/chat/voice",
            files={"audio": ("recording.wav", audio_bytes, "audio/wav")},
            data={"session_id": st.session_state.session_id or ""},
            timeout=60.0,
        )
        resp.raise_for_status()

        sid = resp.headers.get("x-session-id", "")
        if sid:
            st.session_state.session_id = sid

        audio_response = resp.content
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🔊 [Voice response]",
            "audio": audio_response,
        })
    except httpx.HTTPError as exc:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Error: Voice processing failed. {exc}",
        })


def main():
    init_session_state()
    cfg = get_provider_config()

    st.title("Voice AI Chatbot")
    st.caption("Speak or type to chat with the AI assistant")

    render_chat_history()

    col1, col2 = st.columns([3, 1])

    with col1:
        text_input = st.chat_input("Type a message...")
        if text_input:
            send_text_message(text_input)
            st.rerun()

    with col2:
        audio_data = st.audio_input("Record audio", key="audio_recorder")
        if audio_data is not None:
            audio_bytes = audio_data.read()
            if audio_bytes:
                send_voice_message(audio_bytes)
                st.rerun()

    with st.sidebar:
        st.divider()
        st.caption(f"Session: {st.session_state.session_id[:8]}..." if st.session_state.session_id else "No active session")

        try:
            health = httpx.get(f"{API_BASE}/health", timeout=3.0)
            if health.status_code == 200:
                info = health.json()
                st.success(f"API Online — STT: {info['stt_provider']}, LLM: {info['llm_provider']}, TTS: {info['tts_provider']}")
            else:
                st.warning("API returned unexpected status")
        except httpx.HTTPError:
            st.error("API Offline — start the server with `uvicorn api.main:app`")


if __name__ == "__main__":
    main()
