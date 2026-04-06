import httpx
import pytest
from httpx import ASGITransport

from api.main import app
from voice_chatbot.utils import create_silent_wav

_transport = ASGITransport(app=app)


@pytest.fixture
def api_client():
    return httpx.AsyncClient(transport=_transport, base_url="http://test")


async def test_health_endpoint(api_client):
    resp = await api_client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "stt_provider" in data


async def test_text_chat_creates_session(api_client):
    resp = await api_client.post(
        "/api/chat/text",
        json={"message": "Hello"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "session_id" in data
    assert len(data["session_id"]) > 0


async def test_text_chat_maintains_session(api_client):
    resp1 = await api_client.post(
        "/api/chat/text",
        json={"message": "First message"},
    )
    sid = resp1.json()["session_id"]

    resp2 = await api_client.post(
        "/api/chat/text",
        json={"message": "Second message", "session_id": sid},
    )
    assert resp2.json()["session_id"] == sid


async def test_voice_chat_endpoint(api_client):
    audio = create_silent_wav(0.5)
    resp = await api_client.post(
        "/api/chat/voice",
        files={"audio": ("test.wav", audio, "audio/wav")},
        data={"session_id": ""},
    )
    assert resp.status_code == 200
    assert resp.headers.get("content-type") == "audio/wav"
    assert resp.headers.get("x-session-id")
    assert len(resp.content) > 44


async def test_voices_endpoint(api_client):
    resp = await api_client.get("/api/voices")
    assert resp.status_code == 200
    voices = resp.json()
    assert isinstance(voices, list)
