import os
import sys
from pathlib import Path

import pytest

# make src/ importable without editable install
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))

# default to mock providers so API tests work without keys
os.environ.setdefault("STT_PROVIDER", "mock")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("TTS_PROVIDER", "mock")

from voice_chatbot.config import VoiceChatConfig  # noqa: E402
from voice_chatbot.conversation import ConversationMemory  # noqa: E402
from voice_chatbot.llm.mock import MockLLM  # noqa: E402
from voice_chatbot.pipeline import VoicePipeline  # noqa: E402
from voice_chatbot.stt.mock import MockSTT  # noqa: E402
from voice_chatbot.tts.mock import MockTTS  # noqa: E402
from voice_chatbot.utils import create_silent_wav  # noqa: E402


@pytest.fixture
def mock_stt():
    return MockSTT()


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def mock_tts():
    return MockTTS()


@pytest.fixture
def conversation():
    return ConversationMemory(system_prompt="You are a test assistant.", max_turns=5)


@pytest.fixture
def pipeline(mock_stt, mock_llm, mock_tts, conversation):
    return VoicePipeline(stt=mock_stt, llm=mock_llm, tts=mock_tts, conversation=conversation)


@pytest.fixture
def sample_audio():
    return create_silent_wav(duration_seconds=0.5)


@pytest.fixture
def mock_config(monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "mock")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("TTS_PROVIDER", "mock")
    return VoiceChatConfig()
