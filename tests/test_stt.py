from voice_chatbot.stt.mock import MockSTT
from voice_chatbot.utils import create_silent_wav


async def test_mock_stt_returns_default():
    stt = MockSTT()
    result = await stt.transcribe(b"fake audio data")
    assert result == "Hello, this is a test transcription."


async def test_mock_stt_custom_text():
    stt = MockSTT(default_text="Custom transcription output")
    result = await stt.transcribe(b"any bytes")
    assert result == "Custom transcription output"


async def test_mock_stt_with_real_wav():
    stt = MockSTT()
    wav = create_silent_wav(0.5)
    result = await stt.transcribe(wav, format="wav")
    assert isinstance(result, str)
    assert len(result) > 0


# TODO: add integration tests for WhisperSTT with a test API key
