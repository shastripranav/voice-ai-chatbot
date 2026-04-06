from voice_chatbot.tts.mock import MockTTS


async def test_mock_tts_returns_wav():
    tts = MockTTS()
    result = await tts.synthesize("Hello world")
    assert isinstance(result, bytes)
    assert result[:4] == b"RIFF"
    assert len(result) > 44


async def test_mock_tts_duration_scales_with_text():
    tts = MockTTS()
    short = await tts.synthesize("Hi")
    long_text = "This is a much longer sentence that should produce more audio data"
    long = await tts.synthesize(long_text)

    assert len(long) > len(short)


async def test_mock_tts_stream():
    tts = MockTTS()
    chunks = []
    async for chunk in tts.synthesize_stream("Streaming test"):
        chunks.append(chunk)

    assert len(chunks) > 0
    full = b"".join(chunks)
    assert full[:4] == b"RIFF"


async def test_mock_tts_minimum_duration():
    tts = MockTTS()
    result = await tts.synthesize("")
    # even empty text should produce at least 0.5s of audio
    assert len(result) > 44
