async def test_process_voice_end_to_end(pipeline, sample_audio):
    result = await pipeline.process_voice(sample_audio, audio_format="wav")

    assert isinstance(result, bytes)
    assert len(result) > 44  # must be bigger than just a WAV header
    assert result[:4] == b"RIFF"


async def test_process_text_end_to_end(pipeline):
    response = await pipeline.process_text("What's the weather like?")

    assert isinstance(response, str)
    assert len(response) > 0


async def test_pipeline_maintains_conversation(pipeline):
    await pipeline.process_text("Hello")
    await pipeline.process_text("How are you?")

    msgs = pipeline.conversation.get_messages()
    # system + 2 user + 2 assistant = 5
    assert len(msgs) == 5
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"


async def test_pipeline_reset(pipeline):
    await pipeline.process_text("Hi there")
    assert pipeline.conversation.turn_count == 1

    pipeline.reset()
    assert pipeline.conversation.turn_count == 0


async def test_process_text_stream(pipeline):
    tokens = []
    async for token in pipeline.process_text_stream("Tell me a joke"):
        tokens.append(token)

    assert len(tokens) > 0
    full = "".join(tokens)
    assert len(full.strip()) > 0
    # should have saved the response to conversation
    msgs = pipeline.conversation.get_messages()
    assert msgs[-1]["role"] == "assistant"


async def test_process_voice_to_text(pipeline, sample_audio):
    result = await pipeline.process_voice_to_text(sample_audio)

    assert isinstance(result, str)
    assert len(result) > 0
    assert pipeline.conversation.turn_count == 1
