from voice_chatbot.llm.mock import MOCK_RESPONSES, MockLLM


async def test_mock_llm_returns_response():
    llm = MockLLM()
    result = await llm.generate([{"role": "user", "content": "Hello"}])
    assert isinstance(result, str)
    assert result in MOCK_RESPONSES


async def test_mock_llm_cycles_responses():
    llm = MockLLM()
    results = []
    for _ in range(len(MOCK_RESPONSES) + 1):
        resp = await llm.generate([{"role": "user", "content": "test"}])
        results.append(resp)

    # should cycle back to first response
    assert results[0] == results[len(MOCK_RESPONSES)]


async def test_mock_llm_custom_responses():
    custom = ["Response A", "Response B"]
    llm = MockLLM(responses=custom)

    r1 = await llm.generate([{"role": "user", "content": "x"}])
    r2 = await llm.generate([{"role": "user", "content": "y"}])

    assert r1 == "Response A"
    assert r2 == "Response B"


async def test_mock_llm_stream():
    llm = MockLLM()
    tokens = []
    async for token in llm.generate_stream([{"role": "user", "content": "Hi"}]):
        tokens.append(token)

    assert len(tokens) > 0
    full = "".join(tokens).strip()
    assert full in MOCK_RESPONSES
