from voice_chatbot.conversation import ConversationMemory


def test_add_messages():
    conv = ConversationMemory()
    conv.add_user_message("hello")
    conv.add_assistant_message("hi there")

    msgs = conv.get_messages()
    assert len(msgs) == 2
    assert msgs[0] == {"role": "user", "content": "hello"}
    assert msgs[1] == {"role": "assistant", "content": "hi there"}


def test_system_prompt_included():
    conv = ConversationMemory(system_prompt="Be helpful.")
    conv.add_user_message("test")

    msgs = conv.get_messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "Be helpful."


def test_trim_to_max_turns():
    conv = ConversationMemory(max_turns=2)

    for i in range(5):
        conv.add_user_message(f"user msg {i}")
        conv.add_assistant_message(f"assistant msg {i}")

    msgs = conv.get_messages()
    # max_turns=2 means 4 messages max (2 user + 2 assistant)
    assert len(msgs) == 4
    # should keep the most recent messages
    assert msgs[0]["content"] == "user msg 3"


def test_clear():
    conv = ConversationMemory(system_prompt="test")
    conv.add_user_message("a")
    conv.add_assistant_message("b")
    conv.clear()

    msgs = conv.get_messages()
    # only system prompt remains
    assert len(msgs) == 1
    assert msgs[0]["role"] == "system"


def test_turn_count():
    conv = ConversationMemory()
    assert conv.turn_count == 0

    conv.add_user_message("hi")
    assert conv.turn_count == 1

    conv.add_assistant_message("hello")
    assert conv.turn_count == 1

    conv.add_user_message("how are you")
    assert conv.turn_count == 2


def test_system_prompt_setter():
    conv = ConversationMemory(system_prompt="original")
    assert conv.system_prompt == "original"

    conv.system_prompt = "updated"
    msgs = conv.get_messages()
    assert msgs[0]["content"] == "updated"


def test_empty_system_prompt_excluded():
    conv = ConversationMemory(system_prompt="")
    conv.add_user_message("test")

    msgs = conv.get_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
