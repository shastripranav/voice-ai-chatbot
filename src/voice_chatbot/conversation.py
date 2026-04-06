class ConversationMemory:
    """Manages conversation history with system prompt and turn-based trimming."""

    def __init__(self, system_prompt: str = "", max_turns: int = 20):
        self._messages: list[dict[str, str]] = []
        self._system_prompt = system_prompt
        self._max_turns = max_turns

    def add_user_message(self, content: str):
        self._messages.append({"role": "user", "content": content})
        self.trim_to_max_turns()

    def add_assistant_message(self, content: str):
        self._messages.append({"role": "assistant", "content": content})
        self.trim_to_max_turns()

    def get_messages(self) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if self._system_prompt:
            msgs.append({"role": "system", "content": self._system_prompt})
        msgs.extend(self._messages)
        return msgs

    def clear(self):
        self._messages.clear()

    def trim_to_max_turns(self):
        # trim from the front to keep recent context relevant
        max_msgs = self._max_turns * 2
        while len(self._messages) > max_msgs:
            self._messages.pop(0)

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self._messages if m["role"] == "user")

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str):
        self._system_prompt = value
