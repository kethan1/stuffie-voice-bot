"""Short-term conversation memory.

Lucy remembers the last few exchanges so follow-ups like "why?" make sense,
and forgets everything after a quiet period (see the controller). History is
held only in RAM and never written to disk.
"""

from collections import deque

from lucy.core.types import Message, Role


class Conversation:
    def __init__(self, max_exchanges: int) -> None:
        self._exchanges: deque[tuple[str, str]] = deque(maxlen=max(0, max_exchanges))

    def __len__(self) -> int:
        return len(self._exchanges)

    def prompt(self, system_prompt: str, user_text: str) -> list[Message]:
        messages = [Message(Role.SYSTEM, system_prompt)]
        for asked, answered in self._exchanges:
            messages.append(Message(Role.USER, asked))
            messages.append(Message(Role.ASSISTANT, answered))
        messages.append(Message(Role.USER, user_text))
        return messages

    def record(self, user_text: str, reply: str) -> None:
        """Store a completed (or interrupted but partly spoken) exchange."""
        if user_text and reply:
            self._exchanges.append((user_text, reply))

    def clear(self) -> None:
        self._exchanges.clear()
