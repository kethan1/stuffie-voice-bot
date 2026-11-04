"""Real Piper voice + real HTTP client (against a mocked Groq) through the real pipeline.

Needs a downloaded voice, so it only runs when ``LUCY_TEST_VOICE`` points at one::

    LUCY_TEST_VOICE=voices/en_US-amy-medium.onnx pytest tests/test_integration.py
"""

import json
import os
from pathlib import Path

import httpx
import pytest

from lucy.adapters.cloud import ChatModelSpec, StreamingChat, WhisperTranscriber
from lucy.adapters.openai_api import ApiClient
from lucy.adapters.piper_tts import PiperSpeech
from lucy.core.conversation import Conversation
from lucy.core.metrics import TurnTimer
from lucy.core.phrases import PhraseBook
from lucy.core.pipeline import TurnPipeline, TurnSettings
from lucy.core.safety import DistressScreen
from lucy.persona import LUCY
from tests.fakes import FakeIndicator, FakeSpeaker, tone

VOICE = os.environ.get("LUCY_TEST_VOICE")
pytestmark = pytest.mark.skipif(not VOICE, reason="set LUCY_TEST_VOICE to a Piper .onnx voice")

REPLY = ["Honk! Geese fly in a V", " because it saves energy.", " Isn't that neat?"]


def groq(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/audio/transcriptions"):
        return httpx.Response(200, json={"text": "Why do geese fly in a V?", "segments": []})
    events = [{"choices": [{"delta": {"content": piece}}]} for piece in REPLY]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode())


async def test_full_turn_with_real_voice() -> None:
    assert VOICE is not None
    tts = PiperSpeech(Path(VOICE), threads=2)
    api = ApiClient("https://api.test/openai/v1", "sk", transport=httpx.MockTransport(groq))
    speaker = FakeSpeaker()
    pipeline = TurnPipeline(
        stt=WhisperTranscriber(api, model="whisper-large-v3-turbo", language="en"),
        chat=StreamingChat(api, ChatModelSpec("openai/gpt-oss-20b")),
        tts=tts,
        moderator=DistressScreen(),
        speaker=speaker,
        indicator=FakeIndicator(),
        phrases=PhraseBook(LUCY.lines, tts),
        conversation=Conversation(2),
        system_prompt=LUCY.system_prompt,
        settings=TurnSettings(),
    )
    timer = TurnTimer()
    try:
        await pipeline.respond(tone(1.0), timer)
    finally:
        await api.aclose()
        tts.close()
    assert len(speaker.played) == 3  # "Honk!" | the V sentence | "Isn't that neat?"
    assert all(clip.sample_rate == tts.sample_rate for clip in speaker.played)
    assert sum(clip.duration for clip in speaker.played) > 2.0
    first_audio = timer.get("first_audio")
    assert first_audio is not None
    assert first_audio < 2.0
