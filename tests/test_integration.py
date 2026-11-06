"""Real Piper voice + real HTTP client (against a mocked Groq) through the real pipeline.

Needs a downloaded voice, so it only runs when ``LUCY_TEST_VOICE`` points at one::

    LUCY_TEST_VOICE=voices/en_US-lessac-medium.onnx pytest tests/test_integration.py
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
from lucy.core.pipeline import Brain, TurnPipeline, TurnSettings
from lucy.core.safety import DistressScreen
from lucy.core.speech import render
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
        brain=Brain(
            "cloud",
            WhisperTranscriber(api, model="whisper-large-v3-turbo", language="en"),
            StreamingChat(api, ChatModelSpec("openai/gpt-oss-20b")),
        ),
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
    speech = [clip for clip in speaker.played if any(clip.data)]  # without the pauses
    assert len(speech) == 3  # "Honk!" | the V sentence | "Isn't that neat?"
    assert all(clip.sample_rate == tts.sample_rate for clip in speaker.played)
    assert sum(clip.duration for clip in speech) > 2.0
    first_audio = timer.get("first_audio")
    assert first_audio is not None
    assert first_audio < 2.0


STT_MODEL = os.environ.get("LUCY_TEST_STT_MODEL")


@pytest.mark.skipif(not STT_MODEL, reason="set LUCY_TEST_STT_MODEL to a faster-whisper model")
async def test_offline_turn_uses_the_local_brain() -> None:
    from lucy.adapters.local_stt import FasterWhisperTranscriber
    from lucy.tools.bench import resample

    assert VOICE is not None
    assert STT_MODEL is not None

    def no_network(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Temporary failure in name resolution", request=request)

    asked: list[str] = []

    def ollama(request: httpx.Request) -> httpx.Response:
        asked.append(json.loads(request.read())["messages"][-1]["content"])
        events = [{"choices": [{"delta": {"content": p}}]} for p in ["Honk! ", "They like warmth."]]
        body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx.Response(200, content=body.encode())

    tts = PiperSpeech(Path(VOICE), threads=2)
    cloud_api = ApiClient(
        "https://api.test/openai/v1", "sk", transport=httpx.MockTransport(no_network)
    )
    local_api = ApiClient(
        "http://127.0.0.1:11434/v1", "ollama", transport=httpx.MockTransport(ollama)
    )
    local_stt = FasterWhisperTranscriber(Path(STT_MODEL), threads=2)
    speaker = FakeSpeaker()
    pipeline = TurnPipeline(
        brain=Brain(
            "cloud",
            WhisperTranscriber(cloud_api, model="whisper-large-v3-turbo", language="en"),
            StreamingChat(cloud_api, ChatModelSpec("openai/gpt-oss-20b")),
            reachable=cloud_api.reachable,
        ),
        backup=Brain("local", local_stt, StreamingChat(local_api, ChatModelSpec("gemma3:1b"))),
        tts=tts,
        moderator=DistressScreen(),
        speaker=speaker,
        indicator=FakeIndicator(),
        phrases=PhraseBook(LUCY.lines, tts),
        conversation=Conversation(2),
        system_prompt=LUCY.system_prompt,
        settings=TurnSettings(),
    )
    question = resample(await render(tts, "Why do geese fly south in the winter?"), 16_000)
    try:
        await pipeline.respond(question, TurnTimer())
        assert not cloud_api.reachable()  # remembered: the next turn skips the cloud
        await pipeline.respond(question, TurnTimer())
    finally:
        await cloud_api.aclose()
        await local_api.aclose()
        local_stt.close()
        tts.close()
    assert len(asked) == 2
    assert "geese" in asked[0].lower()
    # Each turn: "thinking with my own brain", then "Honk!" and "They like warmth."
    assert len([clip for clip in speaker.played if any(clip.data)]) == 6
