"""``lucy bench``: measure the latency pipeline on the real device, without a microphone.

The question is synthesised with Lucy's own voice, resampled to 16 kHz, and then
pushed through the same STT → LLM → chunker → TTS path a real turn uses. The
first round includes connection set-up (a cold start). Later rounds show the
warm, steady-state latency a child will experience.
"""

import asyncio
import statistics
import time
from contextlib import aclosing

import numpy as np

from lucy.adapters.piper_tts import PiperSpeech
from lucy.app import Cloud, build_cloud, load_voice
from lucy.config import Config
from lucy.core.audio import Pcm
from lucy.core.chunker import SentenceChunker
from lucy.core.textnorm import to_speakable
from lucy.core.types import Message, Role
from lucy.persona import LUCY

STAGES = ("stt", "first_token", "first_chunk", "first_audio", "reply_done")


def resample(audio: Pcm, rate: int) -> Pcm:
    samples = audio.samples().astype(np.float64) / 32768.0
    positions = np.arange(round(audio.duration * rate)) * audio.sample_rate / rate
    return Pcm.from_float(np.interp(positions, np.arange(samples.size), samples), rate)


async def bench(config: Config, question: str, rounds: int) -> int:
    print("loading voice…")
    tts = await asyncio.to_thread(load_voice, config)
    await asyncio.to_thread(tts.warm_up)
    started = time.perf_counter()
    spoken = await tts.synthesize(question)
    synth_s = time.perf_counter() - started
    print(
        f"TTS speed: {spoken.duration:.2f}s of audio in {synth_s:.2f}s "
        f"(real-time factor {synth_s / spoken.duration:.2f}; lower is better)"
    )
    speech = resample(spoken, config.audio.input_sample_rate)

    cloud = build_cloud(config)
    results: list[dict[str, float]] = []
    try:
        for index in range(1, rounds + 1):
            timings = await _one_round(cloud, tts, speech)
            results.append(timings)
            label = "cold" if index == 1 else "warm"
            row = "  ".join(f"{s}={timings[s] * 1000:.0f}ms" for s in STAGES if s in timings)
            print(f"round {index} ({label}): {row}")
    finally:
        await cloud.api.aclose()
        tts.close()

    warm = results[1:] or results
    print("\nmedian of warm rounds, measured from button release (excludes audio output latency):")
    for stage in STAGES:
        values = [r[stage] for r in warm if stage in r]
        if values:
            print(f"  {stage:<12} {statistics.median(values) * 1000:7.0f} ms")
    return 0


async def _one_round(cloud: Cloud, tts: PiperSpeech, speech: Pcm) -> dict[str, float]:
    timings: dict[str, float] = {}
    origin = time.perf_counter()

    def mark(stage: str) -> None:
        timings.setdefault(stage, time.perf_counter() - origin)

    async def voice(text: str) -> None:
        await tts.synthesize(to_speakable(text) or "Honk!")
        mark("first_audio")

    text = await cloud.stt.transcribe(speech)
    mark("stt")
    messages = [Message(Role.SYSTEM, LUCY.system_prompt), Message(Role.USER, text)]
    chunker = SentenceChunker()
    first_voice: asyncio.Task[None] | None = None
    async with aclosing(cloud.chat.stream(messages)) as deltas:
        async for delta in deltas:
            mark("first_token")
            for chunk in chunker.feed(delta):
                if first_voice is None:
                    mark("first_chunk")
                    first_voice = asyncio.create_task(voice(chunk.text))
    for chunk in chunker.flush():
        if first_voice is None:
            mark("first_chunk")
            first_voice = asyncio.create_task(voice(chunk.text))
    mark("reply_done")
    if first_voice is not None:
        await first_voice
    return timings
