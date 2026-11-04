"""``lucy bench``: measure the latency pipeline on the real device, without a microphone.

The question is synthesised with Lucy's own voice, resampled to 16 kHz, and then
pushed through the same STT → LLM → chunker → TTS path a real turn uses. The
first round includes set-up (a cold connection, or loading the local models).
Later rounds show the warm, steady-state latency a child will experience.

``--brain local`` measures the offline backup (faster-whisper + Ollama) instead
of Groq.
"""

import asyncio
import statistics
import time
from contextlib import AsyncExitStack, aclosing

import numpy as np

from lucy.adapters.piper_tts import PiperSpeech
from lucy.app import load_voice
from lucy.brains import build_cloud, build_local
from lucy.config import Config, read_system_prompt
from lucy.core.audio import Pcm
from lucy.core.chunker import SentenceChunker
from lucy.core.errors import ConfigError
from lucy.core.pipeline import Brain
from lucy.core.textnorm import to_speakable
from lucy.core.types import Message, Role
from lucy.persona import LUCY

STAGES = ("stt", "first_token", "first_chunk", "first_audio", "reply_done")


def resample(audio: Pcm, rate: int) -> Pcm:
    samples = audio.samples().astype(np.float64) / 32768.0
    positions = np.arange(round(audio.duration * rate)) * audio.sample_rate / rate
    return Pcm.from_float(np.interp(positions, np.arange(samples.size), samples), rate)


async def bench(config: Config, question: str, rounds: int, *, brain_name: str = "cloud") -> int:
    system_prompt = read_system_prompt(config, LUCY.system_prompt)
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

    results: list[dict[str, float]] = []
    async with AsyncExitStack() as stack:
        stack.callback(tts.close)
        if brain_name == "local":
            local = build_local(config)
            if local is None:
                raise ConfigError("the local brain is not enabled ([local] enabled = true)")
            stack.push_async_callback(local.aclose)
            print("loading the local brain (speech model + Ollama model)…")
            started = time.perf_counter()
            await local.preload(system_prompt)
            print(f"local brain loaded in {time.perf_counter() - started:.1f}s")
            brain = local.brain()
        else:
            cloud = build_cloud(config)
            stack.push_async_callback(cloud.aclose)
            brain = cloud.brain()

        for index in range(1, rounds + 1):
            timings = await _one_round(brain, tts, speech, system_prompt)
            results.append(timings)
            label = "cold" if index == 1 else "warm"
            row = "  ".join(f"{s}={timings[s] * 1000:.0f}ms" for s in STAGES if s in timings)
            print(f"round {index} ({label}): {row}")

    warm = results[1:] or results
    print(
        f"\nmedian of warm rounds on the {brain_name} brain, measured from button release "
        "(excludes audio output latency):"
    )
    for stage in STAGES:
        values = [r[stage] for r in warm if stage in r]
        if values:
            print(f"  {stage:<12} {statistics.median(values) * 1000:7.0f} ms")
    return 0


async def _one_round(
    brain: Brain, tts: PiperSpeech, speech: Pcm, system_prompt: str
) -> dict[str, float]:
    timings: dict[str, float] = {}
    origin = time.perf_counter()

    def mark(stage: str) -> None:
        timings.setdefault(stage, time.perf_counter() - origin)

    async def voice(text: str) -> None:
        await tts.synthesize(to_speakable(text) or "Honk!")
        mark("first_audio")

    text = await brain.stt.transcribe(speech)
    mark("stt")
    messages = [Message(Role.SYSTEM, system_prompt), Message(Role.USER, text)]
    chunker = SentenceChunker()
    first_voice: asyncio.Task[None] | None = None
    async with aclosing(brain.chat.stream(messages)) as deltas:
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
