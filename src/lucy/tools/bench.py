"""``lucy bench``: measure the latency pipeline on the real device, without a microphone.

The question is synthesised with Lucy's on-device voice, resampled to 16 kHz,
and then pushed through the same STT → LLM → chunker → TTS path a real turn
uses, with the voice Lucy really speaks with (the cloud voice when it is set
up). ``first_audio`` is when the first part of her first sentence is ready to
play. The first round includes set-up (cold connections, or loading the local
models). Later rounds show the warm, steady-state latency a child will
experience.

``--brain local`` measures the offline backup (faster-whisper + Ollama) instead
of Groq.
"""

import asyncio
import statistics
import time
from contextlib import AsyncExitStack, aclosing

import numpy as np

from lucy.brains import build_cloud, build_local
from lucy.config import Config, read_system_prompt
from lucy.core.audio import Pcm
from lucy.core.chunker import SentenceChunker
from lucy.core.errors import ConfigError
from lucy.core.pipeline import Brain, TurnSettings
from lucy.core.ports import TextToSpeech
from lucy.core.speech import render
from lucy.core.textnorm import to_speakable
from lucy.core.types import Message, Role
from lucy.persona import LUCY
from lucy.voices import build_voice, load_on_device_voice

STAGES = ("stt", "first_token", "first_chunk", "first_audio", "reply_done")


def resample(audio: Pcm, rate: int) -> Pcm:
    samples = audio.samples().astype(np.float64) / 32768.0
    positions = np.arange(round(audio.duration * rate)) * audio.sample_rate / rate
    return Pcm.from_float(np.interp(positions, np.arange(samples.size), samples), rate)


async def bench(config: Config, question: str, rounds: int, *, brain_name: str = "cloud") -> int:
    system_prompt = read_system_prompt(config, LUCY.system_prompt)
    print("loading voice…")
    on_device = await asyncio.to_thread(load_on_device_voice, config)
    await asyncio.to_thread(on_device.warm_up)
    started = time.perf_counter()
    spoken = await render(on_device, question)
    synth_s = time.perf_counter() - started
    print(
        f"on-device TTS speed: {spoken.duration:.2f}s of audio in {synth_s:.2f}s "
        f"(real-time factor {synth_s / spoken.duration:.2f}; lower is better)"
    )
    speech = resample(spoken, config.audio.input_sample_rate)

    results: list[dict[str, float]] = []
    async with AsyncExitStack() as stack:
        voice = build_voice(config, on_device)
        stack.push_async_callback(voice.aclose)
        print(f"voice: {voice.name}")
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
            timings = await _one_round(brain, voice.tts, speech, system_prompt, config.turn)
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
    brain: Brain, tts: TextToSpeech, speech: Pcm, system_prompt: str, settings: TurnSettings
) -> dict[str, float]:
    timings: dict[str, float] = {}
    origin = time.perf_counter()

    def mark(stage: str) -> None:
        timings.setdefault(stage, time.perf_counter() - origin)

    async def voice(text: str) -> None:
        async with aclosing(tts.stream(to_speakable(text) or "Honk!")) as parts:
            async for _ in parts:
                mark("first_audio")

    text = await brain.stt.transcribe(speech)
    mark("stt")
    messages = [Message(Role.SYSTEM, system_prompt), Message(Role.USER, text)]
    chunker = SentenceChunker(
        first_clause_min_words=settings.first_clause_min_words,
        max_words=settings.max_chunk_words,
    )
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
