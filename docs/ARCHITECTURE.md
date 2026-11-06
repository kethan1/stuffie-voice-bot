# Lucy the Goose: Architecture

Lucy is a push-to-talk voice companion for elementary-school children. She runs
on a Raspberry Pi 4B (4 GB) inside a plush goose. A child squeezes Lucy's wing,
asks a question, lets go, and Lucy answers out loud in one or two short,
friendly sentences.

This document explains how the system is built and why. It covers the
technology choices, the software structure, the runtime behaviour, and how
latency, power, safety, and reliability were engineered.

---

## 1. Goals and constraints

| Goal | What it means in practice |
|---|---|
| **Low response latency** | A child should hear Lucy start talking within about 1–1.5 s of letting go of the wing, and get instant feedback (a chime) at 0 ms. Silence makes a toy feel broken. |
| **Low power** | It may run from a USB power bank. A Pi 4 cannot suspend, so every avoidable wake-up, radio, and busy loop is removed. |
| **Child safety and privacy** | Audio is recorded only while the wing is squeezed and never stored. Signs of distress get a fixed, caring answer without involving the LLM. Children's words stay out of the logs. |
| **Reliability** | No silent failure modes. Lucy says what is wrong ("can a grown-up check the Wi-Fi?"), restarts herself if wedged, and survives being unplugged. |
| **Maintainability** | A student team can read, test, and change it. The core logic is pure Python and is tested without hardware. |

Hard constraints: a Raspberry Pi 4B with 4 GB RAM, a 32 GB microSD card, a
microphone, a speaker, and the decision from the research assignment
(`instructions.pdf`): **GroqCloud (`whisper-large-v3-turbo` +
`openai/gpt-oss-20b`) with Piper text-to-speech on the Pi.** Piper turned out
to sound too robotic for a toy that is meant to feel alive, so Lucy now speaks
with a streamed cloud voice (Speechify) and keeps Piper on the Pi as the voice
that never fails (§3, §5.5).

---

## 2. System overview

```
 ┌──────────────────────── Lucy (plush goose) ─────────────────────────┐
 │                                                                      │
 │  wing button ──GPIO17──┐                ┌──I2S──► MAX98357A ─► speaker│
 │  eye LED ◄────GPIO27───┤  Raspberry Pi  │                             │
 │  USB microphone ──USB──┤  4B · 4 GB     ├──Wi-Fi──┐                   │
 │                        └───────┬────────┘         │                   │
 │      optional offline brain    │ localhost        │                   │
 │      faster-whisper tiny.en ◄──┤                  │                   │
 │      Ollama · gemma3:1b ◄──────┘                  │                   │
 └───────────────────────────────────────────────────┼───────────────────┘
                                                     │ HTTPS (HTTP/2, one connection per service)
                              ┌──────────────────────┴───────────────┐
                              ▼                                      ▼
            ┌────────────── GroqCloud ──────────────┐   ┌──────── Speechify ────────┐
            │ whisper-large-v3-turbo  (speech→text) │   │ simba-3.2 (Lucy's voice,  │
            │ openai/gpt-oss-20b      (reply)       │   │ streamed as raw PCM)      │
            │ openai/gpt-oss-120b     (fallback)    │   └───────────────────────────┘
            └───────────────────────────────────────┘
```

Speech recognition, the language model, and Lucy's voice run in the cloud,
where they are fast and natural. Lucy sends Speechify only the text of her own
replies, a sentence at a time, and plays its audio as it streams in. Piper
stays loaded **on the Pi** as her backup voice. It speaks whenever the cloud
voice cannot (no network, no free characters left, a failing service), so Lucy
can always speak, error messages included.

With `install.sh --local-backup`, Lucy also gets an **offline brain** on the Pi
itself: speech recognition with faster-whisper and replies from Gemma 3 1B served
by Ollama. It is slower, so it is used only when GroqCloud cannot be reached
(§5.4).

---

## 3. Technology decisions

| Area | Choice | Why (and what was rejected) |
|---|---|---|
| **OS** | Raspberry Pi OS **Lite**, 64-bit (Trixie or Bookworm) | First-class Pi kernel, firmware, Wi-Fi, I2S overlays, and libgpiod. Prebuilt aarch64 wheels exist for every dependency. *Lite* means no desktop, compositor, or display stack. DietPi saves little over a tuned Lite install and adds another layer. Buildroot or Yocto would boot faster and idle lower, but are far beyond what a student team can maintain. Ubuntu Server has weaker Pi-specific tooling. |
| **Speech-to-text** | Groq `whisper-large-v3-turbo` | Fast (hundreds of times real time), accurate on children's speech, and on the same API and connection as the LLM. Local Whisper on a Pi 4 would take seconds per utterance and eat the RAM Piper needs. |
| **LLM** | Groq `openai/gpt-oss-20b`, `reasoning_effort=low` | About 1,000 tokens/s on Groq, so a whole 40-word answer arrives in well under a second. Low reasoning effort keeps hidden "thinking" to a few dozen tokens. |
| **LLM fallback** | Groq `openai/gpt-oss-120b` | Free-tier limits are per model, so falling back on a 429 roughly doubles the usable daily quota. It accepts the same request parameters. (`llama-3.1-8b-instant` was retired from the free tier in Aug 2026.) |
| **Offline brain** (optional) | faster-whisper `tiny.en` (int8) + Ollama serving `gemma3:1b` | Used only when Groq is unreachable. Gemma 3 1B (~0.8 GB) is the local model the research chose, and it fits in 4 GB beside Piper. Ollama is what the Raspberry Pi Foundation tutorial uses, detects the Pi 4's CPU features at run time, and speaks the same OpenAI-compatible protocol, so no new chat code was needed. Local speech recognition is required too: without the cloud, Groq's Whisper is gone as well. faster-whisper installs from prebuilt wheels; whisper.cpp would have to be compiled on the Pi. |
| **Voice** (text-to-speech) | Speechify `simba-3.2`, voice `harper_32`, streamed as 22.05 kHz PCM | Natural: 11th of 103 voices in Artificial Analysis's blind TTS arena (Nov 2026). Built for streaming: about 60 ms from request to first audio on Speechify's side. Free plan: 500,000 characters a month (roughly 3,000 replies) with no credit card, and requests are refused rather than billed when it runs out. Rejected: **Kokoro** (82M, the best open model for its size) runs slower than real time even on an unoptimised Pi 5, and a Pi 4 is 2–3× slower. **Groq Orpheus** allows 100 requests a day. **ElevenLabs** and **Cartesia** offer about 20,000 free characters a month. **OpenAI** `gpt-4o-mini-tts` was slow to first audio in independent tests (p95 near 4 s). **Gemini TTS** terms forbid apps likely to be used by under-18s. **Google Chirp 3 HD** needs a credit card and ranks well below. **Azure**'s free tier excludes its HD voices. |
| **Backup voice** | Piper (`piper-tts` 1.x), voice `en_US-lessac-medium` | Runs faster than real time on a Pi 4 with no network, key, or quota. Always loaded, so it can take over mid-conversation. Lessac (trained on a professional narrator's recordings) replaced the original `amy` voice. |
| **Moderation** | On-device distress screen + system prompt + gpt-oss safety training | Groq retired Llama Guard 4 in March 2026. Its replacement (`gpt-oss-safeguard-20b`) is a preview model at 3 requests/minute on the free tier, so it cannot screen every turn. See §8. |
| **Audio I/O** | PortAudio via `sounddevice`, ALSA `plug` for rate conversion | Callback-driven: no polling, and gapless output. `plug` converts rates in C, so Lucy records at Whisper's native 16 kHz and plays at Piper's 22.05 kHz on any hardware. |
| **GPIO** | libgpiod v2 (`gpiod`) | The supported interface on current kernels (sysfs and RPi.GPIO are deprecated). Edges arrive through a file descriptor in the event loop, with kernel debouncing and kernel timestamps: no threads, no polling. |
| **Concurrency** | `asyncio` + a few dedicated threads | Network streaming, timers, and button events are all I/O-bound and event-driven. CPU-bound synthesis runs in one worker thread (onnxruntime releases the GIL). |
| **HTTP** | `httpx` with HTTP/2 (thin client, no vendor SDK) | One multiplexed TLS connection for every request, pre-warmed on button press. Avoiding the SDK avoids its pydantic import cost and keeps the client provider-neutral (Groq, OpenAI, Ollama all speak this protocol). |
| **Config** | One strict TOML file + systemd credentials | Typos fail at start-up. API keys are never in the file or the environment of other processes. |
| **Process supervision** | systemd (`Type=notify`, watchdog, sandbox) | Readiness, self-healing restarts, log capture, and a tight security sandbox, all built into the OS. |
| **Packaging** | uv, `uv.lock`, the `uv_build` backend | One universal, hash-locked lock for the laptop and the Pi. Installs in seconds, nothing compiles on the Pi, and tampered packages are refused. |

---

## 4. Software architecture

Lucy uses a **ports-and-adapters** (hexagonal) structure. The core is pure
application logic: it knows nothing about Groq, Speechify, Piper, PortAudio, or GPIO. It
talks only to small interfaces (`lucy.core.ports`). Adapters implement those
interfaces, and a single composition root wires them together.

```
                 ┌──────────────────────── lucy.core (pure) ─────────────────────────┐
                 │                                                                    │
  Button ───────►│  Controller ──────► TurnPipeline ──► Conversation, PhraseBook       │
  (port)         │  (state machine)    (stream STT→LLM→TTS→speaker)                    │
                 │       │                  │  uses: SentenceChunker, to_speakable,    │
                 │       │                  │        DistressScreen, TurnTimer          │
                 │       │                  │  thinks with: Brain (cloud) → Brain (local) │
                 │       │                  │  speaks with: FallbackVoice (cloud → Piper) │
                 │       ▼                  ▼                                           │
                 │  PowerControl,     SpeechToText, ChatModel, TextToSpeech,           │
                 │  Indicator,        Moderator, Speaker, Microphone      ◄── ports     │
                 │  Prewarmable                                                        │
                 └──────────────▲──────────────────────────▲──────────────────────────┘
                                │ implements               │ implements
   lucy.adapters ──────────────┴──────────────────────────┴─────────────────────────
     gpio.py        GpioButton, LedIndicator (libgpiod)
     sound.py       SoundDeviceMicrophone, SoundDeviceSpeaker (PortAudio)
     speechify_tts.py  SpeechifySpeech (Lucy's voice: streamed PCM over the HTTP client below)
     piper_tts.py   PiperSpeech (onnxruntime; the backup voice)
     cloud.py       WhisperTranscriber, StreamingChat, FallbackChat
     local_stt.py   FasterWhisperTranscriber (on-device speech for the offline brain)
     openai_api.py  ApiClient (httpx, HTTP/2, prewarm, error translation)
     power.py       PowerManager, CpuGovernor, WifiPowerSave
     console.py     KeyboardButton, LogIndicator (bring-up / development)

   lucy.app         composition root: config → adapters → core; lifecycle and shutdown
   lucy.brains      assembles the cloud brain (Groq) and the optional local brain (Ollama)
   lucy.voices      assembles Lucy's voice: Speechify, backed by Piper (or Piper alone)
   lucy.platform    systemd notify/watchdog, journald-aware logging
   lucy.tools       `lucy doctor`, `lucy bench`
   lucy.persona     Lucy's system prompt and fixed lines
```

**Dependency rule:** `lucy.core` imports nothing from `lucy.adapters`,
`lucy.app`, or any hardware or network library (numpy is the only third-party
import). Because of this, the whole conversation flow (barge-in,
timeouts, error phrases, power transitions) is unit-tested with in-memory fakes
(`tests/fakes.py`) on any laptop.

**Errors as a domain concept:** adapters translate library exceptions
(`httpx.ConnectError`, PortAudio errors, HTTP 429, …) into `lucy.core.errors`
types. The core maps those to what Lucy *says* (`pipeline.phrase_for`):

| Error | Lucy says |
|---|---|
| `ServiceUnreachable` (no Wi-Fi, DNS failure) | "I can't reach my thinking cloud right now. Can a grown-up check the Wi-Fi?" |
| `RateLimited` (HTTP 429) | "My goose brain needs a little rest. Let's talk again in a minute!" |
| `AuthenticationError`, `AudioDeviceError`, `ConfigError` | "Something inside me needs a grown-up's help." |
| `ContentRefused` (the reply screen or the cloud voice's content screen refused a sentence) | Lucy stops at once: "Hmm, that's a grown-up topic. Let's talk about something else, like animals or space!" |
| anything else | "My thoughts got all tangled up. Can you ask me again?" |

---

## 5. Runtime behaviour

### 5.1 The push-to-talk state machine (`core/controller.py`)

```
            press                     release (held ≥ 0.3 s)
    IDLE ──────────► LISTENING ─────────────────────────► RESPONDING ──┐
     ▲  ▲               │ release (tap < 0.3 s)               │ press   │ done
     │  │               ▼                                     ▼ (barge  │
     │  └──── "keep squeezing" hint / stop talking      LISTENING  in)  │
     └──────────────────────────────────────────────────────────────────┘
```

* **Press:** stop any speech and cancel any turn in flight (barge-in), switch to
  the ACTIVE power profile, pre-warm the HTTPS connection, open the
  microphone, light the LED, and play a short rising chime.
* **Release (held):** close the microphone, play a "got it" chime, and start a
  turn. Kernel timestamps mean the "held" time is measured at the button
  itself, not when Python got around to it.
* **Tap:** if Lucy was talking, a tap just stops her. If she was idle, she
  explains how to talk to her.
* **Stuck button:** recording stops by itself after `max_record_s`.
* **Hammered button:** an hourly turn budget protects the API quota.
* **Quiet:** after `awake_s` Lucy goes DORMANT (power saving). After
  `forget_after_s` the conversation is forgotten, so the next child starts
  fresh.

Every handler runs under one `asyncio.Lock`, so button edges, the recording
limit timer, and the dormant timer can never interleave half-way.

### 5.2 One turn, streamed end to end (`core/pipeline.py`)

```
 release
   │  trim silence (numpy)          ── nothing left? → "I didn't hear that" (no upload)
   ▼
 Whisper (Groq) ─────────────────── empty? → "I didn't hear that"
   ▼
 DistressScreen (on device, µs) ─── concern? → fixed "tell a grown-up" reply (LLM never sees it)
   ▼
 ┌─ think() ─────────────┐   ┌─ voice() ───────────────┐   ┌─ speak() ───────────────┐
 │ gpt-oss-20b SSE stream│──►│ to_speakable → Lucy's   │──►│ speaker queue, ≥ 0.3 s  │
 │ → SentenceChunker     │   │ voice, one sentence at  │   │ ahead → gapless audio   │
 │   (whole sentences)   │   │ a time, streamed parts  │   │                         │
 └───────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
         unbounded text queue          bounded (1) audio queue
```

The three stages run concurrently (`run_together`, a `TaskGroup` that
re-raises the original error). Each sentence goes to the voice as soon as it
is complete, and its audio starts playing as soon as its first ~100 ms has
arrived. The next sentence is synthesised while this one plays, and the LLM is
usually finished before the first sentence has finished playing. If the child
presses the wing again, the turn task is cancelled: the HTTP streams are
closed, queued audio is discarded within one audio block (~46 ms), and
whatever Lucy had already said is kept in the conversation, so a follow-up
"why?" still makes sense.

### 5.3 Threads

| Thread | Purpose | Wakes the CPU when idle? |
|---|---|---|
| asyncio event loop | everything else | Only for the watchdog ping every 15 s and timers |
| `tts` worker | Piper synthesis (onnxruntime, 3 intra-op threads), only when the backup voice speaks | No (spinning disabled) |
| PortAudio output callback | feed the speaker | Only while the speaker is open (a conversation is active) |
| PortAudio input callback | capture the mic | Only while the wing is squeezed |
| `local-stt` worker (optional) | faster-whisper, only when the cloud is unreachable | No |
| GPIO | none: edges arrive on a file descriptor in the event loop | No |

### 5.4 When the cloud is unreachable: the offline brain

A turn is answered by a `Brain`: a speech-to-text plus a chat model. The cloud
brain is Groq. The optional local brain is faster-whisper plus Ollama on the Pi.

```
 release ─► cloud known down? ──yes──► local brain ─► "let me think with my own goose brain"
               │ no
               ▼
           Groq Whisper ── connect fails ──► local brain (+ that line) ─► local Whisper
               ▼
           Groq gpt-oss ── connect fails before the first word ──► local brain (+ that line)
```

* **Only for unreachability.** Connection failures (no Wi-Fi, DNS failure,
  connect timeout) move a turn to the local brain. A rate limit still falls
  back to gpt-oss-120b, and a bad API key still asks for a grown-up. Those are
  problems another *cloud* model or a person must fix, and a slower local
  answer would hide them.
* **Outages are remembered.** The HTTP client marks Groq unreachable after a
  failed connection, and keeps it marked until any request gets an HTTP
  response back. Later turns go straight to the local brain instead of waiting
  on a connect timeout each time. While Groq is marked down, the press-time
  connection warm-up (§6.2) doubles as a recovery probe, so Lucy returns to the
  cloud on the first turn after the network comes back.
* **Never two brains in one answer.** The switch can only happen before the
  first word of a reply, exactly like the 20B → 120B fallback.
* **Expectations are set.** A 1B model on a Pi 4 takes several seconds, so Lucy
  first says "My cloud is far away right now, so let me think all by myself."
  The line is pre-rendered, queued without waiting, and covers most of the
  delay.
* **Smaller prompts.** The local model gets only the last 2 exchanges (the cloud
  gets 6), because reading a prompt is slow on a Pi 4. Ollama caches the system
  prompt, which is evaluated once at start-up (`preload`), so only the new
  question has to be read.
* **Resident models.** Both models are loaded at start-up, after Lucy is
  already awake, and stay in RAM (about 1.2 GB together). Idle RAM costs no
  power, while reading 0.8 GB from an SD card mid-conversation would cost a
  child tens of seconds.

### 5.5 Lucy's voice: the cloud first, Piper standing by (`core/speech.py`)

`FallbackVoice` decides, one sentence at a time, which voice speaks:

```
 sentence ─► cloud voice resting or known unreachable? ──yes──► Piper
               │ no
               ▼
           Speechify ── fails before any audio (no network, timeout, 5xx, 429, ──► Piper, and the
               │        no free characters left, bad key)                         cloud voice rests
               ▼                                                                  for rest_s (60 s)
           streamed parts ─► speaker
               └── fails after audio started ──► that sentence ends (never repeated in another voice)
               └── content refused ──► Lucy stops and changes the subject (Piper never says it)
```

* **The child never waits for a dead cloud twice.** A failure makes the cloud
  voice rest for a minute (longer if Speechify's `Retry-After` asks), and a
  failed connection marks it unreachable until the press-time warm-up (§6.2)
  gets an answer again. Meanwhile every sentence goes straight to Piper.
* **One voice per sentence.** Like the brains (§5.4), the switch happens only
  before a sentence's first audio.
* **Fixed lines follow the voice.** Lines rendered by Piper while the cloud
  voice was away are dropped from the cache when it comes back, so the next
  "I didn't hear that" is in Lucy's usual voice again.
* **Rate limits.** The free plan allows bursts of 10 requests, then one per
  second. Pre-rendering the fixed lines at start-up is the only burst. If it
  hits the limit, the request waits the second or so Speechify asks for, once,
  rather than switching voices.

Both voices produce 22.05 kHz audio (Speechify streams at Piper's rate), so one
speaker stream plays either, with no resampling on the Pi.

### 5.6 The honk (`core/honk.py`)

Read aloud, "Honk!" is a person saying a word. So `HonkingVoice`, wrapped
around whichever voice speaks, replaces each *interjected* honk ("Honk!",
"Honk honk!", "…, honk!") with a goose call. A honk used as a word ("geese honk
to stay together") is still spoken. The calls are synthesised with numpy at
start-up (~90 ms), like the earcons, so no audio files are shipped:

```
 breath "h" ─► buzzy "aw": pitch jumps up, then glides; period-doubling rasp ─► nasal "ng" ─► soft "k"
   70 ms              ~170 ms (about 20 harmonics shaped by moving formants)     70 ms        ~70 ms
```

Punctuation sets the mood. "Honk!" falls, "Honk?" rises like a question, and
"Honk." or "Honk," is lower and calmer. There are three variants of each,
picked at random, and calls are normalised to the loudness of Lucy's voice
(`tts.honk_volume` scales them). The system prompt keeps honks to about one
answer in three, so the call stays a treat rather than a tic.

---

## 6. Latency engineering

### 6.1 Where the time goes

The figures below are design targets and estimates for a Pi 4 on decent
Wi-Fi, measured from the moment the wing is released. **Measure the real
numbers on your device with `lucy bench`**. Every live turn also logs its
own breakdown (`stt=… first_token=… first_synth=… first_audio=…`).

| Stage | Estimate | What keeps it small |
|---|---|---|
| "Got it" chime | 0 ms | Pre-generated at start-up, queued immediately |
| Close mic, trim silence, encode WAV | ~20–50 ms | In-memory 16 kHz mono PCM; WAV has no encoder cost |
| Upload + Whisper | ~250–500 ms | 16 kHz mono (Whisper's native format, ~32 KB/s), silence trimmed, language pinned, connection already warm |
| Screen | < 1 ms | Local regular expressions |
| LLM time to first token | ~200–500 ms | gpt-oss-20b on Groq, `reasoning_effort=low`, reasoning not streamed, short prompt and history |
| First sentence complete | +~30–150 ms | ~1,000 tok/s; Lucy's answers are one or two short sentences (≤ 25 words) |
| Cloud voice, first ~100 ms of audio | ~100–250 ms | Streamed raw PCM; ~60 ms inside Speechify, plus the round trip; connection pre-warmed at press time |
| *(Piper instead, whole sentence)* | *~200–800 ms* | Warmed-up session, cores at full clock (ACTIVE governor). Only when the cloud voice cannot speak |
| Audio output buffer | ~80 ms | `output_latency_s` |
| **First word heard** | **≈ 0.7–1.5 s** | |

Offline turns are much slower. Rough estimates for a Pi 4 (measure with
`lucy bench --brain local`): local speech recognition 1–3 s, then the first
sentence from Gemma 3 1B after another 3–8 s (a few tokens per second). The "own
goose brain" line plays during that wait.

### 6.2 Techniques used

1. **Streaming everywhere.** LLM tokens become sentences, sentences become
   streamed audio, and audio plays while the rest of the sentence, and of the
   reply, is still being generated.
2. **Whole sentences, sent at once.** A voice gives a whole sentence its
   natural intonation, while a sentence synthesised in pieces sounds stitched
   together. The cloud voice streams fast enough that nothing is gained by
   splitting the first sentence. (`turn.first_clause_min_words` can still
   allow it, to start Piper sooner.)
3. **Pre-warming at press time.** DNS, TCP, and TLS (often 150–400 ms on Wi-Fi)
   happen while the child is still talking, for Groq and Speechify alike.
   HTTP/2 keeps STT and LLM on one warm connection, and each connection is
   kept alive for 120 s.
4. **Wi-Fi power save off while active.** With power save on, the access point
   holds packets until the radio's next wake-up, adding 100–300 ms of jitter
   to every round trip.
5. **CPU governor `performance` while active.** There is no frequency ramp-up
   delay when Piper starts.
6. **Pre-warmed and pre-rendered audio.** The Piper session runs one warm-up
   inference at start-up. Error and safety phrases are rendered once and
   replayed from RAM. Earcons are generated mathematically at start-up.
7. **Gapless playback.** The speaker always has at least 0.3 s of audio queued
   behind what is playing, and the callback pads silence rather than
   underrunning.
8. **No cloud moderation in the critical path.** See §8.

---

## 7. Power engineering

A Raspberry Pi 4 has no suspend state, so power is saved by removing work,
radios, and wake-ups. Lucy switches between three **power profiles**
(`adapters/power.py`):

| Profile | When | CPU governor | Wi-Fi power save | Speaker stream / amp |
|---|---|---|---|---|
| ACTIVE | wing pressed → reply finished | `performance` | off | open / on |
| STANDBY | first `awake_s` (45 s) after a turn | `ondemand` | off | open / on |
| DORMANT | after that, until the next press | `powersave` | **on** | **closed / off** |

Waking from DORMANT costs nothing noticeable. The governor switch is a sysfs
write (microseconds), Wi-Fi power save is switched off while the child is
still talking, and the speaker opens in parallel with recording.

Process-level savings:

* **Event-driven, never polling.** The button is a kernel file descriptor, the
  microphone stream exists only while the wing is squeezed, and the speaker
  stream is closed when dormant. An idle Lucy wakes only for the 15 s
  watchdog ping.
* **Speech is synthesised in the cloud.** While the cloud voice speaks, the
  Pi only moves PCM from the network to the sound card, instead of running
  Piper on three cores for every sentence.
* **onnxruntime spinning disabled** (`session.intra_op.allow_spinning=0`). By
  default its worker threads busy-wait after each inference, which keeps every
  core hot between sentences.
* **Amplifier gating.** With the `max98357a` overlay the kernel drives the amp's
  SD pin from the PCM state, so closing the speaker stream powers the amp down
  (and silences idle hiss inside the plush).

The optional offline brain is designed to cost RAM, not power, while idle. Both
models sit loaded in memory and do no inference until the cloud is unreachable.
(Check on your Pi with `top` that `ollama` stays near 0 % CPU between turns.)

System-level savings (`deploy/install.sh --tune`): headless OS Lite, onboard
audio, Bluetooth, and board LEDs off, no camera or display probing, unused
services disabled, and the journal kept in RAM (no SD-card write wake-ups).
`--read-only` adds an overlay root filesystem.

---

## 8. Safety and privacy

### 8.1 Layers of safety

1. **Push-to-talk only.** Lucy cannot hear anything unless the wing is
   squeezed. There is no wake word and no always-on microphone.
2. **On-device distress screen** (`core/safety.py`). Before anything reaches the
   LLM, the transcript is checked for signs that a child may be hurt, unsafe,
   or in distress ("my brother hits me", "I want to hurt myself", "nobody likes
   me", "he said it's our little secret", "someone on Roblox wants a picture of
   me", …). A match gets a fixed, caring script that points to a trusted
   grown-up such as a teacher or the school counselor (not "a parent", who is
   sometimes the one hurting the child), and those words are never sent to the
   language model. False positives are acceptable by design: the reply is kind
   either way. Softer signs (sadness, no friends) are left to the model, whose
   prompt also points to a trusted grown-up, so ordinary talk is not met with
   an urgent script.
3. **System prompt** (`lucy/persona.py`): short, simple, age-appropriate
   answers; hints rather than homework answers; never ask for personal
   details; steer away from grown-up topics; admit uncertainty; say she is a
   computer program if asked.
4. **Model safety training.** gpt-oss models are safety-tuned.
5. **On-device reply screen** (`BlockedWords` in `core/safety.py`). Every
   sentence of Lucy's reply is checked before any voice sees it, for
   profanity, slurs, sexual words, and drug names, including words spelled out
   letter by letter ("S H I T"). A blocked sentence is handled like a content
   refusal (below). This is the only check on her words when she speaks with
   Piper or thinks with the offline brain, neither of which has a screen of its
   own. Words with an innocent meaning ("weed", "kill") are left to the prompt.
6. **The voice's content screen.** Speechify screens every sentence before
   speaking it. If it refuses one, Lucy stops speaking at once and changes the
   subject. The refused sentence is never handed to Piper instead.
7. **Output hygiene.** Markdown, links, and emoji are stripped before speech.

*Why there is no cloud moderator:* Groq shut down `llama-guard-4-12b` on
2026-03-05. Its replacement, `openai/gpt-oss-safeguard-20b`, is a preview
model limited to 3 requests/minute (2K tokens/minute) on the free tier, so it
cannot sit in front of every turn. The `Moderator` port makes adding one a
small change if the team moves to a paid tier: implement `assess()` and wire
it in `app.py`.

### 8.2 What leaves the device

| Data | Where it goes | Retention |
|---|---|---|
| The recording (only while the wing is held, silence trimmed) | Groq Whisper | Not stored on the Pi. Groq: not retained by default, may be logged up to 30 days for abuse/reliability unless **Zero Data Retention** is enabled (console → Settings → Data Controls). |
| The transcript and the last ≤ 6 exchanges | Groq gpt-oss | Same as above. Held in Pi RAM only, forgotten after 5 minutes of quiet. |
| The text of Lucy's own replies and fixed lines (never the recording or the child's words as such, though a reply can echo them) | Speechify, to be spoken | Governed by Speechify's terms. The requests carry no child or device identifier. Without a Speechify key, nothing goes there. |
| Distress-screen matches | Nowhere | Answered locally. |
| Anything during an offline turn | Nowhere | Answered entirely on the Pi. Ollama is localhost-only, and `OLLAMA_NO_CLOUD=1` disables Ollama's own cloud features. |
| Logs | Pi RAM (volatile journal) | Contain timings and error types only, **no transcripts** unless `log_transcripts = true`. |

**Account ownership:** Groq's Services Agreement requires account holders to
be 18 or older, and makes the developer of an app likely to be used by minors
responsible for legal compliance. The Groq account and API key must therefore
belong to a teacher or parent, who should enable Zero Data Retention before
children use Lucy. The Speechify account and key should likewise belong to a
grown-up.

---

## 9. Reliability

| Failure | Behaviour |
|---|---|
| Wi-Fi down / DNS failure | With the offline brain, Lucy says she will think on her own and answers locally, more slowly. Without it, she says she can't reach her thinking cloud and asks for a grown-up. Her backup voice (Piper) is local, so she can always speak. |
| Cloud voice unavailable (outage, free characters used up, bad key, rate limit) | Piper speaks each sentence the cloud voice cannot, with no wait after the first failure (§5.5). The cloud voice is retried after `rest_s`, and the fixed lines return to it with it. |
| Cloud voice fails mid-sentence | That sentence stops where it is. It is never repeated in the other voice. |
| Network comes back | The next button press probes Groq while the child talks. The first successful answer switches turns back to the cloud. |
| Offline brain also unavailable (Ollama stopped, model missing) | Same as having no offline brain. `lucy doctor` checks both local models. |
| Groq rate limit on gpt-oss-20b | Transparent fallback to gpt-oss-120b (before the first word only, never mid-sentence). If both are limited, "I need a little rest". |
| Groq error mid-reply | Lucy stops after what she has already said (no stitched-together reply). |
| Whisper hears noise | Silence is trimmed locally and never uploaded. Segments with high `no_speech_prob` are dropped. |
| Speaker or mic unplugged | The error is reported. The LED blinks fast and Lucy asks for a grown-up if she still can. |
| Event loop wedged | The systemd watchdog (30 s) restarts the service. |
| Crash | `Restart=always`, 2 s back-off, rate-limited by `StartLimitBurst`. |
| Bad configuration | Refused at start-up with a precise message. Exit code 78 stops restart loops. If the voice loaded, Lucy says she needs a grown-up. |
| Power yanked | `--read-only` overlay root: nothing on the SD card is being written. |
| Missing API key | Lucy still starts, explains out loud that she needs a grown-up, and `lucy doctor` says exactly what to do. Without a Speechify key she simply speaks with her backup voice. |

---

## 10. Security

* **Least privilege.** The service runs as the system user `lucy` (groups `audio`,
  `gpio`) under a systemd sandbox: read-only filesystem, no home directories,
  private `/tmp`, device access limited to ALSA and GPIO, a restricted
  system-call set and address families, and no privilege escalation.
* **One capability, `CAP_NET_ADMIN`,** is granted ambiently so `iw` can toggle
  Wi-Fi power saving.
* **The offline brain** runs in its own service (`deploy/systemd/ollama.service`)
  as a systemd `DynamicUser` that owns only its model store. It has no
  capabilities, no devices, and a 2 GB memory cap, and under memory pressure the
  kernel prefers to kill it rather than Lucy (`OOMScoreAdjust`). It listens on
  127.0.0.1 only.
* **The API keys** live in `/etc/lucy/credentials/groq_api_key` and
  `speechify_api_key` (root, mode 600) and reach the service only through
  `LoadCredential=`. They are not in the config file, the environment, the
  logs, or the repository.
* **Supply chain:** dependencies are pinned with SHA-256 hashes in `uv.lock`,
  resolved with a two-week release-age cutoff, and installed with
  `uv sync --locked --no-build`: prebuilt wheels only, and a lock that is out of
  step with `pyproject.toml` is refused. `uv_build` runs inside uv, so no build
  tool is downloaded either. The installer fetches a pinned uv release, and it
  and the default Piper voice are checksum-verified. So are Ollama (a pinned
  release, hashed while it streams in, and installed only if the hash matches)
  and the faster-whisper model (a pinned Hugging Face commit plus a SHA-256 per
  file). Ollama verifies the Gemma model by digest when it pulls it.

---

## 11. Testing strategy

| Layer | How |
|---|---|
| Core logic (chunker, text cleaning, safety screen, conversation, phrases) | Plain unit tests |
| Turn pipeline and controller (barge-in, taps, record limit, dozing, budget, error phrases) | Fakes for every port (`tests/fakes.py`) |
| HTTP adapters (SSE parsing, error translation, fallback, prewarm) | `httpx.MockTransport` |
| Lucy's voice (PCM streaming in whole samples, content refusals, rate limits, falling back to Piper, resting, recovery) | `tests/test_voice.py`: `httpx.MockTransport` and fakes |
| The honk (which honks are interjections, pitch contour per mood, loudness, no clipping) | `tests/test_honk.py`, with an autocorrelation pitch tracker |
| Audio adapters (gapless splicing, stop, cancel, open failure, mic capping) | A simulated `sounddevice` whose thread calls the real callbacks |
| Config, power knobs, systemd notify | Temp files, a fake sysfs tree, a real Unix datagram socket |
| Offline brain (fallback at each stage, remembered outages, recovery, history limit, never mixing brains) | Fakes in `tests/test_pipeline.py`; a simulated faster-whisper in `tests/test_local.py` |
| Real voice end to end | `tests/test_integration.py` (runs when `LUCY_TEST_VOICE` is set) |
| Real offline turn: Piper speaks a question, faster-whisper hears it, a mocked Ollama answers | `tests/test_integration.py`, `tests/test_local.py` (with `LUCY_TEST_STT_MODEL`) |
| On a laptop | `make doctor`, `make say`, `make run`, `make bench` with `dev.toml`: the same code on the laptop's mic and speakers, Enter as the button, no GPIO or power knobs |
| On the device | `lucy doctor --listen`, `lucy say`, `lucy run --keyboard`, `lucy bench [--brain local]` |

The suite runs in about a second, with no network or hardware, under `ruff`
and `mypy --strict`.

---

## 12. Extending Lucy

* **Another provider:** set `api.base_url` and the model names. Any
  OpenAI-compatible endpoint works (OpenAI, a local Ollama server on another
  machine, …).
* **A different offline model:** `ollama pull` it and set `local.model`. For
  example, `gemma3:270m` is faster but simpler, and anything above about 2 GB
  will not fit beside Piper on a 4 GB Pi. A more accurate local speech model is
  `base.en` (point `local.stt_model` at it; roughly twice as slow).
* **A different voice:** `lucy voices` lists the cloud voices, with samples to
  listen to. Set `tts.cloud.voice_id` to one. For the backup voice, download
  any Piper voice and point `tts.voice` at it. `-low` voices are faster and
  `-high` voices are slower.
* **Another cloud voice service:** implement `TextToSpeech.stream()` (yield
  PCM parts at the backup voice's sample rate) and wire it in `lucy/voices.py`.
* **A cloud moderator:** implement `Moderator.assess()` (see §8.1).
* **A new personality:** set `system_prompt_file`, or edit `lucy/persona.py`.
