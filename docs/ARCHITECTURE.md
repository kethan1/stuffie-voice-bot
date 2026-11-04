# Lucy the Goose: Architecture

Lucy is a push-to-talk voice companion for elementary-school children. She runs
on a Raspberry Pi 4B (4 GB) inside a plush goose. A child squeezes Lucy's wing,
asks a question, lets go, and Lucy answers out loud in one to three short,
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
`openai/gpt-oss-20b`) with Piper text-to-speech on the Pi.**

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
                                                     │ HTTPS (HTTP/2, one connection)
                                                     ▼
                                   ┌────────────── GroqCloud ──────────────┐
                                   │ whisper-large-v3-turbo  (speech→text) │
                                   │ openai/gpt-oss-20b      (reply)       │
                                   │ openai/gpt-oss-120b     (fallback)    │
                                   └───────────────────────────────────────┘
```

Speech recognition and the language model run in the cloud, where they are fast.
Text-to-speech runs **on the Pi**. That keeps Lucy able to speak (including
error messages) when the network is down, avoids a second round trip per
sentence, and costs nothing.

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
| **Text-to-speech** | Piper (`piper-tts` 1.x), voice `en_US-amy-medium` | Runs in real time on a Pi 4 with no network. Groq's hosted TTS (Orpheus) is a preview model with a small daily request limit. |
| **Moderation** | On-device distress screen + system prompt + gpt-oss safety training | Groq retired Llama Guard 4 in March 2026. Its replacement (`gpt-oss-safeguard-20b`) is a preview model at 3 requests/minute on the free tier, so it cannot screen every turn. See §8. |
| **Audio I/O** | PortAudio via `sounddevice`, ALSA `plug` for rate conversion | Callback-driven: no polling, and gapless output. `plug` converts rates in C, so Lucy records at Whisper's native 16 kHz and plays at Piper's 22.05 kHz on any hardware. |
| **GPIO** | libgpiod v2 (`gpiod`) | The supported interface on current kernels (sysfs and RPi.GPIO are deprecated). Edges arrive through a file descriptor in the event loop, with kernel debouncing and kernel timestamps: no threads, no polling. |
| **Concurrency** | `asyncio` + a few dedicated threads | Network streaming, timers, and button events are all I/O-bound and event-driven. CPU-bound synthesis runs in one worker thread (onnxruntime releases the GIL). |
| **HTTP** | `httpx` with HTTP/2 (thin client, no vendor SDK) | One multiplexed TLS connection for every request, pre-warmed on button press. Avoiding the SDK avoids its pydantic import cost and keeps the client provider-neutral (Groq, OpenAI, Ollama all speak this protocol). |
| **Config** | One strict TOML file + systemd credentials | Typos fail at start-up. The API key is never in the file or the environment of other processes. |
| **Process supervision** | systemd (`Type=notify`, watchdog, sandbox) | Readiness, self-healing restarts, log capture, and a tight security sandbox, all built into the OS. |
| **Packaging** | uv, `uv.lock`, the `uv_build` backend | One universal, hash-locked lock for the laptop and the Pi. Installs in seconds, nothing compiles on the Pi, and tampered packages are refused. |

---

## 4. Software architecture

Lucy uses a **ports-and-adapters** (hexagonal) structure. The core is pure
application logic: it knows nothing about Groq, Piper, PortAudio, or GPIO. It
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
                 │       ▼                  ▼                                           │
                 │  PowerControl,     SpeechToText, ChatModel, TextToSpeech,           │
                 │  Indicator,        Moderator, Speaker, Microphone      ◄── ports     │
                 │  Prewarmable                                                        │
                 └──────────────▲──────────────────────────▲──────────────────────────┘
                                │ implements               │ implements
   lucy.adapters ──────────────┴──────────────────────────┴─────────────────────────
     gpio.py        GpioButton, LedIndicator (libgpiod)
     sound.py       SoundDeviceMicrophone, SoundDeviceSpeaker (PortAudio)
     piper_tts.py   PiperSpeech (onnxruntime)
     cloud.py       WhisperTranscriber, StreamingChat, FallbackChat
     local_stt.py   FasterWhisperTranscriber (on-device speech for the offline brain)
     openai_api.py  ApiClient (httpx, HTTP/2, prewarm, error translation)
     power.py       PowerManager, CpuGovernor, WifiPowerSave
     console.py     KeyboardButton, LogIndicator (bring-up / development)

   lucy.app         composition root: config → adapters → core; lifecycle and shutdown
   lucy.brains      assembles the cloud brain (Groq) and the optional local brain (Ollama)
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
 │ gpt-oss-20b SSE stream│──►│ to_speakable → Piper    │──►│ speaker queue, one clip │
 │ → SentenceChunker     │   │ (one worker thread)     │   │ ahead → gapless audio   │
 └───────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
         unbounded text queue          bounded (1) audio queue
```

The three stages run concurrently (`run_together`, a `TaskGroup` that
re-raises the original error). The first clause is spoken while the next is
being synthesised, and the LLM is usually finished before the first clause
has finished playing. If the child presses the wing again, the turn task is
cancelled: the HTTP stream is closed, queued audio is discarded within one
audio block (~46 ms), and whatever Lucy had already said is kept in the
conversation, so a follow-up "why?" still makes sense.

### 5.3 Threads

| Thread | Purpose | Wakes the CPU when idle? |
|---|---|---|
| asyncio event loop | everything else | Only for the watchdog ping every 15 s and timers |
| `tts` worker | Piper synthesis (onnxruntime, 3 intra-op threads) | No (spinning disabled) |
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
| First clause complete | +~20–50 ms | ~1,000 tok/s; the chunker may cut the *first* chunk at a comma |
| Piper, first clause | ~150–600 ms | Short first chunk, warmed-up session, cores at full clock (ACTIVE governor) |
| Audio output buffer | ~80 ms | `output_latency_s` |
| **First word heard** | **≈ 0.8–1.6 s** | |

Offline turns are much slower. Rough estimates for a Pi 4 (measure with
`lucy bench --brain local`): local speech recognition 1–3 s, then the first
sentence from Gemma 3 1B after another 3–8 s (a few tokens per second). The "own
goose brain" line plays during that wait.

### 6.2 Techniques used

1. **Streaming everywhere.** LLM tokens become clauses, clauses become audio,
   and audio plays while the rest is still being generated.
2. **Eager first chunk.** The first chunk may end at a clause boundary once it
   has four or more words. Later chunks end at sentences for natural prosody.
3. **Pre-warming at press time.** DNS, TCP, and TLS (often 150–400 ms on Wi-Fi)
   happen while the child is still talking. HTTP/2 keeps STT and LLM on one
   warm connection, kept alive for 120 s.
4. **Wi-Fi power save off while active.** With power save on, the access point
   holds packets until the radio's next wake-up, adding 100–300 ms of jitter
   to every round trip.
5. **CPU governor `performance` while active.** There is no frequency ramp-up
   delay when Piper starts.
6. **Pre-warmed and pre-rendered audio.** The Piper session runs one warm-up
   inference at start-up. Error and safety phrases are rendered once and
   replayed from RAM. Earcons are generated mathematically at start-up.
7. **Gapless playback.** The speaker always has the next clip queued, and the
   callback pads silence rather than underrunning.
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
   me", …). A match gets a fixed, caring script that always points to a trusted
   grown-up, and those words are never sent to the language model. False
   positives are acceptable by design: the reply is kind either way.
3. **System prompt** (`lucy/persona.py`): short, simple, age-appropriate
   answers; hints rather than homework answers; never ask for personal
   details; steer away from grown-up topics; admit uncertainty; say she is a
   computer program if asked.
4. **Model safety training.** gpt-oss models are safety-tuned.
5. **Output hygiene.** Markdown, links, and emoji are stripped before speech.

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
| Distress-screen matches | Nowhere | Answered locally. |
| Anything during an offline turn | Nowhere | Answered entirely on the Pi. Ollama is localhost-only, and `OLLAMA_NO_CLOUD=1` disables Ollama's own cloud features. |
| Logs | Pi RAM (volatile journal) | Contain timings and error types only, **no transcripts** unless `log_transcripts = true`. |

**Account ownership:** Groq's Services Agreement requires account holders to
be 18 or older, and makes the developer of an app likely to be used by minors
responsible for legal compliance. The Groq account and API key must therefore
belong to a teacher or parent, who should enable Zero Data Retention before
children use Lucy.

---

## 9. Reliability

| Failure | Behaviour |
|---|---|
| Wi-Fi down / DNS failure | With the offline brain, Lucy says she will think on her own and answers locally, more slowly. Without it, she says she can't reach her thinking cloud and asks for a grown-up. Piper is local, so she can always speak. |
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
| Missing API key | Lucy still starts, explains out loud that she needs a grown-up, and `lucy doctor` says exactly what to do. |

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
* **The API key** lives in `/etc/lucy/credentials/groq_api_key` (root, mode 600)
  and reaches the service only through `LoadCredential=`. It is not in the
  config file, the environment, the logs, or the repository.
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
* **A different voice:** download any Piper voice and point `tts.voice` at it.
  `-low` voices are faster and `-high` voices are slower.
* **A cloud moderator:** implement `Moderator.assess()` (see §8.1).
* **A new personality:** set `system_prompt_file`, or edit `lucy/persona.py`.
