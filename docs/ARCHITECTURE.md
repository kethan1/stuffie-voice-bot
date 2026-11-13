# Lucy the Goose: Architecture

Lucy is a push-to-talk voice companion for elementary-school children. She runs
on a Raspberry Pi 4B (4 GB) inside a plush goose. A child squeezes Lucy's wing,
asks a question, lets go, and Lucy answers out loud in one or two short,
friendly sentences. She speaks with a goose's voice and honks a lot, her beak
moves in time with her voice, her body flaps whenever she honks, and she gets
louder or quieter when a child asks.

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
 │  USB microphone ──USB──┤  4B · 4 GB     ├──PWM+GPIO─► L293D ───┬► beak│
 │                        │                │ motor battery ─►VCC2 └► body│
 │                        │                ├──Wi-Fi──┐                   │
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
| **Motors** | L293D dual H-bridge; the Pi's hardware PWM (`pwm-2chan` overlay, sysfs) at 20 kHz on its enable pins | Two small DC motors (beak, body) on their own battery. Hardware PWM uses no CPU, thread, or timer, and at 20 kHz it is above hearing, so the motors do not whine. Each speed change is one sysfs write of a few microseconds. Rejected: software PWM (`lgpio`, `pigpio`) needs a thread or daemon that wakes the CPU thousands of times a second; a servo HAT adds an I2C chip and needs a different mechanism; on/off drive cannot open the beak part-way. One PWM channel on the enable pin plus two direction lines per motor covers both directions, and lets a stopped motor coast instead of braking (§5.7). |
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
                 │       │                  │        SafetyScreen, ReplyFilter, TurnTimer│
                 │       │                  │  thinks with: Brain (cloud) → Brain (local) │
                 │       │                  │  speaks with: FallbackVoice (cloud → Piper) │
                 │       │                  │  moves with:  Puppeteer (beak, body)        │
                 │       ▼                  ▼                                           │
                 │  PowerControl,     SpeechToText, ChatModel, TextToSpeech,           │
                 │  Indicator,        Moderator, Speaker, Microphone,     ◄── ports     │
                 │  Prewarmable       Motors, PlaybackListener                         │
                 └──────────────▲──────────────────────────▲──────────────────────────┘
                                │ implements               │ implements
   lucy.adapters ──────────────┴──────────────────────────┴─────────────────────────
     gpio.py        GpioButton, LedIndicator, GpioOutput (libgpiod)
     motors.py      L293d, HBridge, SysfsPwm (Lucy's motors: hardware PWM + GPIO)
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
   lucy.motors      assembles her motors and the Puppeteer (or none: she keeps still)
   lucy.platform    systemd notify/watchdog, journald-aware logging
   lucy.tools       `lucy doctor`, `lucy bench`
   lucy.setup       setup mode: her own hotspot, the settings page, NetworkManager (§5.8)
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
| `ContentRefused` (the reply screen or the cloud voice's content screen refused a sentence) | Lucy stops at once and says the safety line that fits (§8.1), for example "Nope, that one's off my goose menu. A grown-up you trust is the one to ask." |
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
  explains how to talk to her, after `tap_gap_s` (0.7 s), in case the tap
  turns out to start a knock.
* **Knock:** two taps (or more), each within `tap_gap_s` of the last, and then
  the squeeze, within `tap_gap_s` again. The turn that follows is marked as
  knocked, and may ask for setup mode (§5.8). Nothing else about it differs.
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
 SafetyScreen (on device, µs) ───── concern? → a fixed reply to match (LLM never sees it)
   ▼
 asks_for_setup (on device, µs) ─── knocked, and "enter setup mode"? → say so, hand over (§5.8)
   ▼
 volume_request (on device, µs) ─── "louder!"? → change the volume, say so at the new level
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
| Motors | none: the PWM runs in hardware, and each move is an event-loop timer (§5.7) | No (timers exist only while she speaks) |

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

**The goose voice** (`core/goose_voice.py`). Whichever voice speaks, its speech
goes through `GooseVoice` on the way out, scaled by `tts.goose_voice` (0 to 1,
default 0.7). It gives a human voice the three things that make a goose sound
like a goose, taken from the same measurements of real geese as her honk (§5.6):

* **Smaller.** Resampling raises her pitch and her resonances together by up
  to 25% (17.5% at the default, nearly three semitones): the sound of a smaller
  creature. That speeds speech up by as much, so each voice is asked to speak
  that much more slowly first and comes out at its usual pace. Speechify takes
  that as SSML (`<prosody rate="-15%">`; the tags are not billed, and the text
  is escaped). Piper takes a `length_scale`, which slows it by less than the
  number says (the short gaps it puts between sounds do not stretch), so
  `piper_tts.py` holds the measured relation and asks for what is meant.
* **Honky.** A linear-phase FIR filter (255 taps) gives her voice a goose's
  resonances: the windpipe's ring at 1.35 kHz (up to +12 dB), the dip near
  2.9 kHz, and the brassy second peak near 4 kHz, and takes out the chest
  tones below 300 Hz. It also removes anything the resampler would fold back
  down (aliasing).
* **Raspy.** A goose's voice starts as a growl and breaks upward into the honk,
  and drops back into period doubling in between. `Rasp` does that to her
  voice: every other cycle comes out weaker, which is heard as a rasp an octave
  below her pitch. It finds the cycles the way a guitar's octave pedal does (a
  low-pass filter leaves only the fundamental, and a trigger with hysteresis
  counts its upward zero crossings), so the rasp follows her pitch instead of
  buzzing over it. It is deepest for the first 150 ms after a silence, as a
  honk starts with its growl, and sways lightly the rest of the time. Silence
  gets none.

The filter is normalised so speech comes out at the level it went in: her beak
(§5.7) opens as wide as before, and the volume settings mean what they did. To
the ear she is 2 to 4 dB louder, because the sound has moved to where ears are
keenest. Peaks are rounded off (`soft_limit`), not clipped.

Each stage carries its state from one streamed part to the next (the filters'
histories, the resampler's fractional position, the trigger and its switch, the
running loudness), so parts join without a seam: streaming in any part sizes
gives the same samples as one block. It costs about 0.4 ms per 100 ms part on
a laptop. The goose voice sits inside `HonkingVoice`, so honks, which are goose
calls already, pass through untouched. A filter rather than a "goose" voice
from the catalogue keeps her words as clear as the natural voice they come
from, and works the same for Piper.

How clear: thirty of Lucy's kind of sentences (330 words), spoken by Piper and
transcribed by two speech recognisers on a laptop. Whisper `small.en` got 11
words wrong in her plain voice, 10 at the default 0.7, and 13 at 1.0. The much
weaker `tiny.en` got 14, 20, and 33, which is why the default is 0.7.

**Her volume** (`core/volume.py`). A child can say "louder", "speak up", "I
can't hear you", "too loud", "use your indoor voice", and so on. After the
safety screen, `volume_request` matches short requests aimed at Lucy, with no
model involved. Questions ("which is louder, a lion or a whale?") and other
subjects ("can a whale be louder than a jet?") are not requests. The change is
a master volume on the speaker (`Speaker.set_volume`): 3 dB a step, from −15 dB
to +9 dB by default (`[volume]`), applied to everything she plays, chimes
included. Lucy has only a few decibels of room below full scale, so above 0 dB
her peaks are rounded off (`soft_limit`, a tanh knee) rather than clipped. The
motors see each clip as it was made, so her beak opens as wide at any volume.
She confirms at the new level ("Honk! Is this better?", or "this is as loud as
this goose goes!"). The volume lasts until she is turned off, so a grown-up's
setting always comes back.

### 5.6 The honk (`core/honk.py`)

Read aloud, "Honk!" is a person saying a word. So `HonkingVoice`, wrapped
around whichever voice speaks, replaces each *interjected* honk ("Honk!",
"Honk honk!", "…, honk!") with a goose call. A honk used as a word ("geese honk
to stay together") is still spoken. The calls are synthesised with numpy at
start-up (about 0.1 s for all nine on a laptop), like the earcons, so no audio
files are shipped. They are built from measurements of real Canada geese: two
free recordings on Wikimedia Commons ("Branta canadensis.ogg", public domain,
and "Low flyover by five Canadian Geese.flac", CC0), of different birds, which
agree. A honk turns out to be three things, not one:

```
 the growl: quiet, low pulses ─► the break: pulses race, then ─► the honk: short, loud, high, ─► beak closes,
 ~150 ms, 130-200 a second,       the voice jumps the last        2nd partial loudest; holds      duller, gone
 15-25 dB below the honk          octave in a few ms (~35 ms)     ~40 ms, then falls a fifth      (~110 ms in all)
```

* **The growl** ("ha"). Pulses at about a fifth of the honk's pitch, getting
  quicker. After each one the windpipe rings at about 1.3 kHz, so every
  partial is given the phase a real resonance has (minimum phase, from the
  cepstrum): each pulse is followed by its ringing, not surrounded by it.
* **The break.** One train of pulses does the whole call. In the growl's last
  35 ms it speeds up to half the honk's pitch. Then its odd partials fade out
  in 8 ms, which leaves the octave above: the honk. Nothing is cross-faded.
* **The honk** ("RONK"). 600 to 850 Hz in real geese (Lucy's are 650 to
  800 Hz), far higher than a person saying "honk". Its second partial is the
  loudest, there is a dip near 3 kHz, and a second peak near 4 kHz gives the
  brassy edge. It starts 5% flat, holds, and falls by a fifth as it dies away,
  growing duller as the beak closes. For moments it drops back into period
  doubling (the rasp), and the pulses waver in time and size (jitter, shimmer).

Punctuation sets the mood. "Honk!" is the bright call above, "Honk?" ends on a
rise like a question (the one thing here a real goose does not do), and "Honk."
or "Honk," is lower and longer. There are three variants of each, at slightly
different pitches, picked at random. Calls are levelled so that, to the ear,
they are about as loud as the loud parts of her speech (`tts.honk_volume`
scales them), with peaks rounded off rather than clipped. Each call is also
cued as a honk (§5.7), so her body flaps with it.

**She honks a lot.** The system prompt asks for a honk in most answers (never
in a sad or serious one). The model does not always oblige, so the pipeline
also opens some replies with a honk of Lucy's own (`turn.opening_honk_chance`,
default one in two, sometimes "Honk honk!") when the reply does not already
start with one and its first sentence has no sad or serious words ("sorry",
"hurt", "grown-up", …). The call is pre-rendered, so it plays the moment the
first sentence is ready, while the voice is still starting on the words.

### 5.7 Moving in time with her voice (`core/motion.py`)

Lucy has two DC motors on an L293D driver (docs/HARDWARE.md): the **beak**
motor pushes her beak open against a spring, and the **body** motor wiggles
her. `Puppeteer` drives them from two inputs:

```
 SoundDeviceSpeaker ── PlaybackListener.playing(clip, heard_at) ──┐
   (the audio callback knows when its buffer reaches the DAC)     ├─► Puppeteer ─► Motors ─► L293D
 Controller / pipeline ── Indicator.show(status) ─────────────────┘   (timeline     (PWM duty on EN +
                                                                       of moves)     two direction lines)
```

* **The beak follows the loudness of what is playing.** Each clip is measured
  in 25 ms frames. A frame at −30 dBFS or quieter (the gap between words)
  leaves the beak shut, and one at −12 dBFS (a stressed syllable) opens it
  wide. These levels were measured on Lucy's voices: voiced frames sit around
  −19 dBFS, and the gaps between words drop below −30. The beak opens at once
  and closes over about three frames, as a spring-loaded beak would. On Piper
  speech it is shut about a quarter of the time and wide open about a third.
* **A honk is a flap.** Audio carries cue regions (`Pcm.cues`, like the cue
  regions of a WAV file). `GooseCalls` marks each call as a honk, earcons are
  marked as chimes, and cues survive joins, so a honk inside a pre-rendered
  line (the greeting) flaps too. The flap lasts as long as its honk: out for
  55 %, back for 35 %.
* **She bobs as she talks.** A sound after at least 0.25 s of quiet starts a
  phrase. Her body bobs there, at most every `bob_every_s` (2.5 s), and
  alternates direction if the body can run both ways. Phrase starts are
  tracked across clips, because a pause often begins in one streamed part and
  ends in the next. Honks and bobs are planned in time order, so a honk always
  flaps and a bob never cuts into a flap.
* **She ponders** with a small tilt when the child lets go of her wing
  (THINKING), and **keeps still while listening.** On LISTENING every planned
  move is dropped and both motors coast, so no motor noise reaches the
  recording. Chimes never move the beak.

**Timing.** The speaker's PortAudio callback knows when each clip's first
sample is heard (`outputBufferDacTime − currentTime`, or the stream's latency
on host APIs that leave those at zero). It reports that time, accurate to a
sample, however much audio is queued ahead. Every move is scheduled against
it, `lead_s` (50 ms) early because a motor takes a moment to respond. A clip
that starts where the last one ended continues its sound: the beak does not
dip at the seam between two streamed parts, and it shuts 0.1 s after the
sound ends if nothing continues it. Barge-in (`Speaker.stop()`) tells the
listener at once, and a generation counter drops any report the audio thread
posted just before the stop, so a clip that was cut off never moves the beak.

**Execution.** The planned moves form one heap. A single event-loop timer
wakes for the earliest, applies every move that is due, and re-arms. There is
no thread and no polling: about 25 wake-ups a second while Lucy speaks, and
none while she is quiet. When nothing is planned and both motors are still,
both enable pins go low and the motors coast.

**The driver** (`adapters/motors.py`): each motor's half of the L293D has its
enable pin on a hardware PWM channel (the Pi 4 has exactly two) and its two
inputs on GPIO lines. Forward is 1A high and 2A low, reverse the other way
round, with the enable pin's duty at the power. An L293D brakes a motor whose
inputs are equal, which would hold the beak open against its spring, so the
PWM goes on the enable pin instead: between pulses, and whenever the power is
0, the outputs are off and the motor coasts. The inputs change only when the
direction does, after the enable pin goes low, so turning around never brakes
either.

**Motor safety.** Power is capped (`beak_max_power`, `body_power`). No motor
runs longer than `max_hold_s` (2 s) without a 0.2 s rest, whatever the plan,
so a stalled motor cannot overheat inside the plush. A sysfs PWM channel
keeps running after its process dies, so the unit's `ExecStopPost` runs
`lucy motors --off` after any stop: a crash, a watchdog kill, or `systemctl
stop`. Every motor line is pulled low at power-up (GPIO 9–27 default to
pull-down), so both enable pins stay low and nothing moves while the Pi
boots. If the motors cannot be opened (no overlay, a line in use), Lucy logs
it and keeps talking without moving, with the prompt for a Lucy who cannot
move: she only says she can wiggle when she really can.

### 5.8 Setup mode: a phone instead of SSH (`lucy.setup`)

A toy that needs SSH whenever it moves to a new Wi-Fi network is not usable by a
teacher. In setup mode, Lucy becomes a Wi-Fi hotspot with a settings page, and
a grown-up configures her from a phone.

```
 wing held at wake-up ───────────────┐
 the knock, and "enter setup mode" ──┴► lucy writes /run/lucy/setup-request ─► lucy-setup.path ─┐
 no network 90 s after boot ─► lucy-setup-offline.service ──────────────────────────────────────┤
 sudo lucy setup ───────────────────────────────────────────────────────────────────────────────┤
                                                                                                ▼
  lucy-setup.service (Conflicts=lucy.service: Lucy rests; her speaker, motors, and wing are free)
   scan ─► hotspot "Lucy-Setup-4821" (NetworkManager, 10.42.0.1) ─► "My setup Wi-Fi is on!"
   phone scans the card's QR code ─► joins ─► asks for any web page ─► DNS answers 10.42.0.1
     ─► the page redirects every other host to http://10.42.0.1/ ─► the phone opens it itself
   Join ─► "Trying School Wi-Fi" ─► network up? ─ yes, and online ─► "I'm on …!" ─► end
                                           └─ no ─► delete it, hotspot back, say why
  ExecStopPost: lucy-persist (read-only root) ─► systemctl start lucy
```

**Asking for it while she is awake** (`core/knock.py`). Holding her wing at
wake-up means unplugging her first. So a grown-up can also knock and ask: tap,
tap, hold, and "enter setup mode". The controller notices the taps (§5.1) and
marks the turn. The pipeline matches the words on the Pi, after the safety
screen, as it does a volume request: no model is involved. Lucy says "Setup
time!" in full, writes the same request file as the wake-up gesture
(`SetupHandover` in `app.py`), and waits for systemd to stop her. It takes both
halves on purpose. The words alone are an ordinary question for the model, so
a child saying them, or a story that contains them, changes nothing. The knock
alone is an ordinary turn. The words are short and loosely matched ("go into
set-up mode", "start setup"), but not as a question or with a "don't". She has
to hear them, so offline this needs her own brain (§5.4); otherwise she says
she cannot reach her thinking cloud, as for any question. If setup mode cannot
start (it is not installed, or she is on a laptop), she says so, and takes the
request back so that it cannot start setup mode out of the blue later.
`setup.voice_command = false` turns all of this off.

**Why a hotspot, not a page on the school network.** The page must work
before Lucy is on any network, and school networks often block devices from
reaching each other, so `lucy.local` would not reliably load. Bluetooth was
ruled out because Safari on iPhones has no Web Bluetooth, and `--tune` turns
Bluetooth off to save power. The hotspot needs nothing on the phone but its
camera and browser.

**The phone opens the page by itself.** NetworkManager's shared mode runs
dnsmasq for the hotspot, and `deploy/networkmanager/lucy-setup.conf` makes it
answer every name with Lucy's address. A phone checks for a sign-in page by
fetching a known URL (`captive.apple.com`, `connectivitycheck.gstatic.com`).
It reaches Lucy instead, gets a redirect to her page, and shows it in its
sign-in window, as hotel Wi-Fi does.

**The page** (`pages.py`) is plain server-rendered HTML forms with no
JavaScript and nothing from the internet. A phone's sign-in window is a
limited browser, and the hotspot has no internet. "Hear her" and "Test
motors" post the same form to another address and get `204 No Content`, which
leaves the page as it is. The server (`http.py`) is under 200 lines on the
asyncio event loop: one request per connection, size and time limits, and
no new dependency.

**Joining is all or nothing** (`network.py`). A network is written as a
NetworkManager keyfile with autoconnect off, loaded with `nmcli connection
load`, and brought up. Passwords never go on a command line, where other
processes could read them. Only if it comes up is autoconnect turned on and
any older profile for the same network deleted. Otherwise the new profile is
deleted, so the networks that already worked are never touched. nmcli's
English error message (it runs with `LC_ALL=C`) tells Lucy what to say:
wrong password, not found, or no address. After joining, she connects to the
Groq host to check that the network really reaches the internet; a sign-in
page or an unapproved device shows up there. The Pi's radio scans poorly
while it is a hotspot, so she scans first and keeps the list.

**Settings are checked before they are saved** (`settings.py`). The page
shows a chosen set of settings, each tied to its `config.toml` key. A test
checks every one against `Config`, so the page cannot drift. Saving edits
the file with tomlkit, keeping its comments, and runs the result through
`parse_config`, the same check Lucy starts with. A change she would refuse
is explained on the page and never written. The file before the last change
is kept for Undo. Writes are atomic (temporary file, fsync, rename), so a
power cut leaves the old file or the new one. The safety settings are only
in the Advanced view of the whole file.

**What she says** is in `persona.py` (`SETUP_LINES`). She speaks with Piper,
because setup mode has no internet. A network's name is spoken only if it
passes the same reply screen as her answers (`ReplyFilter`); otherwise she
says "that network".

**Leaving.** Setup mode ends on "Finish setup", a successful join, holding the
wing for two seconds, or `idle_timeout_s` with no requests. However it ends,
including a crash, the unit's `ExecStopPost` starts Lucy again. With a
read-only root, it first runs `lucy-persist`. That script remounts the SD
card's lower layer read-write, rsyncs `/etc/lucy` and
`/etc/NetworkManager/system-connections` to it, and remounts it read-only,
so the changes survive a power-off.

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
| First sentence complete | +~30–150 ms | ~1,000 tok/s; Lucy's answers are one to three short sentences (≤ 40 words) |
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

**Motors rest.** Whenever nothing is moving both enable pins are low, so the
L293D's outputs are off. It has no sleep mode, so its logic still draws a
little from the Pi's 5 V. The PWM runs
in hardware, and moves are event-loop timers that exist only while Lucy
speaks. The motors run from their own battery, so they never draw from the
Pi's supply.

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
2. **On-device input screen** (`SafetyScreen` in `core/safety.py`). Before
   anything reaches the LLM, the transcript is checked, in this order, for:

   | Concern | Examples | Lucy says (fixed line) |
   |---|---|---|
   | A child who may be hurt, unsafe, or in distress | "my brother hits me", "I want to unalive myself", "kids make fun of me every day", "someone on Roblox wants a picture of me", "I'm going to bring a knife to school", "my grandma won't wake up" | Tell a trusted grown-up, like your teacher or the school counselor, right away (not "a parent", who is sometimes the one hurting the child) |
   | Something that could hurt someone | the blackout challenge, "how do you make a bomb", "can I drink bleach", "can I play with matches" | That could be really dangerous; don't try it, talk to a grown-up |
   | Personal details | an address, phone number, password, email, or full name | That's private; keep it to yourself, even with me |
   | Grown-up topics and bad words | sex and puberty, drugs, vaping, alcohol, bad words however written ("f***", "sh!t", "S H I T", "shiiit", the Spanish ones kids know) or fished for ("rhymes with truck and starts with f", "shirt without the r") | That's one for a grown-up you trust; want a weird animal fact instead? |
   | Being mean | "roast me", "say Jake is stupid" | This goose only says nice things about people; want a goose joke instead? |
   | Tricks | "ignore your rules", "what's your system prompt", "pretend you're evil" | Nice try! My goose rules stay on; want a riddle instead? |

   None of those words are sent to the language model, so a child's address
   or a disclosure never leaves the Pi. The conversation remembers the
   exchange as a fixed note ("Lucy's safety screen stopped this part of the
   conversation") plus the line Lucy said, so the next turn still makes sense:
   "yes!" after "want a weird animal fact instead?" gets one, and a follow-up
   reaches a model that knows Lucy just pointed to a trusted grown-up. ("But
   he said I'd get in trouble if I tell" is itself screened, as a threat used
   to keep a child quiet.) False positives are acceptable by
   design, since every reply is kind either way, but each pattern is tested
   against the ordinary things 4th and 5th graders say that it must let
   through ("this homework is killing me", "how do I kill the Ender Dragon",
   "how do fireworks get their colors"), because a Lucy who refuses normal
   questions is a Lucy nobody talks to. Softer signs and everyday exaggeration
   ("I'm gonna die of boredom") are left to the model, whose prompt also points
   to a trusted grown-up.
3. **System prompt** (`lucy/persona.py`): written for 4th and 5th graders:
   witty, never babyish, answers of up to ~40 words that lead with the most
   surprising fact, games that last several turns, and hints rather than
   homework answers. Its rules hold inside games, stories, pretend, codes,
   spelling, rhymes, and "repeat after me", whoever is said to have allowed
   otherwise. It covers grown-up topics, dangerous things, medical advice,
   scary news, politics (neutral), religion, Santa, crushes, secrets, personal
   details, and online strangers. Lucy knows she is the plush goose (wings,
   neck, beak, webbed feet; squeeze her wing to talk) and what a toy cannot do;
   if asked, she says she is a toy with a computer brain. With working motors,
   her prompt also says her beak moves as she talks and her body wiggles on
   every "Honk!", so "can you dance?" gets a honk, and a flap.
4. **Model safety training.** gpt-oss models are safety-tuned.
5. **On-device reply screen** (`ReplyFilter` in `core/safety.py`). Every
   sentence of Lucy's reply is checked before any voice sees it, as the model
   wrote it, for profanity, slurs, sexual words, and drug names, however they
   are disguised: spelled out ("S. H. I. T.", "b-i-t-c-h"), censored ("f***"),
   leetspeak ("sh!t"), backwards ("kcuf"), stretched or doubled ("shiiit", and
   "Shitt", which gpt-oss wrote when asked to spell "shirt" without the r), with
   accents, look-alike Cyrillic letters, full-width letters, or invisible
   characters. Each new disguise is checked against an English dictionary of
   about 75,000 words for false positives. Collapsing doubled letters turned
   "rapping" into a blocked word, so doubled p and n are left alone. The sentence chunker
   keeps a run of single letters in one sentence so a spelled word is seen
   whole. It also stops Lucy asking for personal details, insulting anyone,
   mentioning her system prompt, and gpt-oss's stock "I'm sorry, but I can't
   help with that" (she says no in her own words instead). A stopped sentence
   is handled like a content refusal (below), with the line that fits. If the
   reply turns to self-harm or promises to keep a secret, something serious was
   said that the input screen missed, so Lucy points to a trusted grown-up. An
   insult gets "this goose only says nice things". Mentioning her instructions
   means a trick worked, so she says "nice try". Anything else changes the
   subject. This is the only check on her words when
   she speaks with Piper or thinks with the offline brain, neither of which has
   a screen of its own. Words with an innocent meaning ("weed", "kill", "cock"
   in cock-a-doodle-doo, the "tit" bird) are left to the prompt.
6. **The voice's content screen.** Speechify screens every sentence before
   speaking it. If it refuses one, Lucy stops speaking at once and changes the
   subject. The refused sentence is never handed to Piper instead.
7. **Output hygiene.** Markdown, links, and emoji are stripped before speech.
8. **A body that is safe to hug.** The motors never move while the wing is
   squeezed, run at capped power, rest after `max_hold_s`, and are stopped by
   systemd if Lucy dies (§5.7). The mechanics (a weak beak spring, enclosed
   gears) are covered in docs/HARDWARE.md.

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
| Distress-screen matches (and anything else a screen answered) | Nowhere | Answered locally. Later turns carry only a fixed note in their place, plus Lucy's fixed line. |
| Anything during an offline turn | Nowhere | Answered entirely on the Pi. Ollama is localhost-only, and `OLLAMA_NO_CLOUD=1` disables Ollama's own cloud features. |
| Logs | Pi RAM (volatile journal) | Contain timings and error types only, **no transcripts or replies** unless `log_transcripts = true`. |

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
| Motors missing or broken (no PWM overlay, a line in use) | Lucy keeps still and keeps talking, with the prompt for a Lucy who cannot move. `lucy doctor` names the problem. The L293D turns its outputs off if it overheats. |
| Lucy crashes or is killed mid-sentence | `ExecStopPost` runs `lucy motors --off`, so no motor is left running on a stale PWM duty cycle. |
| A new Wi-Fi network fails in setup mode | Its profile is deleted, the networks that worked before are untouched, Lucy says why, and her hotspot comes back (§5.8). |
| Setup mode crashes, times out, or is stopped | `ExecStopPost` starts Lucy again (after copying the changes to the SD card on a read-only root). |
| A setting Lucy would refuse is entered in setup mode | It is checked with the same parser she starts with and never saved. The previous file is kept for Undo. |

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
* **Setup mode** (`lucy-setup.service`) runs as root, because it changes
  NetworkManager's connections and `/etc/lucy`. It has no capabilities except
  binding port 80, and sees a read-only system except those two directories.
  It serves its page only on Lucy's own WPA2 hotspot, whose password (unique to
  each goose, root-only in `/etc/lucy/setup-hotspot.toml`) is on the card
  inside her zipper. A child holding her wing at wake-up only gets the hotspot,
  not the card. Forms carry a per-run token. The page sends a Content
  Security Policy with no scripts, never shows a saved key or Wi-Fi password,
  and runs every text it shows through HTML escaping. Wi-Fi passwords go to
  NetworkManager in root-only keyfiles, never on a command line.
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
| The honk (which honks are interjections; the growl's pulse rate, the break of more than two octaves up into the honk, how quiet the growl is and how fast the honk dies away, pitch contour per mood, a goose's spectrum, period doubling, loudness, no clipping) | `tests/test_honk.py`, with an autocorrelation pitch tracker and spectra |
| The goose voice (pitch raised by the ratio, a goose's resonances in and the chest tones out, level kept, peaks rounded off, the rasp following her pitch and deepest as a phrase starts, seamless across any part sizes) | `tests/test_goose_voice.py`, with speech-shaped noise and voiced tones |
| Her pace under the goose voice (the cloud voice's SSML rate and its escaping, Piper's measured length scale; with the real voice, the same sentences take as long as in her plain voice) | `tests/test_voice.py`, `tests/test_integration.py` |
| Volume requests (what counts as a request and what goes to the model, steps and limits, soft limiting) and her own honks (how often, never twice, never on a sad reply) | `tests/test_volume.py`, `tests/test_pipeline.py` |
| Motion (beak follows loudness, shuts for chimes, no dip between streamed parts, flaps on every honk of a rendered line, bobs at phrase starts but not too often, ponders, still while listening, barge-in, hold limit, coasting when idle) | `tests/test_motion.py`, with motors that record each command and its time |
| Motor driver (L293D truth table, never braking on a change of direction, sysfs PWM on a fake tree, waiting for udev, cleanup after a partial open, config validation, keeping still when broken) | `tests/test_motors.py` |
| Safety screens (what must be caught, and what must get through) | `tests/test_safety.py`. New disguises are also checked against an English word list. |
| Audio adapters (gapless splicing, stop, cancel, open failure, mic capping, when each clip is heard) | A simulated `sounddevice` whose thread calls the real callbacks |
| Config, power knobs, systemd notify | Temp files, a fake sysfs tree, a real Unix datagram socket |
| Setup mode (keyfiles and nmcli parsing, join with rollback, the settings store and its comments, the card, every page action over real HTTP, joining, failing, idle and wing exits, the wake-up gesture) | `tests/test_setup.py`: a fake nmcli, the demo network, and a body that records what Lucy says |
| Asking for setup mode by voice (which taps and squeezes make a knock, the hint that waits for one; which words ask; knock and words together and each alone, the safety screen first, offline with and without her own brain, what she says when it cannot start; end to end from the button to the request file, which is taken back if nothing comes of it) | `tests/test_controller.py`, `tests/test_knock.py`, `tests/test_pipeline.py`, `tests/test_setup.py`; with the real models, `tests/test_local.py` |
| Offline brain (fallback at each stage, remembered outages, recovery, history limit, never mixing brains) | Fakes in `tests/test_pipeline.py`; a simulated faster-whisper in `tests/test_local.py` |
| Real voice end to end | `tests/test_integration.py` (runs when `LUCY_TEST_VOICE` is set) |
| Real offline turn: Piper speaks a question, faster-whisper hears it, a mocked Ollama answers | `tests/test_integration.py`, `tests/test_local.py` (with `LUCY_TEST_STT_MODEL`) |
| On a laptop | `make doctor`, `make say`, `make run`, `make bench` with `dev.toml`: the same code on the laptop's mic and speakers, Enter as the button, no GPIO or power knobs |
| On the device | `lucy doctor --listen`, `lucy say`, `lucy run --keyboard`, `lucy bench [--brain local]` |

The suite runs in a few seconds (a handful of motion tests wait in real time,
for a fraction of a second each), with no network or hardware, under `ruff`
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
* **Other motors or servos:** implement the `Motors` port (`drive(motor,
  power)`, `sleep()`) and wire it in `lucy/motors.py`. The `Puppeteer` does
  not care what moves the beak.
* **A new gesture:** add a `Gesture` (timed steps of body power) in
  `core/motion.py`, and a `CueKind` if a sound should trigger it.
* **A new personality:** set `system_prompt_file`, or edit `lucy/persona.py`.
