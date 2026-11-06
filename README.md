# Lucy the Goose 🪿

Lucy is a talking plush goose for 4th and 5th graders, built on a Raspberry
Pi 4B. Squeeze her wing, ask a question, and let go. She answers out loud in
one to three short, witty sentences, usually starting within about a second.

```
 squeeze wing ─► record (16 kHz) ─► Groq Whisper ─► on-device safety screen
       ─► Groq gpt-oss-20b (streamed) ─► sentences ─► Speechify voice (streamed) ─► speaker
                                                      └► Piper on the Pi, if the cloud voice can't
```

* **Natural voice:** Lucy speaks with Speechify's `simba-3.2`, one of the most
  natural-sounding voices available (11th of 103 in a blind listening arena).
  It is free for up to 500,000 characters a month, about 3,000 replies, with no
  credit card. Her on-device Piper voice takes over whenever the cloud voice
  can't speak, so she never goes silent. When she says "Honk!" you hear a goose
  call, not the word read aloud.
* **Alive:** her beak opens and shuts in time with every syllable, and her body
  flaps whenever she honks and bobs as she starts each phrase. Two small motors
  on a DRV8833 driver follow the sound itself, as it reaches the speaker. She
  keeps still while she listens, and if the motors aren't wired she simply
  talks without them.
* **Fast:** everything streams. Lucy starts speaking her first sentence while
  the rest is still being written, and plays its audio as it arrives. The
  HTTPS connections are warmed up while the child is still talking. Wi-Fi
  power saving and CPU throttling are switched off only for the moments that
  matter.
* **Frugal:** event-driven with no polling. The mic is open only while the
  wing is squeezed, the amp and speaker sleep when idle, and three power
  profiles (active / standby / dormant) apply on a headless, tuned
  Raspberry Pi OS Lite.
* **Kind and safe:** an on-device screen catches signs that a child is hurt or
  upset and answers with a fixed, caring script that points to a trusted
  grown-up. Dangerous dares, private details, grown-up topics, swear words
  (however they're disguised), meanness, and attempts to trick her each get a
  fixed reply in her own voice, and none of those words leave the Pi. A second
  screen checks every sentence Lucy is about to say. A careful system prompt
  covers everything else. No recordings are kept and no transcripts are
  logged.
* **Robust:** Lucy explains problems out loud ("Can a grown-up check the
  Wi-Fi?"). She falls back to a second model when rate-limited, is restarted
  by the systemd watchdog if wedged, and supports a read-only root filesystem
  for power yanks.
* **Works offline (optional):** with `--local-backup`, Lucy keeps answering
  when GroqCloud is unreachable. She uses faster-whisper and Gemma 3 1B
  (Ollama) on the Pi itself, says she is "thinking with her own goose brain",
  and switches back to the cloud as soon as it is reachable again.

The design rationale (technology choices, latency budget, power strategy,
safety model, and failure handling) is in
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How Lucy is built and why |
| [docs/HARDWARE.md](docs/HARDWARE.md) | Parts list, wiring, building it into the plush |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Install, bring-up, tuning, updating, troubleshooting |
| [instructions.pdf](instructions.pdf) | The research assignment and the model decision behind this project |

## Quick start (on the Pi)

1. Flash **Raspberry Pi OS Lite (64-bit)**, enable SSH and Wi-Fi, and wire the parts
   ([docs/HARDWARE.md](docs/HARDWARE.md)).
2. A grown-up creates a Groq API key at <https://console.groq.com> and turns on
   Zero Data Retention. For Lucy's natural voice, they also create a free
   Speechify API key at <https://platform.speechify.ai/api-keys> (no credit card).
3. Install, then reboot:

   ```bash
   git clone <this repository> lucy && cd lucy
   sudo deploy/install.sh --i2s-amp --motors --tune   # paste the API keys when asked
   # no motors wired yet: leave out --motors
   # optional offline brain (~2 GB):  add --local-backup
   sudo reboot
   ```

4. Check everything, then talk to her:

   ```bash
   sudo lucy doctor --listen
   sudo lucy say "Honk! Hello!"     # with motors: watch her beak and flap
   ```

### API keys

The keys are stored in `/etc/lucy/credentials/groq_api_key` and
`/etc/lucy/credentials/speechify_api_key` (root-only) and reach the service only
as systemd credentials. They never go in the config file or the repository. On
a laptop, put them in `.env` instead (see below).

The Speechify key is optional. Without it, or once the month's free characters
are used up, Lucy speaks with her on-device voice. `sudo lucy voices` lists the
cloud voices, with samples to listen to; set `voice_id` under `[tts.cloud]` in
`/etc/lucy/config.toml` to change hers.

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/), on Linux or macOS. On
Linux, the audio adapters also need PortAudio (`sudo apt install libportaudio2`).
The macOS wheels include it.

```bash
make dev          # uv sync: create .venv with Lucy and the dev tools from uv.lock
make check        # ruff + mypy --strict + pytest (no hardware or network needed)
make integration  # end-to-end test with the real voice and a mocked Groq
make lock         # re-resolve uv.lock to the newest releases at least two weeks old
```

Add or change dependencies with `uv add` / `uv remove` so `uv.lock` stays in step;
the installer refuses a stale lock.

### Running Lucy on a laptop

Lucy runs on a laptop's own microphone and speakers, with the Enter key standing
in for her wing. [`dev.toml`](dev.toml) holds the few settings that differ from
the Pi. It uses the voice in `./voices` and the keyboard button, has no LED, and
leaves the laptop's CPU governor and Wi-Fi alone. Everything else, including
the Groq models, the safety screen, and the streaming pipeline, is the same code
and configuration the Pi runs.

Put your keys in a file named `.env` (gitignored) in the repository as
`GROQ_API_KEY=gsk_...` and `SPEECHIFY_API_KEY=...` (see `.env.sample`), or export
them. Lucy reads `.env` from the directory you run her in. Then:

```bash
make doctor       # fetches the backup voice the first time; checks keys, both voices, mic, speakers, Groq
make say          # Lucy says hello through the speakers (TEXT="..." to change it)
make voices       # the cloud voices she can speak with, with samples
make run          # talk to her: Enter to start, Enter again to stop; Ctrl-C quits
make bench        # latency breakdown, measured from this laptop
```

These are shorthand for `uv run lucy --config dev.toml <command>`,
which also takes any other option (`lucy --help`). Use `CONFIG=my.toml` to try a
different config. With `dev.toml`, the log shows what Lucy heard and what the model replied on each turn.

* **Other sound devices:** `uv run python -m sounddevice` lists them. Put a name
  in `audio.input_device` / `audio.output_device` in `dev.toml`.
* **The offline brain:** run `make stt-model`, install [Ollama](https://ollama.com),
  run `ollama pull gemma3:1b`, and set `[local] enabled = true` in `dev.toml`.
  `make doctor` then checks it too, and `uv run lucy --config dev.toml bench
  --brain local` measures it. To rehearse an outage, start Lucy
  with the laptop's Wi-Fi off. She answers with her own goose brain, and goes
  back to the cloud a turn or so after the Wi-Fi returns.

## Project layout

```
src/lucy/
  core/          pure logic: controller (state machine), pipeline (streaming turn),
                 chunker, text cleaning, safety screen, conversation, phrases, motion, ports
  adapters/      Groq/OpenAI-compatible HTTP, Speechify, Piper, PortAudio, libgpiod,
                 motors (DRV8833 on hardware PWM), power knobs
  platform/      systemd notify/watchdog, journald-aware logging
  tools/         `lucy doctor`, `lucy bench`
  persona.py     Lucy's personality spec, system prompt, and fixed lines
  app.py         composition root (brains.py, voices.py, and motors.py assemble the parts)
  config.py      strict TOML configuration
deploy/          installer, systemd unit, ALSA routing, boot tuning
dev.toml         laptop settings for development (`make run`)
uv.lock          hash-locked dependencies for every platform, including the Pi
tests/           fast tests with fakes for every port (no hardware, no network)
docs/            architecture, hardware, operations
```
