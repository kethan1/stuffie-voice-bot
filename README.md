# Lucy the Goose 🪿

Lucy is a talking plush goose for elementary-school kids, built on a Raspberry
Pi 4B. Squeeze her wing, ask a question, and let go. She answers out loud in
one to three short, friendly sentences, usually starting within about a second.

```
 squeeze wing ─► record (16 kHz) ─► Groq Whisper ─► on-device safety screen
       ─► Groq gpt-oss-20b (streamed) ─► sentence chunks ─► Piper TTS on the Pi ─► speaker
```

* **Fast:** everything streams. Lucy starts speaking the first clause while the
  rest is still being written. The HTTPS connection is warmed up while the
  child is still talking. Wi-Fi power saving and CPU throttling are switched
  off only for the moments that matter.
* **Frugal:** event-driven with no polling. The mic is open only while the
  wing is squeezed, the amp and speaker sleep when idle, and three power
  profiles (active / standby / dormant) apply on a headless, tuned
  Raspberry Pi OS Lite.
* **Kind and safe:** an on-device screen catches signs that a child is hurt or
  upset and answers with a fixed, caring script that points to a trusted
  grown-up. A careful system prompt covers everything else. No recordings are
  kept and no transcripts are logged.
* **Robust:** Lucy explains problems out loud ("Can a grown-up check the
  Wi-Fi?"). She falls back to a second model when rate-limited, is restarted
  by the systemd watchdog if wedged, and supports a read-only root filesystem
  for power yanks.

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
   Zero Data Retention.
3. Install, then reboot:

   ```bash
   git clone <this repository> lucy && cd lucy
   sudo deploy/install.sh --i2s-amp --tune     # paste the API key when asked
   sudo reboot
   ```

4. Check everything, then talk to her:

   ```bash
   sudo lucy doctor --listen
   sudo lucy say "Honk! Hello!"
   ```

### API key

The key is stored in `/etc/lucy/credentials/groq_api_key` (root-only) and
reaches the service only as a systemd credential. It never goes in the config
file or the repository. For development on a laptop, export `GROQ_API_KEY`
instead.

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). On Linux, the
audio adapters also need PortAudio (`sudo apt install libportaudio2`).

```bash
make dev          # uv sync: create .venv with Lucy and the dev tools from uv.lock
make check        # ruff + mypy --strict + pytest (no hardware or network needed)
make voice        # download the default Piper voice into ./voices
make integration  # end-to-end test with the real voice and a mocked Groq
make lock         # re-resolve uv.lock to the newest releases at least two weeks old

export GROQ_API_KEY=...            # then, on a laptop with a mic and speakers:
uv run lucy --config dev.toml run --keyboard
```

Add or change dependencies with `uv add` / `uv remove` so `uv.lock` stays in step;
the installer refuses a stale lock.

`dev.toml` only needs the settings that differ from the defaults, for example:

```toml
[tts]
voice = "voices/en_US-amy-medium.onnx"
[led]
enabled = false
[power]
manage_cpu_governor = false
manage_wifi = false
```

## Project layout

```
src/lucy/
  core/          pure logic: controller (state machine), pipeline (streaming turn),
                 chunker, text cleaning, safety screen, conversation, phrases, ports
  adapters/      Groq/OpenAI-compatible HTTP, Piper, PortAudio, libgpiod, power knobs
  platform/      systemd notify/watchdog, journald-aware logging
  tools/         `lucy doctor`, `lucy bench`
  persona.py     Lucy's personality spec, system prompt, and fixed lines
  app.py         composition root
  config.py      strict TOML configuration
deploy/          installer, systemd unit, ALSA routing, boot tuning
uv.lock          hash-locked dependencies for every platform, including the Pi
tests/           fast tests with fakes for every port (no hardware, no network)
docs/            architecture, hardware, operations
```
