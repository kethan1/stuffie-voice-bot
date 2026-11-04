# Lucy the Goose: Operations

## 1. Prepare the SD card

1. In **Raspberry Pi Imager**, choose *Raspberry Pi 4* and **Raspberry Pi OS Lite
   (64-bit)**.
2. In the OS customisation settings, set the hostname to `lucy`, create a user,
   enter the Wi-Fi network and country, and enable SSH.
3. Boot the Pi, then `ssh <user>@lucy.local`.

## 2. Get an API key (a grown-up does this)

Groq requires account holders to be 18 or older.

1. Sign in at <https://console.groq.com> and create an API key.
2. Turn on **Zero Data Retention**: *Settings → Data Controls*
   (<https://console.groq.com/settings/data-controls>).
3. Keep the key secret. Never commit it, paste it into config files, or share
   it in chat.

## 3. Install

```bash
sudo apt-get install -y git
git clone <this repository> lucy && cd lucy
sudo deploy/install.sh --i2s-amp --tune      # paste the API key when asked
sudo reboot
```

| Option | Use it when |
|---|---|
| `--i2s-amp` | You wired the MAX98357A amplifier |
| `--playback-card ID`, `--capture-card ID` | Auto-detection picked the wrong sound card (`aplay -l`, `arecord -l`) |
| `--voice en_US-lessac-medium` | You want a different Piper voice |
| `--tune` | Always, for a finished toy: boot-time power savings, unused services off, journal in RAM |
| `--local-backup` | You want Lucy to keep answering when GroqCloud is unreachable (no Wi-Fi, school firewall, outage). Installs Ollama with Gemma 3 1B and an on-device speech model: about 2 GB of SD card space and a ~2.5 GB one-time download. Answers offline take several seconds. |
| `--read-only` | Last step, once everything works: protects the SD card from sudden power loss |

To set or replace the API key later:

```bash
sudo install -m 600 /dev/stdin /etc/lucy/credentials/groq_api_key   # paste, Enter, Ctrl-D
sudo systemctl restart lucy
```

## 4. Bring-up checklist

The installer adds a `lucy` command that runs exactly as the service does (same
user, groups, API-key credential, and capability), so what it reports is what
the service experiences.

```bash
sudo lucy doctor --listen          # every line should be ✔; speak when asked
sudo lucy say "Honk! Can you hear me?"
sudo systemctl stop lucy && sudo lucy run --keyboard   # Enter to talk, Enter to stop; Ctrl-C quits
sudo systemctl start lucy
sudo lucy bench                    # latency breakdown, measured on this Pi
```

`lucy bench` prints each stage measured from the button release. If
`first_audio` is high, compare `stt` and `first_token` (network or Groq) with
the gap to `first_audio` (Piper on the Pi).

### Offline brain (if installed with `--local-backup`)

```bash
sudo lucy doctor                   # includes "local speech model" and "local brain (Ollama)"
sudo lucy bench --brain local      # offline latency on this Pi
```

To rehearse an outage without touching the Wi-Fi (which would also drop your
SSH session), point Lucy at an address that can never resolve, then put it back:

```bash
sudo sed -i 's|^base_url = "https://api.groq.com|base_url = "https://offline.invalid|' /etc/lucy/config.toml
sudo systemctl restart lucy        # squeeze the wing: Lucy says she'll think on her own
sudo sed -i 's|^base_url = "https://offline.invalid|base_url = "https://api.groq.com|' /etc/lucy/config.toml
sudo systemctl restart lucy
```

The journal shows `cloud brain unreachable; using the local brain` and
`turn complete (local brain): …`.

## 5. Day-to-day

| Task | Command |
|---|---|
| Live logs | `journalctl -u lucy -f` (and `journalctl -u ollama -f` for the offline brain) |
| Status | `systemctl status lucy` |
| Restart after a config change | `sudo systemctl restart lucy` |
| Temperature / throttling | `vcgencmd measure_temp; vcgencmd get_throttled` |

Every turn logs one line such as
`turn complete (cloud brain): stt=0.41s first_token=0.68s first_synth=0.93s first_audio=0.93s total=7.12s`.
Offline turns say `(local brain)` and include `fallback=…`.
Children's words are **not** logged unless `turn.log_transcripts = true`.

## 6. Tuning

All settings are in `/etc/lucy/config.toml` (documented inline).

| To get | Change |
|---|---|
| Faster first word | A `-low` voice (16 kHz, cheaper to synthesise); `turn.first_clause_min_words = 3` |
| Longer battery life | Shorter `session.awake_s`; `power.standby_governor = "powersave"` |
| Slower, clearer speech for little ones | `tts.length_scale = 1.15` |
| Fewer "I didn't hear that" | Lower `turn.speech_threshold_dbfs` (e.g. `-55`), or raise mic gain in `alsamixer` |
| Answers that are less creative and more steady | `llm.params.temperature = 0.5` |
| More context for follow-ups | `llm.history_exchanges = 10` (slightly slower and costlier) |
| Faster offline answers | `sudo ollama pull gemma3:270m`, then `local.model = "gemma3:270m"` (quicker but simpler answers); `local.params.max_tokens = 60` |
| More accurate offline hearing | A `base.en` faster-whisper model in `local.stt_model` (about twice as slow as `tiny.en`) |

## 7. Updating

```bash
sudo raspi-config nonint disable_overlayfs && sudo reboot   # only if --read-only was used
cd ~/lucy && git pull
sudo deploy/install.sh --tune            # re-runnable; keeps your config and key
sudo deploy/install.sh --read-only && sudo reboot           # only if it was read-only
```

Edited config files are never overwritten. If the template changed, the new
version is written next to yours as `config.toml.new` so you can compare them.
If your config enables the offline brain, re-running the installer keeps it
installed and up to date without `--local-backup`.

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Silent; LED does nothing | Service not running | `systemctl status lucy`, `journalctl -u lucy -e` |
| "Can a grown-up check the Wi-Fi?" | No network or DNS | `ping -c3 api.groq.com`; check Wi-Fi with `nmcli` |
| "Something inside me needs a grown-up's help" | Missing or invalid API key, or an audio device error | `lucy doctor` names which one |
| "My goose brain needs a little rest" | Groq rate limit (both models) or the hourly turn budget | Wait a minute. Check `session.max_turns_per_hour` |
| Often "I didn't hear that" | Mic too quiet | `lucy doctor --listen`, then `alsamixer` (F4 for capture) |
| Lucy hears herself | Mic too close to the speaker | Move them apart. Lucy only listens while the wing is held |
| Crackles or gaps | CPU starved or weak power | `vcgencmd get_throttled` (non-zero = under-voltage); raise `audio.output_latency_s` to `0.12` |
| Button does nothing | Wiring or pin number | `lucy doctor`; `gpioinfo` shows `lucy-button` on line 17 |
| `CPU governor ... not writable` in doctor | Service never started since boot | Start it once; its `ExecStartPre` grants access |
| Lucy "thinks on her own" though the Wi-Fi works | Groq is blocked (firewall, DNS) | `curl -sI https://api.groq.com`; the next press re-checks automatically |
| Offline: "Can a grown-up check the Wi-Fi?" | The offline brain is not running | `systemctl status ollama`; `ollama list` should show `gemma3:1b`; `sudo lucy doctor` |
| Journal: `local brain is not ready` | Ollama started after Lucy, or the model is missing | `sudo ollama pull gemma3:1b && sudo systemctl restart lucy` |

## 9. Uninstall

```bash
sudo systemctl disable --now lucy ollama
sudo rm -rf /opt/lucy /etc/lucy /var/cache/lucy /usr/local/bin/lucy /etc/systemd/system/lucy.service \
  /etc/systemd/journald.conf.d/lucy.conf
# Only if --local-backup was used: Ollama, its libraries, and its models
sudo rm -rf /usr/local/bin/ollama /usr/local/lib/ollama /etc/systemd/system/ollama.service \
  /var/lib/private/ollama /var/lib/ollama
sudo userdel lucy
sudo mv /etc/asound.conf.lucy-backup /etc/asound.conf 2>/dev/null || sudo rm -f /etc/asound.conf
sudo cp /boot/firmware/config.txt.lucy-backup /boot/firmware/config.txt   # if --tune/--i2s-amp was used
sudo systemctl daemon-reload && sudo reboot
```
