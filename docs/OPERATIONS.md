# Lucy the Goose: Operations

## 1. Prepare the SD card

1. In **Raspberry Pi Imager**, choose *Raspberry Pi 4* and **Raspberry Pi OS Lite
   (64-bit)**.
2. In the OS customisation settings, set the hostname to `lucy`, create a user,
   enter the Wi-Fi network and country, and enable SSH.
3. Boot the Pi, then `ssh <user>@lucy.local`.

## 2. Get the API keys (a grown-up does this)

Groq requires account holders to be 18 or older.

1. Sign in at <https://console.groq.com> and create an API key.
2. Turn on **Zero Data Retention**: *Settings → Data Controls*
   (<https://console.groq.com/settings/data-controls>).
3. For Lucy's natural voice, sign up at <https://platform.speechify.ai> (the free
   plan needs no credit card) and create a key at
   <https://platform.speechify.ai/api-keys>. This key is optional: without it
   Lucy speaks with her on-device voice, which sounds more robotic.
4. Keep the keys secret. Never commit them, paste them into config files, or
   share them in chat.

The free Speechify plan includes 500,000 characters a month, roughly 3,000 of
Lucy's replies. When it runs out, Speechify refuses requests instead of
billing, and Lucy uses her on-device voice until the 1st of the next month.
Usage is shown at <https://platform.speechify.ai/usage>.

## 3. Install

```bash
sudo apt-get install -y git
git clone <this repository> lucy && cd lucy
sudo deploy/install.sh --i2s-amp --motors --tune   # paste the API keys when asked
sudo reboot
```

| Option | Use it when |
|---|---|
| `--i2s-amp` | You wired the MAX98357A amplifier |
| `--motors` | You wired the beak and body motors to a DRV8833 (HARDWARE.md, "Motors"). Adds the hardware PWM overlay, lets the `gpio` group use PWM, and turns on `[motors]` |
| `--playback-card ID`, `--capture-card ID` | Auto-detection picked the wrong sound card (`aplay -l`, `arecord -l`) |
| `--voice en_US-hfc_female-medium` | You want a different on-device (backup) Piper voice |
| `--tune` | Always, for a finished toy: boot-time power savings, unused services off, journal in RAM |
| `--local-backup` | You want Lucy to keep answering when GroqCloud is unreachable (no Wi-Fi, school firewall, outage). Installs Ollama with Gemma 3 1B and an on-device speech model: about 2 GB of SD card space and a ~2.5 GB one-time download. Answers offline take several seconds. |
| `--read-only` | Last step, once everything works: protects the SD card from sudden power loss |

To set or replace an API key later:

```bash
sudo install -m 600 /dev/stdin /etc/lucy/credentials/groq_api_key        # paste, Enter, Ctrl-D
sudo install -m 600 /dev/stdin /etc/lucy/credentials/speechify_api_key   # Lucy's natural voice
sudo systemctl restart lucy
```

## 4. Bring-up checklist

The installer adds a `lucy` command that runs exactly as the service does (same
user, groups, API-key credentials, and capability), so what it reports is what
the service experiences.

```bash
sudo lucy doctor --listen          # every line should be ✔; speak when asked
sudo lucy say "Honk! Can you hear me?"              # Lucy's natural (cloud) voice
sudo lucy say --on-device "Honk! Can you hear me?"  # her backup voice on the Pi (same honk)
sudo lucy voices                   # other cloud voices, with links to listen to them
sudo systemctl stop lucy && sudo lucy run --keyboard   # Enter to talk, Enter to stop; Ctrl-C quits
sudo systemctl start lucy
sudo lucy bench                    # latency breakdown, measured on this Pi
```

With motors (`--motors`), check them before closing the plush. Each motor
moves on its own first, then `lucy say` shows the beak following her voice and
the body flapping on the honk:

```bash
sudo systemctl stop lucy
sudo lucy motors                   # beak: wide x3, a little x3; body: forward, then back
sudo lucy say "Honk! Did you know geese fly in a V to save energy?"
sudo systemctl start lucy
```

Then tune `[motors.motion]` in `/etc/lucy/config.toml` (section 6).

`lucy bench` prints each stage measured from the button release. If
`first_audio` is high, compare `stt` and `first_token` (network or Groq) with
the gap to `first_audio` (Lucy's voice: Speechify, or Piper on the Pi when the
cloud voice is unavailable).

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
Children's words and the model's replies are **not** logged unless
`turn.log_transcripts = true`, which adds `child said: …` and `cloud brain replied: …`
(or `local brain replied: …`) lines to each turn.

## 6. Tuning

All settings are in `/etc/lucy/config.toml` (documented inline).

| To get | Change |
|---|---|
| A different voice | `sudo lucy voices`, then `tts.cloud.voice_id` |
| Both voices equally loud | `tts.cloud.volume` (cloud) or `tts.volume` (on-device) |
| Softer honks, or the word instead of the goose call | `tts.honk_volume = 0.6`; `tts.honks = false` |
| Longer or shorter answers | A `system_prompt_file` with a different length rule (the built-in prompt asks for at most 40 words) |
| Faster first word from the on-device voice | `turn.first_clause_min_words = 4` (the first sentence may be split at a comma, which sounds less natural); a `-low` Piper voice |
| Longer battery life | Shorter `session.awake_s`; `power.standby_governor = "powersave"` |
| Slower, clearer speech for little ones | `tts.length_scale = 1.15` (on-device voice); pick a calmer cloud voice with `lucy voices` |
| Fewer "I didn't hear that" | Lower `turn.speech_threshold_dbfs` (e.g. `-55`), or raise mic gain in `alsamixer` |
| Answers that are less creative and more steady | `llm.params.temperature = 0.5` |
| More context for follow-ups | `llm.history_exchanges = 10` (slightly slower and costlier) |
| Faster offline answers | `sudo ollama pull gemma3:270m`, then `local.model = "gemma3:270m"` (quicker but simpler answers); `local.params.max_tokens = 60` |
| More accurate offline hearing | A `base.en` faster-whisper model in `local.stt_model` (about twice as slow as `tiny.en`) |
| A beak that opens wider, or snaps less | `motors.motion.beak_max_power` (up or down). If small syllables don't move it, raise `beak_min_power` until `lucy motors` just moves it |
| A beak that runs ahead of, or behind, her voice | `motors.motion.lead_s`: larger moves sooner (try 0.03–0.1) |
| A calmer or livelier body | `motors.motion.body_power`; `bob_every_s` (0 = flap on honks only) |
| A body motor that may only run forward | `motors.motion.body_two_way = false` |

## 7. Updating

```bash
sudo raspi-config nonint disable_overlayfs && sudo reboot   # only if --read-only was used
cd ~/lucy && git pull
sudo deploy/install.sh --tune            # re-runnable; keeps your config and key
sudo deploy/install.sh --read-only && sudo reboot           # only if it was read-only
```

Edited config files are never overwritten. If the template changed, the new
version is written next to yours as `config.toml.new` so you can compare them.
A config from before the cloud voice still works (missing settings take their
defaults), but it sets `first_clause_min_words = 4`. Change that to `0` so Lucy
speaks in whole sentences, which sounds more natural.
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
| Lucy sounds robotic | Her cloud voice is unavailable, so the on-device voice speaks: no Speechify key, the month's free characters are used up, or Speechify is blocked | `lucy doctor` ("cloud voice"); `journalctl -u lucy -g "cloud voice"`; usage at <https://platform.speechify.ai/usage> |
| Lucy stops mid-answer and changes the subject | Speechify's content screen refused a sentence | Working as intended. The journal shows `ContentRefused` |
| Offline: "Can a grown-up check the Wi-Fi?" | The offline brain is not running | `systemctl status ollama`; `ollama list` should show `gemma3:1b`; `sudo lucy doctor` |
| Journal: `local brain is not ready` | Ollama started after Lucy, or the model is missing | `sudo ollama pull gemma3:1b && sudo systemctl restart lucy` |
| Journal: `Lucy will keep still: her motors are not working` | No PWM overlay (needs a reboot after `--motors`), a line in use, or no permission | `sudo lucy doctor` ("motors") says which; `ls /sys/class/pwm` should show `pwmchip0` |
| `lucy motors`: "cannot claim GPIO … Device or resource busy" | The service holds the lines | `sudo systemctl stop lucy` first |
| A motor turns the wrong way | Its two wires are swapped | Swap them at the driver's AOUT/BOUT terminals |
| The beak barely moves, or only on loud words | Not enough power to beat its spring | Raise `motors.motion.beak_min_power`, or use a weaker spring |
| Motors do nothing, but `lucy motors` gives no error | Motor battery off or flat, no shared ground, or SLP not reaching the Pi | Check the battery switch, the GND link between Pi and driver, and `motors.sleep_line` |
| A motor stops for a moment in a long sentence | The `max_hold_s` safety rest | Working as intended: no motor runs more than 2 s without a pause |

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
sudo rm -f /etc/udev/rules.d/60-lucy-pwm.rules   # if --motors added it
sudo cp /boot/firmware/config.txt.lucy-backup /boot/firmware/config.txt   # if --tune/--i2s-amp/--motors was used
sudo systemctl daemon-reload && sudo reboot
```
