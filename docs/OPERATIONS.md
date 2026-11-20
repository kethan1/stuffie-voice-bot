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
| `--motors` | You wired the beak and body motors to an L293D (HARDWARE.md, "Motors"). Adds the hardware PWM overlay, lets the `gpio` group use PWM, and turns on `[motors]` |
| `--playback-card ID`, `--capture-card ID` | Auto-detection picked the wrong sound card (`aplay -l`, `arecord -l`). Written to `audio.speaker_card`/`audio.microphone_card` and `/etc/asound.conf`; when that card is unplugged, Lucy uses any other USB speaker (or mic) that is plugged in |
| `--voice en_US-hfc_female-medium` | You want a different on-device (backup) Piper voice |
| `--tune` | Always, for a finished toy: boot-time power savings, unused services off, journal in RAM |
| `--local-backup` | You want Lucy to keep answering when GroqCloud is unreachable (no Wi-Fi, school firewall, outage). Installs Ollama with Gemma 3 1B and an on-device speech model: about 2 GB of SD card space and a ~2.5 GB one-time download. Answers offline take several seconds. |
| `--read-only` | Last step, once everything works: protects the SD card from sudden power loss |

Wherever NetworkManager runs (Raspberry Pi OS Bookworm and later), the installer
also sets up setup mode (section 5), and ends by printing Lucy's setup card.

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

Then tune `[motors.motion]` in `/etc/lucy/config.toml` (section 7), or from a phone in
setup mode (section 5), which can try the numbers before saving them.

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

## 5. Setup mode: a phone instead of SSH

Once Lucy is installed, a grown-up can change her Wi-Fi and her settings from a
phone. Nobody needs SSH, a keyboard, or a monitor.

**Lucy's setup card.** The installer ends by printing a QR code for her own
setup Wi-Fi, with its name (`Lucy-Setup-` and four digits) and password. Print
the card from her settings page ("Her setup card"), or run
`sudo lucy setup-card` to see it again. Keep it inside her zipper, where only
grown-ups go: whoever has it can change her settings.

**Starting setup mode**, any of:

* Hold her wing while she wakes up: plug her in and keep squeezing until she
  says "Setup time!"
* While she is awake, knock and ask: tap her wing twice, quickly, then squeeze
  it and hold while you say "enter setup mode", and let go. She says "Setup
  time!" It takes both halves. The words without the taps are a question like
  any other, and the taps without the words are an ordinary turn, so children
  do not start it by accident. She has to hear the words, so this needs Wi-Fi
  (or her offline brain, section 4).
* Take her somewhere with no Wi-Fi she knows. About 90 seconds after she
  wakes up, she starts setup mode by herself.
* Over SSH: `sudo lucy setup`.

**Using it:**

1. Lucy says that her setup Wi-Fi is on, and its name.
2. Scan the card's QR code with the phone's camera to join it (or join
   `Lucy-Setup-…` by hand with the password on the card).
3. Her settings page opens by itself, as on hotel Wi-Fi. If it doesn't, open
   <http://10.42.0.1> in the browser. Keep the page open while you work; if
   the phone asks, choose to stay connected or use the network without internet.
4. On the page:
   * **Wi-Fi:** pick a network, then type its password (and a username, for
     school networks that ask for one). When you tap **Join**, the phone loses
     Lucy's Wi-Fi. Listen to her. If it worked, she says so, and setup is
     over. If not, she says why (a wrong password, a network she can't find
     from there, a network with no way to the internet), forgets it, and
     turns her setup Wi-Fi back on within a minute: join it again to retry.
     The networks she already knew are never touched by a failed attempt.
   * **Lucy** and **Motors:** her voice, her goose voice, loudness, honks, chimes, how long she
     stays quick to answer, the questions-per-hour limit, and the beak and
     body powers. **Hear her** and **Test motors** try the numbers on the
     page before you save them.
   * **Conversations:** how many conversations Lucy has written down, and
     **Read them**: what was said to her and what she answered, the latest
     first, ten conversations to a page. Turns where a safety screen stepped
     in are marked in red with the reason (for example "may need a grown-up
     (being-hurt)"), and **Where a safety screen stepped in** shows only those
     conversations. **Delete every conversation** removes them for good. The
     two settings beside the link turn the writing off, and set how many days
     conversations are kept (30; 0 keeps them until you delete them).
   * **API keys:** paste a new Groq or Speechify key. The page never shows a
     saved key, only its last four characters.
   * **Advanced:** the whole `config.toml`, with its notes. Lucy refuses
     anything she couldn't start with, and **Undo** puts back the file as it
     was before the last change.
5. Tap **Finish setup**, hold her wing for two seconds, or just leave her: after
   10 minutes with nothing happening she finishes by herself. She wakes up
   with her new settings.

| What changes | Where it is kept |
|---|---|
| Wi-Fi networks | NetworkManager: `/etc/NetworkManager/system-connections/` |
| Settings | `/etc/lucy/config.toml` (the version before the last change: `config.toml.previous`) |
| API keys | `/etc/lucy/credentials/` |
| Conversations (her logbook) | `/var/lib/lucy/logbook.db`, written by Lucy as she talks; setup mode only reads and deletes it |

With a read-only root (`--read-only`), these changes would otherwise be lost at
power-off, so setup mode copies them to the SD card when it ends
(`/usr/local/sbin/lucy-persist`). That includes her logbook, and it is the only
time the logbook reaches the card: with a read-only root, conversations since
setup mode last ended are lost if she is unplugged. To keep them, open setup
mode (and tap **Finish setup**) before unplugging her, or leave the root
writable.

**Her logbook.** Lucy writes down every turn in which she hears words, also the
ones a safety screen answers for her, so that a child who tells her something
serious has told a grown-up too. Read it regularly. It holds what children
said, by first name if they gave one, so:

* tell the families and the school that she keeps it, as you would for any
  other record of children;
* keep her setup card, and her SD card, where only grown-ups go: either one
  opens the logbook, which is not encrypted;
* delete it from the page when it has served its purpose, or shorten
  `logbook.keep_days`.

No audio is ever kept, the logbook never leaves the Pi, and Lucy never reads it
back: what one child said is never repeated to another.

**School networks.** Lucy can log in to networks that ask for a username and
password (WPA2-Enterprise with PEAP, the usual kind); she does not check the
network's certificate. If the school only lets in approved devices, the page
shows her Wi-Fi (MAC) address to give to its IT team: she always uses the Pi's
own address, never a random one. A network that needs a sign-in web page (many
guest networks) cannot work: she joins it but cannot reach the internet, and
says so.

**Settings** (`[setup]` in `/etc/lucy/config.toml`):

| Setting | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Setup mode at all |
| `hold_to_start_s` | `3.0` | How long the wing must stay held once she is awake; `0` turns the gesture off |
| `voice_command` | `true` | Two taps, then holding her wing and saying "enter setup mode", starts it while she is awake (also on her settings page) |
| `when_offline` | `true` | Start by herself when no network connects after boot. Turn it off if she runs offline on purpose (`--local-backup` with no Wi-Fi) |
| `offline_wait_s` | `90.0` | How long after boot she waits for a network |
| `idle_timeout_s` | `600.0` | How long setup mode waits with nothing happening |

On a laptop, `make setup` serves the same page at <http://127.0.0.1:8080> with
pretend Wi-Fi: it saves to `setup-dev.toml`, a copy of `dev.toml`, and a
password of `wrong-password` pretends to fail.

## 6. Day-to-day

| Task | Command |
|---|---|
| Read what was said | Setup mode, **Conversations** (section 5). Over SSH: `sudo sqlite3 /var/lib/lucy/logbook.db "SELECT datetime(at,'unixepoch','localtime'), outcome, concern, heard, said FROM turns ORDER BY id DESC LIMIT 20"` (after `sudo apt-get install sqlite3`) |
| Live logs | `journalctl -u lucy -f` (and `journalctl -u ollama -f` for the offline brain) |
| Status | `systemctl status lucy` |
| Restart after a config change | `sudo systemctl restart lucy` |
| Setup mode | `sudo lucy setup`; its log: `journalctl -u lucy-setup -f` |
| Temperature / throttling | `vcgencmd measure_temp; vcgencmd get_throttled` |

Every turn logs one line such as
`turn complete (cloud brain): stt=0.41s first_token=0.68s first_synth=0.93s first_audio=0.93s total=7.12s`.
Offline turns say `(local brain)` and include `fallback=…`.
Children's words and the model's replies are **not** in this journal unless
`turn.log_transcripts = true`, which adds `child said: …` and `cloud brain replied: …`
(or `local brain replied: …`) lines to each turn. They are kept in her logbook
instead (section 5), which only root and Lucy can read.
In a longer conversation, every third turn or so is followed by
`folded 3 exchange(s) into the notes (23 words) in 0.19s`: Lucy's memory of what
was said earlier (ARCHITECTURE.md §5.9). The notes themselves are logged
(`Lucy's notes: …`) only with `turn.log_transcripts = true`.

## 7. Tuning

All settings are in `/etc/lucy/config.toml` (documented inline).

| To get | Change |
|---|---|
| A different voice | `sudo lucy voices`, then `tts.cloud.voice_id` |
| Both voices equally loud | `tts.cloud.volume` (cloud) or `tts.volume` (on-device) |
| Softer honks, or the word instead of the goose call | `tts.honk_volume = 0.6`; `tts.honks = false` |
| More or less of a goose voice | `tts.goose_voice`: 0 is her plain voice, 1 the most goose-like (default 0.7: higher, honky, and raspy, with every word still clear). In setup mode, **Hear her** tries a value before you save it |
| More or fewer honks | `turn.opening_honk_chance` (0 to 1): how often she starts a reply with a honk of her own. The prompt asks for honks in most answers too |
| Louder or softer chimes | `session.earcon_volume` (1 is a little louder than her voice) |
| Children may not change her volume, or not as much | `volume.commands = false`; or a smaller range with `volume.max_db` and `volume.min_db` |
| Louder or softer as usual | `volume.speaker_db`: the speaker's own volume (dB, as `alsamixer` shows) at her usual loudness. Lucy sets it each time she opens her speaker, so changes made in `alsamixer` do not last. "Louder" raises it up to `volume.speaker_max_db`; volume commands do not amplify audio in software. A card without a mixer cannot change volume this way |
| Longer or shorter answers | A `system_prompt_file` with a different length rule (the built-in prompt asks for at most 40 words) |
| Faster first word from the on-device voice | `turn.first_clause_min_words = 4` (the first sentence may be split at a comma, which sounds less natural); a `-low` Piper voice |
| Longer battery life | Shorter `session.awake_s`; `power.standby_governor = "powersave"` |
| Slower, clearer speech for little ones | `tts.length_scale = 1.15` (on-device voice); pick a calmer cloud voice with `lucy voices` |
| Fewer "I didn't hear that" | Lower `turn.speech_threshold_dbfs` (e.g. `-55`), or raise mic gain in `alsamixer` |
| Answers that are less creative and more steady | `llm.params.temperature = 0.5` |
| A conversation remembered for longer after it goes quiet | `session.forget_after_s` (300: she forgets everything five minutes after the last answer) |
| More of a conversation word for word | `llm.history_exchanges = 10` (each exchange adds 40 to 80 tokens to every request; the older half still becomes notes when it is full) |
| Longer or shorter notes on the earlier parts | `memory.max_words` (60) |
| Conversations kept for longer, or not as long | `logbook.keep_days` (30; 0 keeps them until you delete them), or **Conversations** in setup mode |
| No record of what is said to her | `logbook.enabled = false`, or untick **Write down every conversation** in setup mode. Then delete what is already there on the Conversations page |
| No notes, only the last few exchanges | `memory.notes = false`, or untick **Remember more of each conversation** in setup mode |
| Faster offline answers | `sudo ollama pull gemma3:270m`, then `local.model = "gemma3:270m"` (quicker but simpler answers); `local.params.max_tokens = 60` |
| More accurate offline hearing | A `base.en` faster-whisper model in `local.stt_model` (about twice as slow as `tiny.en`) |
| A beak that opens wider, or snaps less | `motors.motion.beak_max_power` (up or down). If small syllables don't move it, raise `beak_min_power` until `lucy motors` just moves it |
| A beak that runs ahead of, or behind, her voice | `motors.motion.lead_s`: larger moves sooner (try 0.03–0.1) |
| A calmer or livelier body | `motors.motion.body_power`; `bob_every_s` (0 = flap on honks only) |
| A body motor that may only run forward | `motors.motion.body_two_way = false` |

## 8. Updating

```bash
sudo raspi-config nonint disable_overlayfs && sudo reboot   # only if --read-only was used
cd ~/lucy && git pull
sudo deploy/install.sh --tune            # re-runnable; keeps your config and key
sudo deploy/install.sh --read-only && sudo reboot           # only if it was read-only
```

Edited config files are never overwritten. If the template changed, the new
version is written next to yours as `config.toml.new` so you can compare them.
Settings your file does not have take their defaults. One of those is new and
worth knowing about: from this version Lucy **writes down every conversation**
in her logbook (section 5) unless your file says `[logbook]` `enabled = false`.
A config from before the cloud voice still works (missing settings take their
defaults), but it sets `first_clause_min_words = 4`. Change that to `0` so Lucy
speaks in whole sentences, which sounds more natural.
If your config enables the offline brain, re-running the installer keeps it
installed and up to date without `--local-backup`.

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Silent; LED does nothing | Service not running | `systemctl status lucy`, `journalctl -u lucy -e` |
| "Can a grown-up check the Wi-Fi?" | No network or DNS | `ping -c3 api.groq.com`; check Wi-Fi with `nmcli` |
| "Something inside me needs a grown-up's help" | Missing or invalid API key, or an audio device error | `lucy doctor` names which one |
| "My goose brain needs a little rest" | Groq rate limit (both models) or the hourly turn budget | Wait a minute. Check `session.max_turns_per_hour` |
| Often "I didn't hear that" | Mic too quiet | `lucy doctor --listen`, then `alsamixer` (F4 for capture) |
| Lucy hears herself | Mic too close to the speaker | Move them apart. Lucy only listens while the wing is held |
| Crackles or gaps | CPU starved or weak power | `vcgencmd get_throttled` (non-zero = under-voltage); raise `audio.output_latency_s` to `0.12` |
| Goes silent in the middle of talking | The speaker came unplugged (a USB cable working loose) | Plug it back in and squeeze her wing: she opens it again for her next sound, with no restart. `journalctl -u lucy -g speaker` shows when it stopped; `journalctl -k -g 'USB disconnect'` shows a cable that keeps dropping out. Two USB devices that drop out together, or `device descriptor read/64, error -32` again and again, point to a hub without enough power |
| Silent after swapping in a different speaker | Usually nothing: she uses any USB speaker in place of the configured one. `sudo lucy doctor` names the card she uses. Lucy sets its own volume to `volume.speaker_db` when she opens it, so it should be as loud as the old one if both report decibels; `lucy doctor` shows the level | To make the new speaker the configured one, re-run the installer with `--playback-card <id>` |
| Button does nothing | Wiring or pin number | `lucy doctor`; `gpioinfo` shows `lucy-button` on line 17 |
| `CPU governor ... not writable` in doctor | Service never started since boot | Start it once; its `ExecStartPre` grants access |
| Lucy forgets a name she was told a few minutes ago, mid-conversation | Her notes are not being written (the note model is rate-limited or no longer offered), or they were screened | `journalctl -u lucy -g notes`; `sudo lucy doctor` ("AI service") names a missing model: set `memory.model` to one the provider offers |
| The Conversations page says nothing is written down | Writing is turned off, nobody has talked to her since it was turned on, or the file cannot be written | `sudo lucy doctor` ("logbook") counts what is there, or says why it cannot be written; `journalctl -u lucy -g logbook` |
| Conversations are gone after unplugging her | With `--read-only`, the logbook is in RAM until setup mode ends | Open setup mode and tap **Finish setup** before unplugging her (section 5) |
| The times on the Conversations page are wrong | The Pi has no clock battery: until she has been online, her clock carries on from when she was last shut down | They are right for turns made after she connects. Set the time zone with `sudo raspi-config nonint do_change_timezone Europe/London` (yours) |
| Lucy has forgotten everything after a pause | More than `session.forget_after_s` (five minutes) of quiet | Working as intended, so the next child starts fresh. Raise it for longer memories |
| Lucy "thinks on her own" though the Wi-Fi works | Groq is blocked (firewall, DNS) | `curl -sI https://api.groq.com`; the next press re-checks automatically |
| Lucy is much louder or quieter than usual | A child asked her to ("louder", "too loud") | Ask her ("quieter"), or `sudo systemctl restart lucy`: she always wakes up at her usual loudness |
| Lucy sounds robotic | Her cloud voice is unavailable, so the on-device voice speaks: no Speechify key, the month's free characters are used up, or Speechify is blocked | `lucy doctor` ("cloud voice"); `journalctl -u lucy -g "cloud voice"`; usage at <https://platform.speechify.ai/usage> |
| Lucy stops mid-answer and changes the subject | Speechify's content screen refused a sentence | Working as intended. The journal shows `ContentRefused` |
| Offline: "Can a grown-up check the Wi-Fi?" | The offline brain is not running | `systemctl status ollama`; `ollama list` should show `gemma3:1b`; `sudo lucy doctor` |
| Journal: `local brain is not ready` | Ollama started after Lucy, or the model is missing | `sudo ollama pull gemma3:1b && sudo systemctl restart lucy` |
| Journal: `Lucy will keep still: her motors are not working` | No PWM overlay (needs a reboot after `--motors`), a line in use, or no permission | `sudo lucy doctor` ("motors") says which; `ls /sys/class/pwm` should show `pwmchip0` |
| `lucy motors`: "cannot claim GPIO … Device or resource busy" | The service holds the lines | `sudo systemctl stop lucy` first |
| A motor turns the wrong way | Its two wires are swapped | Swap them at the L293D's outputs (1Y/2Y for the beak, 3Y/4Y for the body) |
| The beak barely moves, or only on loud words | Not enough power to beat its spring (the L293D loses about 2 V) | Raise `motors.motion.beak_min_power`, charge the motor battery, or use a weaker spring |
| Motors do nothing, but `lucy motors` gives no error | Motor battery off or flat, no shared ground, or no logic supply | Check the battery switch, the GND link between the Pi and the L293D, and that its VCC1 (pin 16) has the Pi's 5 V |
| A motor stops for a moment in a long sentence | The `max_hold_s` safety rest | Working as intended: no motor runs more than 2 s without a pause |
| "Setup time!", then "I couldn't turn on my setup Wi-Fi" | NetworkManager could not start the hotspot. Most often the Wi-Fi country is not set | `journalctl -u lucy-setup -e`; set the country with `sudo raspi-config nonint do_wifi_country GB` (your country's code) |
| The phone joined her setup Wi-Fi, but no page opened | The phone skipped its sign-in check | Open <http://10.42.0.1> in the browser |
| Holding her wing at wake-up does nothing | Setup mode is not installed (no NetworkManager), or `setup.hold_to_start_s = 0` | `systemctl status lucy-setup.path`; re-run the installer |
| Tap, tap, hold, "enter setup mode" gets an ordinary answer | She did not take it for the knock (each tap under 0.3 s, and no more than 0.7 s between them and before the squeeze), she heard other words, or `setup.voice_command = false` | `journalctl -u lucy -g knock` shows each knock she noticed. Tap quicker, or raise `session.tap_gap_s`. `turn.log_transcripts = true` shows what she heard |
| "I can't start my setup mode right now" | Setup mode is not installed (no NetworkManager), or Lucy is not running as her service | `systemctl status lucy-setup.path`; re-run the installer |
| She starts setup mode at every boot | No Wi-Fi she knows is in range | Add one in setup mode, or set `setup.when_offline = false` |
| Changes from setup mode are gone after unplugging her | With `--read-only`, copying them to the SD card failed | `journalctl -u lucy-setup -g lucy-persist` |

## 10. Uninstall

```bash
sudo systemctl disable --now lucy ollama
sudo systemctl disable --now lucy-setup.path lucy-setup-offline.service
sudo nmcli connection delete "Lucy setup hotspot"
sudo rm -rf /var/lib/lucy   # her logbook: every conversation she wrote down
sudo rm -rf /opt/lucy /etc/lucy /var/cache/lucy /usr/local/bin/lucy /etc/systemd/system/lucy.service \
  /etc/systemd/journald.conf.d/lucy.conf /etc/systemd/system/lucy-setup.service \
  /etc/systemd/system/lucy-setup.path /etc/systemd/system/lucy-setup-offline.service \
  /etc/NetworkManager/dnsmasq-shared.d/lucy-setup.conf /usr/local/sbin/lucy-persist
# Only if --local-backup was used: Ollama, its libraries, and its models
sudo rm -rf /usr/local/bin/ollama /usr/local/lib/ollama /etc/systemd/system/ollama.service \
  /var/lib/private/ollama /var/lib/ollama
sudo userdel lucy
sudo mv /etc/asound.conf.lucy-backup /etc/asound.conf 2>/dev/null || sudo rm -f /etc/asound.conf
sudo rm -f /etc/udev/rules.d/60-lucy-pwm.rules   # if --motors added it
sudo cp /boot/firmware/config.txt.lucy-backup /boot/firmware/config.txt   # if --tune/--i2s-amp/--motors was used
sudo systemctl daemon-reload && sudo reboot
```
