#!/usr/bin/env bash
# Install (or upgrade) Lucy the Goose on Raspberry Pi OS Lite (64-bit, Bookworm or Trixie).
#
#   sudo deploy/install.sh [options]
#
#   --i2s-amp              add the MAX98357A I2S amplifier overlay (reboot required)
#   --motors               Lucy's beak and body motors on a DRV8833: the hardware PWM overlay
#                          (reboot required), PWM access for the gpio group, [motors] enabled
#   --playback-card NAME   ALSA card id for the speaker   (default: auto-detect)
#   --capture-card NAME    ALSA card id for the microphone (default: auto-detect)
#   --voice NAME           on-device Piper voice, e.g. en_US-lessac-medium (default)
#   --tune                 power tuning: boot config, unused services off, journal in RAM
#   --local-backup         an offline brain on the Pi (faster-whisper + Ollama with Gemma 3 1B),
#                          used when GroqCloud is unreachable; needs ~2 GB of SD card space
#   --read-only            make the root filesystem read-only (overlayfs); do this last
#   --no-start             install everything but do not (re)start the service
#
# Safe to re-run: every step checks what is already there. Existing config files are
# never overwritten; an updated template is written next to them as *.new instead.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX=/opt/lucy
ETC=/etc/lucy
SERVICE_USER=lucy
VOICE=en_US-lessac-medium
VOICE_BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main
# Checksums of the default voice, so a corrupted or tampered download is refused.
declare -A VOICE_SHA256=(
  [en_US-lessac-medium.onnx]=5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f
  [en_US-lessac-medium.onnx.json]=efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0
)
# A pinned uv release, checksum-verified like the voice. It installs the hash-locked uv.lock.
UV_VERSION=0.12.8
declare -A UV_SHA256=(
  [aarch64]=ba8661f4fd207c8e94814191598e619b355ac10d5014e851e21eb800f9ef2b00
  [x86_64]=2e2b37e9811e17675a9e70bed5e1a58fc8c0388be63d751d72cc735188c149ff
)
UV="$PREFIX/bin/uv"
# The offline brain. Ollama is pinned and checksum-verified (aarch64 build); only its CPU
# libraries are unpacked, skipping ~1.5 GB of CUDA libraries a Pi cannot use.
OLLAMA_VERSION=v0.40.7
OLLAMA_SHA256=c74249e87795b1437ce0f484adfd05b4628d7497472c475b13ec6db69ecc69cf
LOCAL_LLM=gemma3:1b
STT_MODEL=faster-whisper-tiny.en
STT_MODEL_URL=https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/0d3d19a32d3338f10357c0889762bd8d64bbdeba
declare -A STT_MODEL_SHA256=(
  [config.json]=14b1b421a90349bc551b881461426b561a874049cb9e4c4864f2ca384f6a7cc5
  [model.bin]=1a5afae06a4db91c975c9a9d78be5cc110ee4ea022ad57d55492e4550e936b2a
  [tokenizer.json]=929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df
  [vocabulary.txt]=ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf
)
I2S_AMP=0 MOTORS=0 TUNE=0 READ_ONLY=0 START=1 LOCAL_BACKUP=0 PLAYBACK_CARD="" CAPTURE_CARD=""

log() { printf '\033[1;33m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;31m!!\033[0m %s\n' "$*" >&2; }
die() { warn "$*"; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --i2s-amp) I2S_AMP=1 ;;
    --motors) MOTORS=1 ;;
    --playback-card) PLAYBACK_CARD="${2:?}"; shift ;;
    --capture-card) CAPTURE_CARD="${2:?}"; shift ;;
    --voice) VOICE="${2:?}"; shift ;;
    --tune) TUNE=1 ;;
    --local-backup) LOCAL_BACKUP=1 ;;
    --read-only) READ_ONLY=1 ;;
    --no-start) START=0 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
  shift
done

[[ $EUID -eq 0 ]] || die "run as root: sudo $0 $*"
[[ "$(uname -m)" == aarch64 ]] || warn "expected a 64-bit Raspberry Pi OS (aarch64); continuing anyway"
BOOT_CONFIG=/boot/firmware/config.txt
[[ -f $BOOT_CONFIG ]] || BOOT_CONFIG=/boot/config.txt

# Install $1 to $2 unless an edited copy is already there (then write $2.new).
install_config() {
  local src=$1 dest=$2 mode=${3:-0644}
  if [[ ! -e $dest ]]; then
    install -D -m "$mode" "$src" "$dest"
  elif ! cmp -s "$src" "$dest"; then
    install -m "$mode" "$src" "$dest.new"
    log "kept your $dest; the latest template is at $dest.new"
  fi
}

# Replace (or append) a "# >>> name >>>" ... "# <<< name <<<" block in a file.
set_block() {
  local file=$1 name=$2 content=$3
  [[ -f $file.lucy-backup ]] || cp -a "$file" "$file.lucy-backup"
  sed -i "/^# >>> $name >>>/,/^# <<< $name <<</d" "$file"
  printf '%s\n' "$content" >>"$file"
}

# Download $2 to $1 unless present, then check it against the sha256 $3 (removed on mismatch).
fetch_verified() {
  local dest=$1 url=$2 sha=$3
  if [[ ! -s $dest ]]; then
    curl -fsSL --retry 3 -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
  fi
  echo "$sha  $dest" | sha256sum --check --quiet ||
    { rm -f "$dest"; die "checksum mismatch for $dest (download removed; re-run to retry)"; }
}

# Stream Ollama's release archive straight into place: hashed on the way in, CPU parts only,
# and nothing installed unless the checksum matches.
install_ollama() {
  if [[ "$(/usr/local/bin/ollama --version 2>&1)" == *"${OLLAMA_VERSION#v}"* ]]; then return; fi
  [[ $(uname -m) == aarch64 ]] || die "the pinned Ollama build is for 64-bit Raspberry Pi OS (aarch64)"
  local tmp; tmp=$(mktemp -d)
  mkfifo "$tmp/archive"
  sha256sum <"$tmp/archive" >"$tmp/sum" &
  local hasher=$!
  mkdir "$tmp/root"
  curl -fsSL --retry 3 \
    "https://github.com/ollama/ollama/releases/download/$OLLAMA_VERSION/ollama-linux-arm64.tar.zst" |
    tee "$tmp/archive" | zstd -dcq |
    tar -x -C "$tmp/root" --exclude='lib/ollama/cuda_*' --exclude='lib/ollama/rocm*' \
      --exclude='lib/ollama/vulkan*'
  wait "$hasher"
  if [[ "$(cut -d' ' -f1 "$tmp/sum")" != "$OLLAMA_SHA256" ]]; then
    rm -rf "$tmp"
    die "checksum mismatch for Ollama $OLLAMA_VERSION (nothing was installed)"
  fi
  install -m 0755 "$tmp/root/bin/ollama" /usr/local/bin/ollama
  rm -rf /usr/local/lib/ollama
  mv "$tmp/root/lib/ollama" /usr/local/lib/ollama
  rm -rf "$tmp"
}

# True if the installed config turns on the [$1] section (so upgrades keep it on).
section_enabled() {
  [[ -f $ETC/config.toml ]] && python3 -c '
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    sys.exit(0 if tomllib.load(f).get(sys.argv[2], {}).get("enabled") is True else 1)
' "$ETC/config.toml" "$1" 2>/dev/null
}

# Turn on the [$1] section of the installed config.
enable_section() {
  local name=$1
  if grep -q "^\[$name\]" "$ETC/config.toml"; then
    sed -i "/^\[$name\]/,/^\[/ s/^enabled = false/enabled = true/" "$ETC/config.toml"
  else
    printf '\n[%s]\nenabled = true\n' "$name" >>"$ETC/config.toml"
  fi
  section_enabled "$name" || warn "could not enable [$name] in $ETC/config.toml; set enabled = true there"
}

# The id of an ALSA card with a PCM of the given kind (p = playback, c = capture).
# USB devices win over the onboard outputs; HDMI audio is never chosen.
detect_card() {
  local kind=$1 dir id fallback=""
  for dir in /proc/asound/card[0-9]*; do
    if [[ ! -e $dir/id ]] || ! compgen -G "$dir/pcm*$kind" >/dev/null; then continue; fi
    id=$(<"$dir/id")
    [[ $id == vc4hdmi* ]] && continue
    if [[ -e $dir/usbid ]]; then echo "$id"; return; fi
    fallback=${fallback:-$id}
  done
  echo "$fallback"
}

if [[ $LOCAL_BACKUP -eq 0 ]] && section_enabled local; then
  LOCAL_BACKUP=1
  log "keeping the offline brain that $ETC/config.toml enables"
fi
if [[ $MOTORS -eq 0 ]] && section_enabled motors; then
  MOTORS=1
  log "keeping the motors that $ETC/config.toml enables"
fi
EXTRAS=(--extra pi)
[[ $LOCAL_BACKUP -eq 0 ]] || EXTRAS+=(--extra local)

log "installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  python3 libportaudio2 alsa-utils iw ca-certificates curl rsync zstd >/dev/null

log "creating the '$SERVICE_USER' service account"
getent group gpio >/dev/null || groupadd --system gpio
if [[ ! -e /etc/udev/rules.d/60-lucy-gpio.rules ]] && ! grep -rqs 'gpiochip' /usr/lib/udev/rules.d /etc/udev/rules.d; then
  echo 'SUBSYSTEM=="gpio", KERNEL=="gpiochip*", GROUP="gpio", MODE="0660"' >/etc/udev/rules.d/60-lucy-gpio.rules
  udevadm control --reload && udevadm trigger --subsystem-match=gpio
fi
id -u "$SERVICE_USER" &>/dev/null ||
  useradd --system --home-dir "$PREFIX" --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
usermod -a -G audio,gpio "$SERVICE_USER"

log "installing uv $UV_VERSION"
install -d -m 0755 "$PREFIX" "$PREFIX/bin" "$PREFIX/voices"
if [[ "$("$UV" --version 2>/dev/null)" != "uv $UV_VERSION"* ]]; then
  arch=$(uname -m)
  [[ -n ${UV_SHA256[$arch]:-} ]] || die "no pinned uv build for $arch"
  tarball=$(mktemp)
  curl -fsSL --retry 3 -o "$tarball" \
    "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$arch-unknown-linux-gnu.tar.gz"
  echo "${UV_SHA256[$arch]}  $tarball" | sha256sum --check --quiet ||
    { rm -f "$tarball"; die "checksum mismatch for uv (download removed; re-run to retry)"; }
  tar -xzf "$tarball" -C "$PREFIX/bin" --strip-components=1 "uv-$arch-unknown-linux-gnu/uv"
  rm -f "$tarball"
fi

log "installing Lucy into $PREFIX"
rsync -a --delete --exclude '.venv' --exclude '.git' --exclude '__pycache__' \
  --exclude '*.onnx' --exclude '*.onnx.json' "$REPO_DIR/" "$PREFIX/src/"
# Exactly what uv.lock pins, hash-verified, on the system Python. --no-build refuses any
# dependency without a prebuilt wheel, so nothing compiles on the Pi. Bytecode is compiled
# now because the sandboxed service cannot write it.
UV_CACHE_DIR=/var/cache/lucy/uv UV_PROJECT_ENVIRONMENT="$PREFIX/venv" UV_PYTHON_DOWNLOADS=never \
  "$UV" sync --quiet --project "$PREFIX/src" --python /usr/bin/python3 --locked --no-dev \
  "${EXTRAS[@]}" --no-editable --no-build --compile-bytecode
UV_CACHE_DIR=/var/cache/lucy/uv "$UV" cache prune --quiet

log "fetching the Piper voice $VOICE"
IFS=- read -r lang name quality <<<"$VOICE"
[[ -n ${quality:-} ]] || die "voice must look like en_US-lessac-medium"
for file in "$VOICE.onnx" "$VOICE.onnx.json"; do
  dest="$PREFIX/voices/$file"
  url="$VOICE_BASE/${lang%%_*}/$lang/$name/$quality/$file"
  if [[ -n ${VOICE_SHA256[$file]:-} ]]; then
    fetch_verified "$dest" "$url" "${VOICE_SHA256[$file]}"
  else
    [[ -s $dest ]] || { curl -fsSL --retry 3 -o "$dest.part" "$url" && mv "$dest.part" "$dest"; }
    warn "no pinned checksum for $file; trusting the download"
  fi
done
chmod -R a+rX "$PREFIX"
install -m 0755 "$REPO_DIR/deploy/bin/lucy" /usr/local/bin/lucy

log "installing configuration into $ETC"
install -d -m 0755 "$ETC"
install_config "$REPO_DIR/deploy/config.toml" "$ETC/config.toml"
sed -i "s|^voice = \"/opt/lucy/voices/.*\"|voice = \"$PREFIX/voices/$VOICE.onnx\"|" "$ETC/config.toml"
install -d -m 0700 "$ETC/credentials"
KEY_FILE="$ETC/credentials/groq_api_key"
if [[ ! -s $KEY_FILE ]]; then
  if [[ -t 0 ]]; then
    read -r -s -p "Paste the Groq API key (input hidden; Enter to skip): " key; echo
    if [[ -n $key ]]; then (umask 077; printf '%s' "$key" >"$KEY_FILE"); fi
  fi
  [[ -s $KEY_FILE ]] || warn "no API key yet: put it in $KEY_FILE (root-only, mode 600)"
fi
[[ ! -e $KEY_FILE ]] || chmod 0600 "$KEY_FILE"
# Optional: Lucy's natural cloud voice. The file always exists, because the service loads it as a
# credential; while it is empty, Lucy speaks with her on-device voice.
VOICE_KEY_FILE="$ETC/credentials/speechify_api_key"
if [[ ! -s $VOICE_KEY_FILE ]]; then
  if [[ -t 0 ]]; then
    read -r -s -p "Paste the Speechify API key for Lucy's natural voice (hidden; Enter to skip): " key; echo
    if [[ -n $key ]]; then (umask 077; printf '%s' "$key" >"$VOICE_KEY_FILE"); fi
  fi
  [[ -e $VOICE_KEY_FILE ]] || (umask 077; : >"$VOICE_KEY_FILE")
  [[ -s $VOICE_KEY_FILE ]] || warn "no Speechify key: Lucy uses her on-device voice until it is in $VOICE_KEY_FILE"
fi
chmod 0600 "$VOICE_KEY_FILE"

if [[ $LOCAL_BACKUP -eq 1 ]]; then
  log "installing the offline brain: speech model $STT_MODEL"
  install -d -m 0755 "$PREFIX/models/$STT_MODEL"
  for file in "${!STT_MODEL_SHA256[@]}"; do
    fetch_verified "$PREFIX/models/$STT_MODEL/$file" "$STT_MODEL_URL/$file" "${STT_MODEL_SHA256[$file]}"
  done
  chmod -R a+rX "$PREFIX/models"

  log "installing the offline brain: Ollama $OLLAMA_VERSION and $LOCAL_LLM"
  install_ollama
  install -m 0644 "$REPO_DIR/deploy/systemd/ollama.service" /etc/systemd/system/ollama.service
  systemctl daemon-reload
  systemctl enable --now ollama.service >/dev/null
  systemctl restart ollama.service
  for _ in $(seq 30); do ollama list &>/dev/null && break; sleep 1; done
  ollama pull "$LOCAL_LLM"
  enable_section local
fi

log "configuring audio"
if [[ $I2S_AMP -eq 1 ]]; then
  # SD (shutdown) on GPIO4, the overlay's default: the kernel powers the amp only while audio plays.
  set_block "$BOOT_CONFIG" lucy-audio "$(printf '%s\n' '# >>> lucy-audio >>>' '[all]' \
    'dtoverlay=max98357a' '# <<< lucy-audio <<<')"
  PLAYBACK_CARD=${PLAYBACK_CARD:-MAX98357A}
fi
PLAYBACK_CARD=${PLAYBACK_CARD:-$(detect_card p)}
CAPTURE_CARD=${CAPTURE_CARD:-$(detect_card c)}
if [[ -n $PLAYBACK_CARD && -n $CAPTURE_CARD ]]; then
  [[ ! -e /etc/asound.conf || -e /etc/asound.conf.lucy-backup ]] || cp -a /etc/asound.conf /etc/asound.conf.lucy-backup
  sed -e "s/@PLAYBACK_CARD@/$PLAYBACK_CARD/g" -e "s/@CAPTURE_CARD@/$CAPTURE_CARD/g" \
    "$REPO_DIR/deploy/alsa/asound.conf" >/etc/asound.conf
  log "speaker card: $PLAYBACK_CARD, microphone card: $CAPTURE_CARD (change with --playback-card/--capture-card)"
else
  warn "could not find a speaker and a microphone (see 'aplay -l' and 'arecord -l'); re-run once they are plugged in"
fi

if [[ $MOTORS -eq 1 ]]; then
  log "setting up the beak and body motors"
  # Hardware PWM channel 0 on GPIO12 (the beak) and channel 1 on GPIO13 (the body).
  set_block "$BOOT_CONFIG" lucy-motors "$(printf '%s\n' '# >>> lucy-motors >>>' '[all]' \
    'dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4' '# <<< lucy-motors <<<')"
  # Raspberry Pi OS already lets the gpio group use PWM; add the same rule anywhere it does not.
  if ! grep -rqs 'SUBSYSTEM=="pwm' /usr/lib/udev/rules.d /etc/udev/rules.d; then
    cat >/etc/udev/rules.d/60-lucy-pwm.rules <<'RULE'
# Let the gpio group export and drive hardware PWM channels (Lucy's motors). Exporting a
# channel sends a "change" event for its chip, so new channels are covered too.
SUBSYSTEM=="pwm", KERNEL=="pwmchip*", RUN+="/bin/sh -c 'chgrp -R gpio /sys%p && chmod -R g+rwX /sys%p'"
RULE
    udevadm control --reload
  fi
  enable_section motors
fi

if [[ $TUNE -eq 1 ]]; then
  log "applying power tuning"
  block="$(cat "$REPO_DIR/deploy/boot/config.txt.lucy")"
  [[ $PLAYBACK_CARD != Headphones ]] || block="$(grep -v '^dtparam=audio=off' <<<"$block")"
  set_block "$BOOT_CONFIG" lucy "$block"
  for unit in bluetooth.service hciuart.service triggerhappy.service triggerhappy.socket ModemManager.service; do
    systemctl disable --now "$unit" &>/dev/null || true
  done
  install -D -m 0644 "$REPO_DIR/deploy/journald/lucy.conf" /etc/systemd/journald.conf.d/lucy.conf
  systemctl restart systemd-journald
fi

log "installing the systemd service"
install -m 0644 "$REPO_DIR/deploy/systemd/lucy.service" /etc/systemd/system/lucy.service
systemctl daemon-reload
systemctl enable lucy.service >/dev/null
if [[ $START -eq 1 ]]; then
  systemctl restart lucy.service || warn "lucy did not start; see: journalctl -u lucy -e"
fi

if [[ $READ_ONLY -eq 1 ]]; then
  log "enabling the read-only root filesystem (overlayfs)"
  command -v raspi-config >/dev/null || die "raspi-config is required for --read-only"
  raspi-config nonint enable_overlayfs
  raspi-config nonint enable_bootro || true
  warn "from the next boot, changes to the SD card are discarded at power-off."
  warn "to update later: sudo raspi-config nonint disable_overlayfs && sudo reboot"
fi

log "done. Next steps:"
echo "   sudo lucy doctor --listen     # check everything (runs exactly as the service does)"
echo "   sudo lucy say 'Honk!'          # test the speaker (and, with motors, the beak)"
if [[ $MOTORS -eq 1 ]]; then
  echo "   sudo systemctl stop lucy && sudo lucy motors; sudo systemctl start lucy   # check the motors"
fi
echo "   journalctl -u lucy -f"
if [[ $I2S_AMP -eq 1 || $MOTORS -eq 1 || $TUNE -eq 1 || $READ_ONLY -eq 1 ]]; then
  warn "reboot to apply boot-time changes: sudo reboot"
fi
