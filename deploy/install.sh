#!/usr/bin/env bash
# Install (or upgrade) Lucy the Goose on Raspberry Pi OS Lite (64-bit, Bookworm or Trixie).
#
#   sudo deploy/install.sh [options]
#
#   --i2s-amp              add the MAX98357A I2S amplifier overlay (reboot required)
#   --playback-card NAME   ALSA card id for the speaker   (default: auto-detect)
#   --capture-card NAME    ALSA card id for the microphone (default: auto-detect)
#   --voice NAME           Piper voice, e.g. en_US-amy-medium (default)
#   --tune                 power tuning: boot config, unused services off, journal in RAM
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
VOICE=en_US-amy-medium
VOICE_BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main
# Checksums of the default voice, so a corrupted or tampered download is refused.
declare -A VOICE_SHA256=(
  [en_US-amy-medium.onnx]=b3a6e47b57b8c7fbe6a0ce2518161a50f59a9cdd8a50835c02cb02bdd6206c18
  [en_US-amy-medium.onnx.json]=95a23eb4d42909d38df73bb9ac7f45f597dbfcde2d1bf9526fdeaf5466977d77
)
# A pinned uv release, checksum-verified like the voice. It installs the hash-locked uv.lock.
UV_VERSION=0.12.8
declare -A UV_SHA256=(
  [aarch64]=ba8661f4fd207c8e94814191598e619b355ac10d5014e851e21eb800f9ef2b00
  [x86_64]=2e2b37e9811e17675a9e70bed5e1a58fc8c0388be63d751d72cc735188c149ff
)
UV="$PREFIX/bin/uv"
I2S_AMP=0 TUNE=0 READ_ONLY=0 START=1 PLAYBACK_CARD="" CAPTURE_CARD=""

log() { printf '\033[1;33m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;31m!!\033[0m %s\n' "$*" >&2; }
die() { warn "$*"; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --i2s-amp) I2S_AMP=1 ;;
    --playback-card) PLAYBACK_CARD="${2:?}"; shift ;;
    --capture-card) CAPTURE_CARD="${2:?}"; shift ;;
    --voice) VOICE="${2:?}"; shift ;;
    --tune) TUNE=1 ;;
    --read-only) READ_ONLY=1 ;;
    --no-start) START=0 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
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

log "installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  python3 libportaudio2 alsa-utils iw ca-certificates curl rsync >/dev/null

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
  --extra pi --no-editable --no-build --compile-bytecode
UV_CACHE_DIR=/var/cache/lucy/uv "$UV" cache prune --quiet

log "fetching the Piper voice $VOICE"
IFS=- read -r lang name quality <<<"$VOICE"
[[ -n ${quality:-} ]] || die "voice must look like en_US-amy-medium"
for file in "$VOICE.onnx" "$VOICE.onnx.json"; do
  dest="$PREFIX/voices/$file"
  if [[ ! -s $dest ]]; then
    curl -fsSL --retry 3 -o "$dest.part" "$VOICE_BASE/${lang%%_*}/$lang/$name/$quality/$file"
    mv "$dest.part" "$dest"
  fi
  if [[ -n ${VOICE_SHA256[$file]:-} ]]; then
    echo "${VOICE_SHA256[$file]}  $dest" | sha256sum --check --quiet ||
      { rm -f "$dest"; die "checksum mismatch for $file (download removed; re-run to retry)"; }
  else
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
echo "   sudo lucy say 'Honk!'          # test the speaker"
echo "   journalctl -u lucy -f"
if [[ $I2S_AMP -eq 1 || $TUNE -eq 1 || $READ_ONLY -eq 1 ]]; then
  warn "reboot to apply boot-time changes: sudo reboot"
fi
