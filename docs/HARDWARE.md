# Lucy the Goose: Hardware

## Bill of materials

| Part | Notes |
|---|---|
| Raspberry Pi 4 Model B, 4 GB | Fit a small heatsink: it lives inside a plush. |
| 32 GB microSD, A1 or A2 rated | A1/A2 cards handle small random reads/writes far better. The optional offline brain (`--local-backup`) needs about 2 GB more, and loads faster from a fast card. |
| USB mini microphone | Plug-and-play. Ideally one that supports 16 kHz natively (`arecord --dump-hw-params -D hw:CARD=<id>`). |
| MAX98357A I2S amplifier breakout + 3 W, 4 Ω speaker | Loud, efficient, and gated by the kernel when idle. A USB speaker also works. |
| Momentary push button | Sewn into the wing. A 12 mm tactile switch or a soft "squeeze" switch. |
| 5 mm LED + 330 Ω resistor (optional) | Status light, e.g. behind an eye. |
| 5.1 V / 3 A USB-C supply, or a USB-C PD power bank (5 V / 3 A) | A weak power bank causes under-voltage resets (`vcgencmd get_throttled`). |
| Female-to-female jumper wires, heat-shrink | Keep connections secure: children squeeze things. |

## Wiring

Pin numbers are **BCM GPIO** (what Lucy's config uses) and **physical header
pin** (what you count on the board).

| Connection | Pi pin | Notes |
|---|---|---|
| Button, one leg | GPIO17 (pin 11) | Internal pull-up is enabled by Lucy, so no resistor is needed |
| Button, other leg | GND (pin 9) | |
| LED anode (+, long leg) via 330 Ω | GPIO27 (pin 13) | |
| LED cathode (−) | GND (pin 14) | |
| MAX98357A VIN | 5 V (pin 2) | |
| MAX98357A GND | GND (pin 6) | |
| MAX98357A BCLK | GPIO18 (pin 12) | I2S bit clock |
| MAX98357A LRC | GPIO19 (pin 35) | I2S word select |
| MAX98357A DIN | GPIO21 (pin 40) | I2S data |
| MAX98357A SD | GPIO4 (pin 7) | Shutdown pin: the kernel turns the amp on only while audio plays |
| MAX98357A GAIN | leave unconnected | 9 dB default; tie to GND for 12 dB if Lucy is too quiet |
| USB microphone | any USB port | |

```
           3V3  (1) (2)  5V ─────────── MAX98357A VIN
               (3) (4)  5V
               (5) (6)  GND ─────────── MAX98357A GND
 amp SD ─ GPIO4 (7) (8)
  button ─ GND  (9) (10)
  button ─ GPIO17 (11) (12) GPIO18 ──── MAX98357A BCLK
     LED ─ GPIO27 (13) (14) GND ─────── LED cathode
               ...
     amp LRC ─ GPIO19 (35) (36)
               ...         (40) GPIO21 ─ MAX98357A DIN
```

Then install with `sudo deploy/install.sh --i2s-amp --tune` and reboot. The
installer adds `dtoverlay=max98357a` to `/boot/firmware/config.txt`. Check
that the card appeared with `aplay -l` (look for `MAX98357A`).

Using a USB speaker instead: skip `--i2s-amp`. The installer picks USB sound
cards automatically, or pass `--playback-card <id>` (see `aplay -l`).

## Building it into the plush

* **Heat:** a Pi 4 is a 3–7 W heater. Put the Pi in a small ventilated box or
  mesh pocket with a heatsink, never wrapped directly in stuffing. Lucy's
  power profiles keep it cool when idle. Check the temperature with
  `vcgencmd measure_temp` after a long conversation (keep it below ~70 °C).
* **Microphone** near the head, facing out through a small opening in the
  fabric. **Speaker** in the belly, facing the other way. Distance between the
  two avoids picking up Lucy's own voice.
* **Button** in a wing, padded so a gentle squeeze presses it.
* **Child safety:** no exposed wires, pins, or small parts. Secure the battery,
  route cables so they can't be pulled, and close the access seam with a zip
  or hook-and-loop that a grown-up opens. Supervise use.
