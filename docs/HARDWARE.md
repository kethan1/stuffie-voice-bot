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
| DRV8833 dual motor driver breakout | Drives the beak and body motors from their own battery. Its inputs work at the Pi's 3.3 V logic. |
| 2 small DC gear motors (3–6 V, e.g. N20 or TT) | One pushes the beak open against a spring, one wiggles the body. Prefer a low stall current (under about 0.5 A). |
| Motor battery: 4 × AA NiMH (4.8 V) in a holder with a switch | Motor power only (the DRV8833's VM, 2.7–10.8 V). The Pi keeps its own supply, so a motor starting up can never brown it out. |
| A light spring or elastic band | Closes the beak whenever the motor lets go. |
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
| DRV8833 AIN1 | GPIO12 (pin 32) | Beak speed: hardware PWM channel 0 |
| DRV8833 AIN2 | GPIO22 (pin 15) | Beak direction |
| DRV8833 BIN1 | GPIO13 (pin 33) | Body speed: hardware PWM channel 1 |
| DRV8833 BIN2 | GPIO23 (pin 16) | Body direction |
| DRV8833 nSLEEP (SLP) | GPIO24 (pin 18) | Lucy wakes the driver only to move. See "Motors" if your board pulls it up |
| DRV8833 GND | GND (pin 34) | **Shared ground**: the Pi, the driver, and the motor battery's − |
| DRV8833 VM | motor battery + | **Never** the Pi's 5 V pins |
| Beak motor | DRV8833 AOUT1, AOUT2 | Swap the two wires if the beak pushes shut instead of open |
| Body motor | DRV8833 BOUT1, BOUT2 | |

```
                    3V3  (1) (2)  5V ──────── MAX98357A VIN
                         (3) (4)  5V
                         (5) (6)  GND ─────── MAX98357A GND
         amp SD ─ GPIO4  (7) (8)
         button ─ GND    (9) (10)
         button ─ GPIO17 (11) (12) GPIO18 ─── MAX98357A BCLK
            LED ─ GPIO27 (13) (14) GND ────── LED cathode
  DRV8833 AIN2 ─ GPIO22 (15) (16) GPIO23 ─── DRV8833 BIN2
                         (17) (18) GPIO24 ─── DRV8833 nSLEEP
                         ...
                         (31) (32) GPIO12 ─── DRV8833 AIN1 (beak PWM)
  DRV8833 BIN1 ─ GPIO13 (33) (34) GND ────── DRV8833 GND (+ motor battery −)
        amp LRC ─ GPIO19 (35) (36)
                         ...  (40) GPIO21 ─── MAX98357A DIN
```

Then install with `sudo deploy/install.sh --i2s-amp --motors --tune` and
reboot. The installer adds `dtoverlay=max98357a` (the amplifier) and
`dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4` (the motors' hardware PWM)
to `/boot/firmware/config.txt`, and turns on `[motors]` in
`/etc/lucy/config.toml`. Check that the card appeared with `aplay -l` (look
for `MAX98357A`), and that the PWM did with `ls /sys/class/pwm` (`pwmchip0`).

Using a USB speaker instead: skip `--i2s-amp`. The installer picks USB sound
cards automatically, or pass `--playback-card <id>` (see `aplay -l`).

## Motors

Lucy's beak opens in time with her voice, and her body flaps on every "Honk!"
and bobs as she talks (`lucy.core.motion`, ARCHITECTURE.md §5.7).

**The beak** is a hinged lower beak that the motor pulls open against a light
spring. The motor only ever pushes it open, by an amount that follows how loud
each syllable is. When the motor lets go, the spring snaps the beak shut. Set
`motors.motion.beak_min_power` to the least power that just starts to move it,
and `beak_max_power` to the power that opens it fully without slamming it.

**The body** motor can drive anything that wiggles: the wings through a crank,
the neck, or the whole body on a pivot. If it can run both ways, Lucy flaps
out and back and her bobs alternate like a sway. If it may only run forward
(for example, against a return spring), set `motors.motion.body_two_way =
false`.

**Check the wiring** before closing the plush. The service must be stopped,
because it holds the lines:

```
sudo systemctl stop lucy
sudo lucy motors          # the beak opens wide 3 times, then a little 3 times; the body forward, then back
sudo lucy say "Honk! Did you know geese can fly in a V?"   # watch the beak follow her voice
sudo systemctl start lucy
```

If a motor turns the wrong way, swap its two wires at the driver.

**Driver boards.** Any DRV8833 breakout works (Adafruit, Pololu, and the
generic purple boards). Some pull SLP (nSLEEP) up to the motor battery on the
board. If yours does, **do not wire SLP to the Pi**: a battery above 3.3 V
would push current into the GPIO pin. Leave SLP unconnected and set
`motors.sleep_line = -1`. The driver then stays awake, which costs a couple of
milliamps from the motor battery. The Pi's ground, the driver's GND, and the
motor battery's − must all be connected, or the driver cannot read the Pi's
signals.

**The PWM channels** come from the Pi's PWM hardware, so the onboard 3.5 mm
audio must stay off (`dtparam=audio=off`, which `--tune` sets): it uses the
same hardware. GPIO12 and GPIO13 are not used by the I2S amplifier
(GPIO18/19/21), so the two never clash.

**Safety.** Software keeps the motors gentle. Power is capped, no motor runs
longer than `max_hold_s` (2 s) without a rest, nothing moves while the wing is
squeezed, and the driver sleeps whenever Lucy is still. If Lucy crashes or is
killed, systemd stops both motors (`ExecStopPost=lucy motors --off`). Every
motor line is pulled low at power-up, so nothing moves while the Pi boots. The
mechanics are up to the builder:

* Use a **weak spring** on the beak, so a finger in it is never pinched hard,
  and round off its edges.
* **Enclose every gear and linkage** in a box or sleeve, so no fingers,
  fabric, or stuffing can be caught in them.
* Choose motors and a battery voltage within the motors' rating. The DRV8833
  limits current (about 1.5 A per motor) and shuts down if it overheats, but
  a small motor can still get hot if it is stalled at too high a voltage.
* **Secure the battery** in a closed pocket with its switch reachable only by
  a grown-up. Use rechargeable NiMH or a proper holder; never loose cells.

## Building it into the plush

* **Heat:** a Pi 4 is a 3–7 W heater. Put the Pi in a small ventilated box or
  mesh pocket with a heatsink, never wrapped directly in stuffing. Lucy's
  power profiles keep it cool when idle. Check the temperature with
  `vcgencmd measure_temp` after a long conversation (keep it below ~70 °C).
* **Microphone** near the head, facing out through a small opening in the
  fabric. **Speaker** in the belly, facing the other way. Distance between the
  two avoids picking up Lucy's own voice.
* **Motors** in foam, away from the microphone. They never move while Lucy
  listens, but mounting them softly keeps them quiet while she talks.
* **Button** in a wing, padded so a gentle squeeze presses it.
* **Setup card:** print Lucy's setup card (docs/OPERATIONS.md, "Setup mode")
  and keep it in a pocket inside the access seam. It is how a grown-up reaches
  her settings from a phone, so it belongs where children don't go.
* **Child safety:** no exposed wires, pins, or small parts. Secure the battery,
  route cables so they can't be pulled, and close the access seam with a zip
  or hook-and-loop that a grown-up opens. Supervise use.
