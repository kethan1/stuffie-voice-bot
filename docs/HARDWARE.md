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
| L293D dual H-bridge motor driver (16-pin DIP) | Drives the beak and body motors from their own battery. It reads the Pi's 3.3 V signals as high, and has its motor diodes built in (the "D"; a plain L293 would need eight more). |
| 2 small DC gear motors (6 V, e.g. N20 or TT) | One pushes the beak open against a spring, one wiggles the body. The stall current must be under 0.6 A, the L293D's limit per motor. |
| Motor battery: 6 × AA NiMH (7.2 V) in a holder with a switch | Motor power only (the L293D's VCC2, 4.5–36 V). The L293D loses about 2 V inside itself, so the motors get about 5 V. The Pi keeps its own supply, so a motor starting up can never brown it out. |
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
| L293D 1,2EN (chip pin 1) | GPIO12 (pin 32) | Beak speed: hardware PWM channel 0 |
| L293D 1A (chip pin 2) | GPIO22 (pin 15) | Beak forward (opens it) |
| L293D 2A (chip pin 7) | GPIO23 (pin 16) | Beak backward |
| L293D 3,4EN (chip pin 9) | GPIO13 (pin 33) | Body speed: hardware PWM channel 1 |
| L293D 3A (chip pin 10) | GPIO24 (pin 18) | Body forward |
| L293D 4A (chip pin 15) | GPIO25 (pin 22) | Body backward |
| L293D VCC1 (chip pin 16) | 5 V (pin 4) | The chip's logic supply |
| L293D GND (chip pins 4, 5, 12, 13) | GND (pin 34) | **Shared ground**: the Pi, the L293D, and the motor battery's −. Connect all four |
| L293D VCC2 (chip pin 8) | motor battery + | **Never** the Pi's 5 V pins |
| Beak motor | L293D 1Y, 2Y (chip pins 3, 6) | Swap the two wires if the beak pushes shut instead of open |
| Body motor | L293D 3Y, 4Y (chip pins 11, 14) | |

```
                    3V3  (1) (2)  5V ──────── MAX98357A VIN
                         (3) (4)  5V ──────── L293D VCC1 (chip pin 16)
                         (5) (6)  GND ─────── MAX98357A GND
         amp SD ─ GPIO4  (7) (8)
         button ─ GND    (9) (10)
         button ─ GPIO17 (11) (12) GPIO18 ─── MAX98357A BCLK
            LED ─ GPIO27 (13) (14) GND ────── LED cathode
      L293D 1A ─ GPIO22 (15) (16) GPIO23 ─── L293D 2A
                         (17) (18) GPIO24 ─── L293D 3A
                         (19) (20)
                         (21) (22) GPIO25 ─── L293D 4A
                         ...
                         (31) (32) GPIO12 ─── L293D 1,2EN (beak PWM)
   L293D 3,4EN ─ GPIO13 (33) (34) GND ────── L293D GND (+ motor battery −)
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
cards automatically, or pass `--playback-card <id>` (see `aplay -l`). A USB
speaker can later be swapped for another, or moved to another port, without
re-installing: when the configured card is missing, Lucy uses whichever USB
speaker is plugged in.

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

**The L293D.** Each motor uses one half of the chip. Its enable pin (1,2EN or
3,4EN) sets the speed with hardware PWM, and its two inputs (1A and 2A, or 3A
and 4A) set the direction. The PWM is on the enable pins because an L293D
brakes a motor whose two inputs are at the same level. With the enable pin
low, its outputs are off and the motor coasts, so the spring can snap the
beak shut. When wiring it:

* **Two supplies.** VCC1 (chip pin 16) powers the chip's logic: connect it to
  the Pi's 5 V. VCC2 (chip pin 8) powers the motors: connect it to the motor
  battery, never to the Pi.
* **All four GND pins** (chip pins 4, 5, 12, 13) go to ground. They also carry
  the chip's heat away, so on a soldered board give them a wide copper area.
  The Pi's ground, the L293D's GND, and the motor battery's − must all be
  connected, or the chip cannot read the Pi's signals.
* **It loses about 2 V.** The L293D's output transistors drop about 2 to
  2.5 V, so each motor gets that much less than the battery. That is why the
  battery is 7.2 V for 6 V motors. If the beak barely moves, raise
  `beak_min_power` before anything else.
* **It has no sleep mode.** With both enable pins low the outputs are off and
  nothing moves, but the chip still draws a little from the Pi's 5 V and from
  the motor battery. Switch the motor battery off when Lucy is put away.
* Pin 1 of the chip is next to the notch (or dot) on its top. Upgrading from a
  DRV8833? `deploy/install.sh` removes its old settings from
  `/etc/lucy/config.toml`; rewire to the table above.

**The PWM channels** come from the Pi's PWM hardware, so the onboard 3.5 mm
audio must stay off (`dtparam=audio=off`, which `--tune` sets): it uses the
same hardware. GPIO12 and GPIO13 are not used by the I2S amplifier
(GPIO18/19/21), so the two never clash.

**Safety.** Software keeps the motors gentle. Power is capped, no motor runs
longer than `max_hold_s` (2 s) without a rest, nothing moves while the wing is
squeezed, and both motors coast whenever Lucy is still. If Lucy crashes or is
killed, systemd stops both motors (`ExecStopPost=lucy motors --off`). Every
motor line is pulled low at power-up, so nothing moves while the Pi boots. The
mechanics are up to the builder:

* Use a **weak spring** on the beak, so a finger in it is never pinched hard,
  and round off its edges.
* **Enclose every gear and linkage** in a box or sleeve, so no fingers,
  fabric, or stuffing can be caught in them.
* Choose motors and a battery voltage within the motors' rating. The L293D
  turns its outputs off if it overheats, but it does not limit current, so
  pick motors that stall at under 0.6 A. A small motor can still get hot if
  it is stalled at too high a voltage.
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
