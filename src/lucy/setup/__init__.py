"""Setup mode: Lucy's own Wi-Fi and a settings page, so nobody needs SSH or a monitor.

A grown-up holds Lucy's wing while she wakes up (or she finds no network after boot). She
stops being a goose for a while, turns on her own Wi-Fi hotspot, and says so. Scanning the
card sewn inside her zipper joins it, and the phone opens her settings page by itself, as
on hotel Wi-Fi. From there the grown-up picks a network, changes her settings, and tries
her voice and motors. She says out loud whether a network worked, because the phone loses
touch with her while she tries it, and she puts everything back if it did not.

::

    network.py   NetworkManager: scanning, the hotspot, joining with rollback (and a demo)
    settings.py  the fields on the page; config.toml edited in place, checked before saving
    hotspot.py   her setup Wi-Fi's name and password, and the printable card with its QR code
    pages.py     the HTML (no JavaScript, so it works in a phone's sign-in window)
    http.py      a small HTTP/1.1 server on the event loop
    files.py     writing a file in one step, so a power cut never leaves half of it
    body.py      her on-device voice, beak, and body while in setup mode
    mode.py      SetupMode: ties it together, serves the page, and decides when to leave

It runs as lucy-setup.service (deploy/systemd), which stops lucy.service while it runs and
starts her again afterwards, however it ends.
"""
