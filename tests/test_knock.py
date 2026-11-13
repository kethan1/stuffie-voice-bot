"""What counts as asking for setup mode (after the knock: see tests/test_controller.py)."""

import pytest

from lucy.core.knock import asks_for_setup


@pytest.mark.parametrize(
    "said",
    [
        "Enter into setup mode.",
        "Enter Setup Mode",  # how Whisper writes a short command
        "enter into set up mode",
        "Enter into set-up mode, please.",
        "Lucy, enter into setup mode, please.",
        "Lucy enter into set up mode please.",
        "Go into setup mode.",
        "Go to setup mode",
        "Start setup mode",
        "Setup mode.",
        "Set up mode.",
        "Okay Lucy, can you go into your setup mode now?",
        "Enter setup",
        "Enter into the setup",
        "Start setup",
        "Open your setup",
        "Go into setup",
    ],
)
def test_asks_for_setup(said: str) -> None:
    assert asks_for_setup(said)


@pytest.mark.parametrize(
    "said",
    [
        "",
        "Why do geese fly in a V shape?",
        "What is setup mode?",
        "How do I enter setup mode?",
        "Is this setup mode?",
        "Don't go into setup mode.",
        "Don\N{RIGHT SINGLE QUOTATION MARK}t enter setup mode",  # how Whisper writes it
        "Do not start setup mode",
        "Never enter setup mode",
        "No setup mode",
        "Let's get set up for a riddle",
        "Can you help me set up my science project?",
        "Tell me a story about a goose who set up a lemonade stand",
        "Enter the cave",
        "Setup",
        "Once there was a robot who would always enter into setup mode when it rained",
    ],
)
def test_does_not(said: str) -> None:
    assert not asks_for_setup(said)
