VOICE := en_US-amy-medium
VOICE_URL := https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium
# The offline brain's speech model, at the same pinned revision deploy/install.sh uses.
STT_MODEL := models/faster-whisper-tiny.en
STT_MODEL_URL := https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/0d3d19a32d3338f10357c0889762bd8d64bbdeba

.PHONY: dev check lint typecheck test integration voice stt-model lock clean run doctor say bench

dev:
	uv sync --extra pi --extra local

check: lint typecheck test

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy src

test:
	uv run pytest -q

voice: voices/$(VOICE).onnx

voices/$(VOICE).onnx:
	mkdir -p voices
	curl -fsSL -o $@.json $(VOICE_URL)/$(VOICE).onnx.json
	curl -fsSL -o $@ $(VOICE_URL)/$(VOICE).onnx

stt-model: $(STT_MODEL)/model.bin

$(STT_MODEL)/model.bin:
	mkdir -p $(STT_MODEL)
	for f in config.json tokenizer.json vocabulary.txt model.bin; do \
		curl -fsSL -o $(STT_MODEL)/$$f $(STT_MODEL_URL)/$$f || exit 1; done

# End-to-end tests with the real voice and speech model (Groq and Ollama are mocked).
integration: voice stt-model
	LUCY_TEST_VOICE=voices/$(VOICE).onnx LUCY_TEST_STT_MODEL=$(STT_MODEL) \
		uv run pytest -q tests/test_integration.py tests/test_local.py

# -- on a laptop (dev.toml: laptop mic and speakers, Enter instead of the wing button) -----
# The API key comes from GROQ_API_KEY or a gitignored .env file.
# `make run CONFIG=my.toml` uses another config.
CONFIG := dev.toml
LUCY = uv run lucy --config $(CONFIG)

run: voice
	$(LUCY) run

doctor: voice
	$(LUCY) doctor --listen

say: voice
	$(LUCY) say "$(or $(TEXT),Honk! Hello from your laptop.)"

bench: voice
	$(LUCY) bench

# Re-resolve every dependency to its newest release that is at least two weeks old
# (exclude-newer in pyproject.toml). uv.lock is universal: it also pins the Pi's aarch64 wheels.
lock:
	uv lock --upgrade

clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache dist build
