VOICE := en_US-amy-medium
VOICE_URL := https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium

.PHONY: dev check lint typecheck test integration voice lock clean

dev:
	uv sync --extra pi

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

integration: voice
	LUCY_TEST_VOICE=voices/$(VOICE).onnx uv run pytest -q tests/test_integration.py

# Re-resolve every dependency to its newest release that is at least two weeks old
# (exclude-newer in pyproject.toml). uv.lock is universal: it also pins the Pi's aarch64 wheels.
lock:
	uv lock --upgrade

clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache dist build
