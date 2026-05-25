"""Integration tests for the Ollama inference client.

Scope:
  * Happy path: send a trivial prompt, assert a string comes back.
  * Configuration verification: tenacity retrier wired correctly, even
    if not triggered in the happy path.
  * Error classification: ``_is_retryable`` correctly separates
    transient failures from programmer bugs.

Pre-requisite for the happy-path test:
  * Ollama reachable at ``localhost:11434`` with the configured model
    pulled (``bash scripts/setup_infra.sh``).
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator

import httpx
import pytest
from tenacity import AsyncRetrying, stop_after_attempt

from swarm.inference.ollama_client import (
    InferenceResponse,
    OllamaClient,
    _is_retryable,
)

OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
MODEL_NAME = os.getenv("TEST_MODEL_NAME", "llama3.2:1b")


def _ollama_reachable() -> bool:
    try:
        with socket.create_connection((OLLAMA_HOST, OLLAMA_PORT), timeout=1):
            return True
    except OSError:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_reachable(),
    reason=(
        f"Ollama not reachable on {OLLAMA_HOST}:{OLLAMA_PORT}. "
        "Run `bash scripts/setup_infra.sh` first."
    ),
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
async def client() -> AsyncIterator[OllamaClient]:
    c = OllamaClient(
        endpoint=OLLAMA_URL,
        model_name=MODEL_NAME,
        max_retries=3,
        backoff_base_s=1.0,  # tighter than default — tests shouldn't sit on backoff
        backoff_jitter=True,
        # First call may cold-load the model; give it room.
        read_timeout_s=180.0,
    )
    try:
        yield c
    finally:
        await c.aclose()


# --------------------------------------------------------------------------- #
# Happy-path inference
# --------------------------------------------------------------------------- #


@requires_ollama
@pytest.mark.asyncio
async def test_generate_returns_valid_structured_response(client: OllamaClient) -> None:
    resp = await client.generate(
        prompt="What is 2+2? Answer with a single number and nothing else.",
        temperature=0.0,
        max_tokens=16,
    )

    # Structural contract.
    assert isinstance(resp, InferenceResponse)
    assert isinstance(resp.text, str)
    assert resp.text.strip(), "Ollama returned empty text"

    # Token accounting populated.
    assert resp.tokens_in > 0, "prompt_eval_count missing from Ollama response"
    assert resp.tokens_out > 0, "eval_count missing from Ollama response"

    # Latency measured.
    assert resp.latency_ms > 0

    # Model echoed back (allow tag variations, e.g. ":1b").
    assert MODEL_NAME.split(":")[0] in resp.model.lower()


@requires_ollama
@pytest.mark.asyncio
async def test_generate_accepts_system_prompt_and_seed(client: OllamaClient) -> None:
    """Sanity check that optional knobs are forwarded without errors."""
    resp = await client.generate(
        prompt="Say the word 'banana' and nothing else.",
        system="You are a terse assistant. Output exactly what is asked.",
        temperature=0.0,
        seed=42,
        max_tokens=8,
    )
    assert resp.text.strip()


# --------------------------------------------------------------------------- #
# Retry configuration (verified WITHOUT actually triggering retries)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_retry_config_attached_to_client() -> None:
    c = OllamaClient(
        endpoint=OLLAMA_URL,
        model_name=MODEL_NAME,
        max_retries=5,
        backoff_base_s=2.5,
        backoff_jitter=True,
    )
    try:
        # Public knobs surfaced for observability / verification.
        assert c.max_retries == 5
        assert c.backoff_base_s == 2.5
        assert c.backoff_jitter is True

        # Internal tenacity object is a proper AsyncRetrying.
        assert isinstance(c._retrier, AsyncRetrying)

        # stop_after_attempt counts include the initial attempt, so the
        # max attempt number is max_retries + 1.
        assert isinstance(c._retrier.stop, stop_after_attempt)
        assert c._retrier.stop.max_attempt_number == 6  # 5 retries + 1 initial

        # Reraise must be on — otherwise tenacity wraps the original
        # exception in RetryError and we lose the http status info.
        assert c._retrier.reraise is True

        # before_sleep hook present (used for retry telemetry).
        assert c._retrier.before_sleep is not None
    finally:
        await c.aclose()


def test_retry_classifier_treats_5xx_as_retryable() -> None:
    request = httpx.Request("POST", "http://x/api/generate")
    resp_503 = httpx.Response(503, request=request)
    err = httpx.HTTPStatusError("unavailable", request=request, response=resp_503)
    assert _is_retryable(err) is True


def test_retry_classifier_treats_4xx_as_non_retryable() -> None:
    request = httpx.Request("POST", "http://x/api/generate")
    resp_400 = httpx.Response(400, request=request)
    err = httpx.HTTPStatusError("bad request", request=request, response=resp_400)
    assert _is_retryable(err) is False

    resp_404 = httpx.Response(404, request=request)
    err = httpx.HTTPStatusError("not found", request=request, response=resp_404)
    assert _is_retryable(err) is False


def test_retry_classifier_handles_network_errors() -> None:
    assert _is_retryable(httpx.TimeoutException("read timeout")) is True
    assert _is_retryable(httpx.ConnectError("connection refused")) is True
    assert _is_retryable(httpx.ReadError("reset")) is True


def test_retry_classifier_rejects_programmer_bugs() -> None:
    assert _is_retryable(ValueError("typo")) is False
    assert _is_retryable(KeyError("missing")) is False
    assert _is_retryable(RuntimeError("oops")) is False


def test_invalid_retry_config_rejected() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        OllamaClient(max_retries=-1)
    with pytest.raises(ValueError, match="backoff_base_s"):
        OllamaClient(backoff_base_s=0.0)
