"""Async Ollama client with tenacity-based resilience.

Resilience policy (edital item 4):
  * Exponential backoff with jitter on retry.
  * Max retries configurable per worker (from ``RunPlan.resilience``).
  * Retries fire ONLY on transient failures (network, timeout, 5xx).
  * 4xx responses fail fast — they signal a bug (bad model name,
    malformed payload), and retrying them just wastes time.

The client is async so a single worker process can in principle keep
multiple in-flight inference calls (useful later if we want a single
worker to pull a batch of messages and pipeline them).
"""

from __future__ import annotations

import logging
import time
from types import TracebackType
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt
from tenacity import (
    AsyncRetrying,
    RetryError,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

# Default network-level timeouts — generous because cold-loading a model
# into Ollama can take 10–30s. Override per call site if needed.
_DEFAULT_CONNECT_TIMEOUT_S = 10.0
_DEFAULT_READ_TIMEOUT_S = 120.0


class InferenceResponse(BaseModel):
    """Structured response returned by :meth:`OllamaClient.generate`.

    Carries only what the worker needs to (a) build a ``Result`` message
    and (b) emit observability metrics. Raw Ollama fields not listed
    here are dropped.
    """

    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=0)
    model: str
    tokens_in: NonNegativeInt = 0
    tokens_out: NonNegativeInt = 0
    latency_ms: NonNegativeInt = 0


def _is_retryable(exc: BaseException) -> bool:
    """Classify exceptions: retry transient failures, surface bugs.

    Retryable:
      * Network errors (DNS, connection refused, reset, etc.).
      * Read/connect/pool timeouts.
      * HTTP 5xx responses (Ollama is overloaded or recovering).

    Non-retryable (fail fast):
      * HTTP 4xx (bad model, malformed payload — code bug).
      * Anything else (programming errors, unexpected exceptions).
    """
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    return False


class OllamaClient:
    """Async client for Ollama's ``/api/generate`` endpoint."""

    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        model_name: str = "llama3.2:1b",
        *,
        connect_timeout_s: float = _DEFAULT_CONNECT_TIMEOUT_S,
        read_timeout_s: float = _DEFAULT_READ_TIMEOUT_S,
        max_retries: int = 3,
        backoff_base_s: float = 2.0,
        backoff_max_s: float = 30.0,
        backoff_jitter: bool = True,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if backoff_base_s <= 0:
            raise ValueError("backoff_base_s must be > 0")

        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.backoff_jitter = backoff_jitter

        self._client = httpx.AsyncClient(
            base_url=self.endpoint,
            timeout=httpx.Timeout(
                connect=connect_timeout_s,
                read=read_timeout_s,
                write=connect_timeout_s,
                pool=connect_timeout_s,
            ),
        )

        # tenacity "attempt" includes the first try, so total attempts =
        # 1 (initial) + max_retries.
        if backoff_jitter:
            wait_strategy = wait_exponential_jitter(
                initial=backoff_base_s,
                max=backoff_max_s,
            )
        else:
            wait_strategy = wait_exponential(
                multiplier=backoff_base_s,
                max=backoff_max_s,
            )

        self._retrier: AsyncRetrying = AsyncRetrying(
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_strategy,
            retry=retry_if_exception(_is_retryable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    # ------------------------------------------------------------------ #
    # Async context-manager plumbing
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        seed: int | None = None,
        max_tokens: int | None = None,
        model_name: str | None = None,
    ) -> InferenceResponse:
        """Run a single non-streaming inference.

        Parameters
        ----------
        prompt : str
            The user-facing prompt (already rendered from a template).
        system : str, optional
            System prompt. The worker loads this from ``prompts/`` per
            strategy and forwards it here unchanged.
        temperature : float
            Sampling temperature. Must be > 0 for self-consistency.
        seed : int, optional
            Per-sample seed. Self-consistency varies this to get
            independent reasoning traces.
        max_tokens : int, optional
            Maps to Ollama's ``num_predict``.
        model_name : str, optional
            Override the model for this call (otherwise uses the
            client's configured default).
        """
        payload: dict[str, Any] = {
            "model": model_name or self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system is not None:
            payload["system"] = system
        if seed is not None:
            payload["options"]["seed"] = seed
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        data, latency_ms = await self._retrier(self._post_generate, payload)

        return InferenceResponse(
            text=data.get("response", ""),
            model=data.get("model", payload["model"]),
            tokens_in=int(data.get("prompt_eval_count", 0)),
            tokens_out=int(data.get("eval_count", 0)),
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    async def _post_generate(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """One HTTP attempt — wrapped by ``_retrier`` in :meth:`generate`.

        Returns the parsed JSON body and the latency of this attempt
        (so total stats reflect the SUCCESSFUL call, not cumulative
        wall-clock across retries).
        """
        start = time.perf_counter()
        resp = await self._client.post("/api/generate", json=payload)
        resp.raise_for_status()
        latency_ms = int((time.perf_counter() - start) * 1000)
        return resp.json(), latency_ms


__all__ = ["InferenceResponse", "OllamaClient", "RetryError"]
