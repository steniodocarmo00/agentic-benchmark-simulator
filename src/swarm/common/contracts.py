"""Pydantic v2 contracts shared by Master, Workers, and Aggregator.

Design rules:
  * Messages that travel through SQS (``Task``, ``Result``) are FROZEN —
    they are wire artifacts and must not mutate after creation.
  * Configuration models (``RunPlan`` and friends) are mutable so the
    Master can patch/derive them at boot.
  * ``extra="forbid"`` everywhere: catches schema drift loudly instead
    of silently accepting a producer that added a field the consumer
    does not understand.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Strategy(str, Enum):
    """Prompting strategy applied to a single GSM8K problem."""

    ZERO_SHOT = "zero_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    SELF_CONSISTENCY = "self_consistency"


class ResultStatus(str, Enum):
    """Terminal status of a Worker's attempt to answer a Task."""

    SUCCESS = "success"  # inference + parse succeeded
    PARSE_FAILED = "parse_failed"  # model answered but parser failed
    INFERENCE_FAILED = "inference_failed"  # all retries exhausted


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Timezone-aware UTC ``now`` (Pydantic serializes to ISO-8601 with offset)."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Configuration models (mutable — used by the Master at boot)
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    provider: Literal["ollama"] = "ollama"
    endpoint: str = Field(min_length=1)
    name: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: Strategy
    prompt_id: str = Field(min_length=1)
    k_samples: PositiveInt


class ResilienceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inference_max_retries: PositiveInt = 3
    inference_backoff_base_s: PositiveFloat = 2.0
    inference_backoff_jitter: bool = True
    sqs_max_receive_count: PositiveInt = 3


class WorkersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Edital requirement: real parallelism with >= 2 nodes.
    count: int = Field(ge=2)
    poll_wait_seconds: NonNegativeInt = 10
    visibility_timeout_seconds: PositiveInt = 120


class RunPlan(BaseModel):
    """Declarative description of an experiment.

    The Master loads this (from ``config/run.yaml``), then materializes
    ``subset_size * sum(k_samples)`` ``Task`` messages onto the work queue.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    benchmark: Literal["gsm8k"]
    subset_size: PositiveInt
    seed: int
    model: ModelConfig
    strategies: list[StrategyConfig] = Field(min_length=1)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    workers: WorkersConfig


# ---------------------------------------------------------------------------
# Wire messages (frozen — immutable envelopes that flow through SQS)
# ---------------------------------------------------------------------------


class Task(BaseModel):
    """One atomic unit of work. Master → ``tasks`` queue → Worker."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    task_id: UUID = Field(default_factory=uuid4)
    run_id: str = Field(min_length=1)
    problem_id: str = Field(min_length=1)
    strategy: Strategy
    sample_idx: NonNegativeInt
    question: str = Field(min_length=1)
    # Prompt provenance — every Task is reproducible from these three:
    prompt_id: str = Field(min_length=1)
    prompt_version: PositiveInt
    prompt_hash: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=_utcnow)


class Result(BaseModel):
    """Output of a single inference. Worker → ``results`` queue → Aggregator.

    Correctness evaluation (vs. gold answer) is intentionally NOT here —
    the Aggregator owns that, since the Worker has no business holding
    the answer key.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    task_id: UUID
    run_id: str = Field(min_length=1)
    problem_id: str = Field(min_length=1)
    strategy: Strategy
    sample_idx: NonNegativeInt
    status: ResultStatus
    raw_response: str | None = None
    parsed_answer: str | None = None
    latency_ms: NonNegativeInt
    tokens_in: NonNegativeInt = 0
    tokens_out: NonNegativeInt = 0
    worker_id: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=1)
    error: str | None = None
    completed_at: datetime = Field(default_factory=_utcnow)


__all__ = [
    "ModelConfig",
    "ResilienceConfig",
    "Result",
    "ResultStatus",
    "RunPlan",
    "Strategy",
    "StrategyConfig",
    "Task",
    "WorkersConfig",
]
