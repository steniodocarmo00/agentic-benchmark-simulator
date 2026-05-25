"""Integration test: Task survives a full SQS round-trip via ElasticMQ.

Scope:
  * Send a ``Task`` to a queue.
  * Long-poll, receive it back, validate the Pydantic contract.
  * Ack (delete) and confirm the queue is empty.
  * Verify the DLQ RedrivePolicy is wired on the source queue.

Out of scope (deliberately): any LLM / inference logic.

Pre-requisite:
  * ElasticMQ must be reachable at ``localhost:9324``
    (``bash scripts/setup_infra.sh`` brings it up).
"""

from __future__ import annotations

import json
import socket
from uuid import uuid4

import pytest

from swarm.common.contracts import Strategy, Task
from swarm.messaging.sqs_client import SQSClient

ELASTICMQ_HOST = "localhost"
ELASTICMQ_PORT = 9324
ELASTICMQ_URL = f"http://{ELASTICMQ_HOST}:{ELASTICMQ_PORT}"


def _elasticmq_reachable() -> bool:
    try:
        with socket.create_connection((ELASTICMQ_HOST, ELASTICMQ_PORT), timeout=1):
            return True
    except OSError:
        return False


# Skip the whole module if the local infra is not running. Keeps `pytest`
# green on machines where the user hasn't booted docker yet.
pytestmark = pytest.mark.skipif(
    not _elasticmq_reachable(),
    reason=(
        f"ElasticMQ not reachable on {ELASTICMQ_HOST}:{ELASTICMQ_PORT}. "
        "Run `bash scripts/setup_infra.sh` first."
    ),
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def sqs() -> SQSClient:
    return SQSClient(endpoint_url=ELASTICMQ_URL)


@pytest.fixture()
def queues(sqs: SQSClient) -> tuple[str, str]:
    """Create a unique (tasks, dlq) pair per test and tear them down."""
    suffix = uuid4().hex[:8]
    tasks_q = f"it-tasks-{suffix}"
    dlq_q = f"it-tasks-dlq-{suffix}"

    sqs.ensure_queue(dlq_q)
    sqs.ensure_queue(tasks_q)
    sqs.configure_dlq(tasks_q, dlq_q, max_receive_count=3)

    yield tasks_q, dlq_q

    sqs.delete_queue(tasks_q)
    sqs.delete_queue(dlq_q)


def _sample_task() -> Task:
    return Task(
        run_id="exp-it-001",
        problem_id="gsm8k-0042",
        strategy=Strategy.CHAIN_OF_THOUGHT,
        sample_idx=0,
        question="If John has 5 apples and eats 2, how many remain?",
        prompt_id="cot.v1",
        prompt_version=1,
        prompt_hash="sha256:deadbeef",
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_task_roundtrip_send_receive_ack(sqs: SQSClient, queues: tuple[str, str]) -> None:
    tasks_q, _ = queues
    sent = _sample_task()

    # 1. Send.
    message_id = sqs.send(tasks_q, sent)
    assert message_id, "SQS must return a non-empty MessageId"

    # 2. Receive (long-poll up to 5s).
    received = sqs.receive(tasks_q, Task, max_messages=1, wait_seconds=5)
    assert len(received) == 1, "exactly one message expected on the queue"
    envelope = received[0]

    # 3. Contract verification — every field round-trips intact.
    body = envelope.body
    assert isinstance(body, Task)
    assert body.task_id == sent.task_id
    assert body.run_id == sent.run_id
    assert body.problem_id == sent.problem_id
    assert body.strategy is Strategy.CHAIN_OF_THOUGHT
    assert body.sample_idx == sent.sample_idx
    assert body.question == sent.question
    assert body.prompt_id == sent.prompt_id
    assert body.prompt_version == sent.prompt_version
    assert body.prompt_hash == sent.prompt_hash
    # Datetime survives ISO-8601 round-trip with tz info preserved.
    assert body.created_at == sent.created_at
    assert body.created_at.tzinfo is not None

    # 4. Ack — delete the message.
    sqs.ack(envelope)

    # 5. Queue must be empty after ack (short poll, 1s).
    leftovers = sqs.receive(tasks_q, Task, max_messages=1, wait_seconds=1)
    assert leftovers == [], "queue should be drained after ack"


def test_task_frozen_message_is_immutable() -> None:
    """Sanity check: Task contract must reject in-place mutation."""
    task = _sample_task()
    with pytest.raises(Exception):  # pydantic.ValidationError in v2 (frozen)
        task.run_id = "tampered"  # type: ignore[misc]


def test_dlq_redrive_policy_is_wired(sqs: SQSClient, queues: tuple[str, str]) -> None:
    tasks_q, dlq_q = queues
    attrs = sqs.get_attributes(tasks_q, ["RedrivePolicy"])
    assert "RedrivePolicy" in attrs, "source queue must expose a RedrivePolicy"

    policy = json.loads(attrs["RedrivePolicy"])
    # SQS returns maxReceiveCount as a string; tolerate both forms.
    assert str(policy["maxReceiveCount"]) == "3"
    assert policy["deadLetterTargetArn"].endswith(f":{dlq_q}"), (
        f"redrive target {policy['deadLetterTargetArn']!r} does not point at {dlq_q!r}"
    )


def test_invalid_payload_raises_on_receive(sqs: SQSClient, queues: tuple[str, str]) -> None:
    """A malformed body must fail at parse time, not silently propagate."""
    tasks_q, _ = queues
    url = sqs.get_queue_url(tasks_q)
    sqs._client.send_message(QueueUrl=url, MessageBody='{"not":"a task"}')

    with pytest.raises(Exception):  # pydantic.ValidationError
        sqs.receive(tasks_q, Task, max_messages=1, wait_seconds=5)
