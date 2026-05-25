"""Environment-agnostic SQS client.

Talks to either:
  * **Local ElasticMQ** — set ``SQS_ENDPOINT_URL=http://localhost:9324``.
  * **Real AWS SQS** — leave ``SQS_ENDPOINT_URL`` unset; standard boto3
    credentials chain applies.

The same Python code runs unchanged in both cases. That is the whole
point of the abstraction.

This layer is intentionally THIN:
  * No retry logic here — boto3 already retries transient SQS errors.
    Application-level retries (e.g. inference) live in ``swarm.inference``.
  * No business logic — only ``send / receive / ack / wire-dlq / admin``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ReceivedMessage(Generic[T]):
    """Envelope returned by :meth:`SQSClient.receive`.

    Carries both the parsed Pydantic body and the SQS metadata the caller
    needs to ack (delete) or otherwise manage the message.
    """

    body: T
    receipt_handle: str
    queue_url: str
    message_id: str
    attributes: dict = field(default_factory=dict)


class SQSClient:
    """Thin wrapper around the boto3 SQS client."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        region: str = "us-east-1",
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        # When pointing at ElasticMQ / LocalStack, boto3 still demands SOME
        # credentials — even though the emulator ignores them. Inject
        # placeholders so the credential chain doesn't try to query IMDS.
        if endpoint_url is not None:
            access_key = access_key or "test"
            secret_key = secret_key or "test"

        self._client: BaseClient = boto3.client(
            "sqs",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self._url_cache: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls) -> "SQSClient":
        """Build a client from environment variables.

        Recognized:
          * ``SQS_ENDPOINT_URL`` — set for ElasticMQ; unset for AWS.
          * ``AWS_REGION`` — defaults to ``us-east-1``.
          * ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` — standard.
        """
        return cls(
            endpoint_url=os.getenv("SQS_ENDPOINT_URL") or None,
            region=os.getenv("AWS_REGION", "us-east-1"),
            access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

    # ------------------------------------------------------------------ #
    # Admin
    # ------------------------------------------------------------------ #

    def get_queue_url(self, name: str) -> str:
        """Resolve a queue name to its URL (cached)."""
        cached = self._url_cache.get(name)
        if cached is not None:
            return cached
        resp = self._client.get_queue_url(QueueName=name)
        url = resp["QueueUrl"]
        self._url_cache[name] = url
        return url

    def ensure_queue(self, name: str) -> str:
        """Idempotent ``CreateQueue``. Returns the queue URL."""
        try:
            return self.get_queue_url(name)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in ("AWS.SimpleQueueService.NonExistentQueue", "QueueDoesNotExist"):
                raise
        resp = self._client.create_queue(QueueName=name)
        url = resp["QueueUrl"]
        self._url_cache[name] = url
        return url

    def delete_queue(self, name: str) -> None:
        """Best-effort delete (used by test teardown). Silent if missing."""
        try:
            url = self.get_queue_url(name)
        except ClientError:
            return
        try:
            self._client.delete_queue(QueueUrl=url)
        finally:
            self._url_cache.pop(name, None)

    def get_attributes(
        self, name: str, attribute_names: list[str] | None = None
    ) -> dict[str, str]:
        """Return queue attributes (``RedrivePolicy``, ``QueueArn``, ...)."""
        url = self.get_queue_url(name)
        resp = self._client.get_queue_attributes(
            QueueUrl=url,
            AttributeNames=attribute_names or ["All"],
        )
        return resp.get("Attributes", {})

    def configure_dlq(
        self,
        source_queue: str,
        dlq_queue: str,
        max_receive_count: int = 3,
    ) -> None:
        """Wire ``source_queue`` → ``dlq_queue`` via SQS RedrivePolicy.

        After ``max_receive_count`` failed receives, SQS automatically
        moves the message to the DLQ. Required by the edital (item 4).
        """
        if max_receive_count < 1:
            raise ValueError("max_receive_count must be >= 1")

        src_url = self.get_queue_url(source_queue)
        dlq_arn = self.get_attributes(dlq_queue, ["QueueArn"])["QueueArn"]

        policy = json.dumps(
            {
                "deadLetterTargetArn": dlq_arn,
                "maxReceiveCount": str(max_receive_count),
            }
        )
        self._client.set_queue_attributes(
            QueueUrl=src_url,
            Attributes={"RedrivePolicy": policy},
        )
        logger.info(
            "DLQ wired: %s -> %s (maxReceiveCount=%d)",
            source_queue,
            dlq_queue,
            max_receive_count,
        )

    # ------------------------------------------------------------------ #
    # Data plane
    # ------------------------------------------------------------------ #

    def send(self, queue_name: str, payload: BaseModel) -> str:
        """Serialize ``payload`` to JSON and send. Returns the SQS MessageId."""
        url = self.get_queue_url(queue_name)
        resp = self._client.send_message(
            QueueUrl=url,
            MessageBody=payload.model_dump_json(),
        )
        return resp["MessageId"]

    def receive(
        self,
        queue_name: str,
        model_cls: type[T],
        max_messages: int = 1,
        wait_seconds: int = 10,
        visibility_timeout: int | None = None,
    ) -> list[ReceivedMessage[T]]:
        """Long-poll the queue and parse bodies as ``model_cls``.

        Returns an empty list on timeout. The caller is responsible for
        calling :meth:`ack` on every message that was processed
        successfully; un-acked messages reappear after the visibility
        timeout (and eventually move to the DLQ).
        """
        if not 1 <= max_messages <= 10:
            raise ValueError("max_messages must be between 1 and 10 (SQS limit)")
        if not 0 <= wait_seconds <= 20:
            raise ValueError("wait_seconds must be between 0 and 20 (SQS limit)")

        url = self.get_queue_url(queue_name)
        kwargs: dict = {
            "QueueUrl": url,
            "MaxNumberOfMessages": max_messages,
            "WaitTimeSeconds": wait_seconds,
            "AttributeNames": ["All"],
        }
        if visibility_timeout is not None:
            kwargs["VisibilityTimeout"] = visibility_timeout

        resp = self._client.receive_message(**kwargs)
        raw = resp.get("Messages", [])
        return [
            ReceivedMessage(
                body=model_cls.model_validate_json(m["Body"]),
                receipt_handle=m["ReceiptHandle"],
                queue_url=url,
                message_id=m["MessageId"],
                attributes=m.get("Attributes", {}),
            )
            for m in raw
        ]

    def ack(self, message: ReceivedMessage) -> None:
        """Delete a successfully processed message from the queue."""
        self._client.delete_message(
            QueueUrl=message.queue_url,
            ReceiptHandle=message.receipt_handle,
        )


__all__ = ["ReceivedMessage", "SQSClient"]
