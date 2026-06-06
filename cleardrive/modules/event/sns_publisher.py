import json
from datetime import datetime, timezone
from typing import Any

import boto3

from cleardrive.core.config import DEFAULT_AWS_REGION, env_optional_str, env_str

_AWS_CREDENTIALS_HELP = (
    "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env, "
    "or configure the AWS CLI (~/.aws/credentials)."
)


class SnsPublisher:
    """Publishes license plate events to an AWS SNS topic."""

    def __init__(
        self,
        topic_arn: str | None = None,
        region: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
    ) -> None:
        self.topic_arn = topic_arn or env_optional_str("SNS_TOPIC_ARN")
        self.region = region or _resolve_aws_region()
        self.access_key_id = access_key_id or env_optional_str("AWS_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or env_optional_str("AWS_SECRET_ACCESS_KEY")
        self.session_token = session_token or env_optional_str("AWS_SESSION_TOKEN")
        self._client: Any | None = None

    def setup(self) -> None:
        if self._client is not None:
            return

        if not self.topic_arn:
            raise ValueError("SNS_TOPIC_ARN is required to publish events to SNS")

        session = boto3.Session(
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            aws_session_token=self.session_token,
            region_name=self.region,
        )
        if session.get_credentials() is None:
            raise ValueError(f"AWS credentials are required to publish SNS events. {_AWS_CREDENTIALS_HELP}")

        self._client = session.client("sns")

    def teardown(self) -> None:
        self._client = None

    def publish(self, plate: str, metadata: dict[str, Any] | None = None) -> str:
        """Publish a whitelisted plate event and return the SNS message ID."""
        if self._client is None:
            self.setup()

        assert self._client is not None

        message = {
            "plate": plate,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }

        response = self._client.publish(
            TopicArn=self.topic_arn,
            Message=json.dumps(message),
            Subject=f"ClearDrive: {plate}",
        )
        return response["MessageId"]


def _resolve_aws_region() -> str:
    for key in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        value = env_optional_str(key)
        if value is not None:
            return value
    return env_str("AWS_REGION", DEFAULT_AWS_REGION)
