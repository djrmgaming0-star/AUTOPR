import os
import requests


def send_slack_notification(message: str) -> None:
    """Send a message to Slack using an Incoming Webhook."""

    webhook_url = os.getenv("SLACK_WEBHOOK_URL")

    if not webhook_url:
        raise RuntimeError("SLACK_WEBHOOK_URL is not set")

    response = requests.post(
        webhook_url,
        json={"text": message},
        timeout=10,
    )

    response.raise_for_status()
