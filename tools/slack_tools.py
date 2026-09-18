"""
AutoPR - Role 2: Slack Notification Tools

Sends a concise result notification to a Slack Incoming Webhook.
"""

import json
import os
from typing import Dict, Any

import requests


class SlackToolError(Exception):
    """Raised when a Slack notification cannot be delivered."""


def send_slack_notification(
    message: str,
    webhook_url: str | None = None,
    timeout: int = 10,
) -> Dict[str, Any]:
    """
    Send a Slack webhook notification.

    The webhook URL can be passed explicitly or read from SLACK_WEBHOOK_URL.
    Returns a deterministic result for the agent.
    """
    webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")

    if not webhook_url:
        raise SlackToolError(
            "SLACK_WEBHOOK_URL is not configured."
        )

    if not message or not message.strip():
        raise SlackToolError("Slack message cannot be empty.")

    payload = {"text": message}

    try:
        response = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise SlackToolError(
            f"Slack request failed: {exc}"
        ) from exc

    if response.status_code < 200 or response.status_code >= 300:
        raise SlackToolError(
            f"Slack returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    return {
        "success": True,
        "status_code": response.status_code,
        "message": message,
    }


def build_completion_message(
    work_item_id: str,
    pr_url: str,
    tests_passed: bool,
    files_changed: int | None = None,
    retries: int | None = None,
) -> str:
    """Create a consistent AutoPR completion message."""
    validation = "PASSED" if tests_passed else "FAILED"

    lines = [
        f"*AutoPR completed Work Item {work_item_id}*",
        f"PR: {pr_url}",
        f"Validation: {validation}",
    ]

    if files_changed is not None:
        lines.append(f"Files changed: {files_changed}")

    if retries is not None:
        lines.append(f"Self-debug retries: {retries}")

    return "\n".join(lines)
