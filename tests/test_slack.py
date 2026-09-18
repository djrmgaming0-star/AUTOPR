from unittest.mock import patch

from backend.integrations.slack import send_slack_notification


@patch("backend.integrations.slack.requests.post")
def test_send_slack_notification(mock_post, monkeypatch):
    monkeypatch.setenv(
        "SLACK_WEBHOOK_URL",
        "https://example.com/fake-webhook",
    )

    mock_post.return_value.raise_for_status.return_value = None

    send_slack_notification("Test AutoPR message")

    mock_post.assert_called_once_with(
        "https://example.com/fake-webhook",
        json={"text": "Test AutoPR message"},
        timeout=10,
    )
