"""Email/webhook alerts for critical events, with rate limiting."""

from __future__ import annotations

from enum import Enum
from typing import Optional


class AlertSeverity(Enum):
    """Severity level for an alert."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertManager:
    """Sends email/webhook alerts for critical bot events, rate-limited."""

    def __init__(
        self,
        email_config: Optional[dict] = None,
        webhook_config: Optional[dict] = None,
        rate_limit_minutes: int = 15,
    ) -> None:
        """Store alert channel configuration and rate limit."""
        ...

    def send_alert(
        self, message: str, severity: AlertSeverity = AlertSeverity.INFO
    ) -> None:
        """Send an alert through all configured channels, respecting rate limits."""
        ...

    def send_email(self, subject: str, body: str) -> None:
        """Send an alert via email."""
        ...

    def send_webhook(self, payload: dict) -> None:
        """Send an alert via webhook."""
        ...

    def is_rate_limited(self, alert_key: str) -> bool:
        """Check whether an alert of this type was sent too recently."""
        ...
