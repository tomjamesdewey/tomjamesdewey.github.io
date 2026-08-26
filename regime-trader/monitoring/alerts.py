"""Email/webhook alerts for critical events, with rate limiting.

Every alert is delivered to the console and to alerts.log (via
``monitoring.logger.StructuredLogger``, when one is supplied); email and
webhook delivery are optional, enabled only when their respective config
dicts are provided. Each alert *type* (regime change for a given symbol,
a specific circuit breaker, etc.) is independently rate-limited to at
most one delivery per ``rate_limit_minutes``.
"""

from __future__ import annotations

import json
import logging
import smtplib
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Optional

from rich.console import Console

logger = logging.getLogger(__name__)

DEFAULT_RATE_LIMIT_MINUTES = 15
WEBHOOK_TIMEOUT_SECONDS = 10


class AlertSeverity(Enum):
    """Severity level for an alert."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


_SEVERITY_STYLE = {
    AlertSeverity.INFO: "cyan",
    AlertSeverity.WARNING: "yellow",
    AlertSeverity.CRITICAL: "bold red",
}


class AlertManager:
    """Sends console/log/email/webhook alerts for critical bot events, rate-limited."""

    def __init__(
        self,
        email_config: Optional[dict] = None,
        webhook_config: Optional[dict] = None,
        rate_limit_minutes: int = DEFAULT_RATE_LIMIT_MINUTES,
        console: Optional[Console] = None,
        structured_logger: Optional[Any] = None,
    ) -> None:
        """Store alert channel configuration and rate limit.

        ``email_config`` (if set): {smtp_host, smtp_port, username,
        password, to_address}. ``webhook_config`` (if set): {url}.
        ``structured_logger`` is a ``monitoring.logger.StructuredLogger``
        used to also write every alert to alerts.log; optional.
        """
        self.email_config = email_config
        self.webhook_config = webhook_config
        self.rate_limit_minutes = rate_limit_minutes
        self.console = console or Console()
        self.structured_logger = structured_logger
        self._last_sent: dict[str, datetime] = {}

    def is_rate_limited(self, alert_key: str) -> bool:
        """Check whether an alert of this type was sent too recently."""
        last_sent = self._last_sent.get(alert_key)
        if last_sent is None:
            return False
        return datetime.now(timezone.utc) - last_sent < timedelta(minutes=self.rate_limit_minutes)

    def send_alert(
        self,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        alert_key: Optional[str] = None,
        **extra: Any,
    ) -> bool:
        """Send an alert through all configured channels, respecting rate
        limits. Returns True if it was actually sent (False if rate-limited)."""
        alert_key = alert_key or message
        if self.is_rate_limited(alert_key):
            return False
        self._last_sent[alert_key] = datetime.now(timezone.utc)

        style = _SEVERITY_STYLE[severity]
        self.console.print(f"[{style}]\N{BELL} [{severity.value.upper()}] {message}[/{style}]")

        if self.structured_logger is not None:
            self.structured_logger.log_alert(alert_type=alert_key, message=message, severity=severity.value, **extra)

        if self.email_config:
            try:
                self.send_email(subject=f"[regime-trader] {severity.value.upper()}: {alert_key}", body=message)
            except Exception:
                logger.exception("Failed to send alert email for %s", alert_key)

        if self.webhook_config:
            try:
                self.send_webhook({"alert_key": alert_key, "message": message, "severity": severity.value, **extra})
            except Exception:
                logger.exception("Failed to send alert webhook for %s", alert_key)

        return True

    def send_email(self, subject: str, body: str) -> None:
        """Send an alert via email."""
        if not self.email_config:
            raise RuntimeError("send_email called without email_config set")

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = self.email_config["username"]
        msg["To"] = self.email_config["to_address"]

        with smtplib.SMTP(self.email_config["smtp_host"], self.email_config["smtp_port"], timeout=WEBHOOK_TIMEOUT_SECONDS) as smtp:
            smtp.starttls()
            smtp.login(self.email_config["username"], self.email_config["password"])
            smtp.send_message(msg)

    def send_webhook(self, payload: dict) -> None:
        """Send an alert via webhook (JSON POST)."""
        if not self.webhook_config:
            raise RuntimeError("send_webhook called without webhook_config set")

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.webhook_config["url"], data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=WEBHOOK_TIMEOUT_SECONDS) as response:
                if response.status >= 400:
                    raise RuntimeError(f"Webhook returned HTTP {response.status}")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Webhook request failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Named triggers
    # ------------------------------------------------------------------

    def alert_regime_change(self, symbol: str, old_regime: str, new_regime: str) -> bool:
        return self.send_alert(
            f"{symbol}: regime changed {old_regime} -> {new_regime}",
            AlertSeverity.WARNING,
            alert_key=f"regime_change:{symbol}",
        )

    def alert_circuit_breaker(self, breaker_type: str, drawdown_pct: float) -> bool:
        return self.send_alert(
            f"Circuit breaker fired: {breaker_type} (drawdown {drawdown_pct:.2%})",
            AlertSeverity.CRITICAL,
            alert_key=f"circuit_breaker:{breaker_type}",
        )

    def alert_large_pnl(self, symbol: str, pnl_pct: float, threshold_pct: float) -> bool:
        direction = "gain" if pnl_pct >= 0 else "loss"
        return self.send_alert(
            f"{symbol}: large {direction} {pnl_pct:+.2%} (threshold {threshold_pct:.2%})",
            AlertSeverity.WARNING,
            alert_key=f"large_pnl:{symbol}",
        )

    def alert_data_feed_down(self, symbol: str) -> bool:
        return self.send_alert(
            f"{symbol}: data feed appears down", AlertSeverity.CRITICAL, alert_key=f"data_feed_down:{symbol}"
        )

    def alert_api_lost(self) -> bool:
        return self.send_alert("Alpaca API connection lost", AlertSeverity.CRITICAL, alert_key="api_lost")

    def alert_hmm_retrained(self, symbol: str, n_regimes: int) -> bool:
        return self.send_alert(
            f"{symbol}: HMM retrained ({n_regimes} regimes)",
            AlertSeverity.INFO,
            alert_key=f"hmm_retrained:{symbol}",
        )

    def alert_flicker_exceeded(self, symbol: str, flicker_rate: int, threshold: int) -> bool:
        return self.send_alert(
            f"{symbol}: flicker rate {flicker_rate} exceeds threshold {threshold}",
            AlertSeverity.WARNING,
            alert_key=f"flicker_exceeded:{symbol}",
        )
