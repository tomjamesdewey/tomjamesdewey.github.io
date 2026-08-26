"""Tests for monitoring.alerts.AlertManager."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from monitoring.alerts import AlertManager, AlertSeverity
from monitoring.logger import StructuredLogger

EMAIL_CONFIG = {
    "smtp_host": "smtp.example.com",
    "smtp_port": 587,
    "username": "bot@example.com",
    "password": "secret",
    "to_address": "owner@example.com",
}
WEBHOOK_CONFIG = {"url": "https://hooks.example.com/hook"}


def _null_console() -> Console:
    return Console(file=io.StringIO(), width=100)


def make_manager(**overrides) -> AlertManager:
    kwargs = dict(rate_limit_minutes=15, console=_null_console())
    kwargs.update(overrides)
    return AlertManager(**kwargs)


# ----------------------------------------------------------------------
# Rate limiting
# ----------------------------------------------------------------------


def test_is_rate_limited_false_before_any_send() -> None:
    manager = make_manager()
    assert manager.is_rate_limited("some_key") is False


def test_send_alert_returns_true_then_false_within_window() -> None:
    manager = make_manager()
    assert manager.send_alert("first", alert_key="k") is True
    assert manager.is_rate_limited("k") is True
    assert manager.send_alert("second", alert_key="k") is False


def test_send_alert_not_rate_limited_after_window_elapses() -> None:
    from datetime import timedelta, timezone, datetime as dt

    manager = make_manager(rate_limit_minutes=15)
    manager.send_alert("first", alert_key="k")
    # Simulate 16 minutes having passed since the last send.
    manager._last_sent["k"] = dt.now(timezone.utc) - timedelta(minutes=16)

    assert manager.is_rate_limited("k") is False
    assert manager.send_alert("second", alert_key="k") is True


def test_different_alert_keys_are_independently_rate_limited() -> None:
    manager = make_manager()
    assert manager.send_alert("a", alert_key="key_a") is True
    assert manager.send_alert("b", alert_key="key_b") is True  # different key, not limited


def test_alert_key_defaults_to_message_when_not_given() -> None:
    manager = make_manager()
    assert manager.send_alert("same message") is True
    assert manager.send_alert("same message") is False
    assert manager.send_alert("different message") is True


# ----------------------------------------------------------------------
# Delivery
# ----------------------------------------------------------------------


def test_send_alert_writes_to_structured_logger() -> None:
    sl = MagicMock(spec=StructuredLogger)
    manager = make_manager(structured_logger=sl)

    manager.send_alert("something happened", AlertSeverity.WARNING, alert_key="k", symbol="SPY")

    sl.log_alert.assert_called_once_with(alert_type="k", message="something happened", severity="warning", symbol="SPY")


def test_send_alert_prints_to_console() -> None:
    console = _null_console()
    manager = make_manager(console=console)

    manager.send_alert("something happened", AlertSeverity.CRITICAL, alert_key="k")

    output = console.file.getvalue()
    assert "something happened" in output
    assert "CRITICAL" in output


def test_send_alert_without_email_or_webhook_config_does_not_call_them() -> None:
    manager = make_manager()
    with patch.object(manager, "send_email") as mock_email, patch.object(manager, "send_webhook") as mock_webhook:
        manager.send_alert("msg")
    assert not mock_email.called
    assert not mock_webhook.called


def test_send_alert_calls_email_and_webhook_when_configured() -> None:
    manager = make_manager(email_config=EMAIL_CONFIG, webhook_config=WEBHOOK_CONFIG)
    with patch.object(manager, "send_email") as mock_email, patch.object(manager, "send_webhook") as mock_webhook:
        manager.send_alert("msg", AlertSeverity.CRITICAL, alert_key="k")
    mock_email.assert_called_once()
    mock_webhook.assert_called_once()


def test_send_alert_email_failure_does_not_prevent_delivery_elsewhere() -> None:
    manager = make_manager(email_config=EMAIL_CONFIG, webhook_config=WEBHOOK_CONFIG)
    with patch.object(manager, "send_email", side_effect=RuntimeError("smtp down")), patch.object(
        manager, "send_webhook"
    ) as mock_webhook:
        sent = manager.send_alert("msg", alert_key="k")
    assert sent is True  # overall send_alert still succeeds
    mock_webhook.assert_called_once()


def test_send_email_requires_config() -> None:
    manager = make_manager()
    with pytest.raises(RuntimeError):
        manager.send_email("subject", "body")


def test_send_email_uses_smtp_starttls_login_and_send() -> None:
    manager = make_manager(email_config=EMAIL_CONFIG)
    with patch("smtplib.SMTP") as mock_smtp_cls:
        manager.send_email("subject", "body")

    instance = mock_smtp_cls.return_value.__enter__.return_value
    instance.starttls.assert_called_once()
    instance.login.assert_called_once_with(EMAIL_CONFIG["username"], EMAIL_CONFIG["password"])
    instance.send_message.assert_called_once()


def test_send_webhook_requires_config() -> None:
    manager = make_manager()
    with pytest.raises(RuntimeError):
        manager.send_webhook({"a": 1})


def test_send_webhook_posts_json_to_configured_url() -> None:
    manager = make_manager(webhook_config=WEBHOOK_CONFIG)
    mock_response = MagicMock(status=200)
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_response
        manager.send_webhook({"a": 1})

    request = mock_urlopen.call_args[0][0]
    assert request.full_url == WEBHOOK_CONFIG["url"]
    assert request.get_method() == "POST"
    assert request.data == b'{"a": 1}'


def test_send_webhook_raises_on_http_error_status() -> None:
    manager = make_manager(webhook_config=WEBHOOK_CONFIG)
    mock_response = MagicMock(status=500)
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_response
        with pytest.raises(RuntimeError):
            manager.send_webhook({"a": 1})


# ----------------------------------------------------------------------
# Named triggers
# ----------------------------------------------------------------------


def test_alert_regime_change_message_and_key() -> None:
    manager = make_manager()
    with patch.object(manager, "send_alert", wraps=manager.send_alert) as spy:
        manager.alert_regime_change("SPY", "BEAR", "BULL")
    spy.assert_called_once()
    args, kwargs = spy.call_args
    assert "SPY" in args[0]
    assert "BEAR" in args[0]
    assert "BULL" in args[0]
    assert kwargs["alert_key"] == "regime_change:SPY"
    assert args[1] == AlertSeverity.WARNING


def test_alert_circuit_breaker_is_critical() -> None:
    manager = make_manager()
    with patch.object(manager, "send_alert", wraps=manager.send_alert) as spy:
        manager.alert_circuit_breaker("daily_halt", 0.035)
    args, kwargs = spy.call_args
    assert kwargs["alert_key"] == "circuit_breaker:daily_halt"
    assert args[1] == AlertSeverity.CRITICAL
    assert "3.50%" in args[0]


def test_alert_large_pnl_direction_wording() -> None:
    manager = make_manager()
    with patch.object(manager, "send_alert", wraps=manager.send_alert) as spy:
        manager.alert_large_pnl("SPY", -0.06, 0.05)
    assert "loss" in spy.call_args[0][0]

    with patch.object(manager, "send_alert", wraps=manager.send_alert) as spy2:
        manager.alert_large_pnl("QQQ", 0.06, 0.05)
    assert "gain" in spy2.call_args[0][0]


def test_alert_data_feed_down_key_is_per_symbol() -> None:
    manager = make_manager()
    assert manager.alert_data_feed_down("SPY") is True
    assert manager.alert_data_feed_down("SPY") is False  # rate limited
    assert manager.alert_data_feed_down("QQQ") is True  # different symbol


def test_alert_api_lost_uses_fixed_key() -> None:
    manager = make_manager()
    assert manager.alert_api_lost() is True
    assert manager.alert_api_lost() is False


def test_alert_hmm_retrained_is_info_severity() -> None:
    manager = make_manager()
    with patch.object(manager, "send_alert", wraps=manager.send_alert) as spy:
        manager.alert_hmm_retrained("SPY", 4)
    args, kwargs = spy.call_args
    assert args[1] == AlertSeverity.INFO
    assert "4 regimes" in args[0]


def test_alert_flicker_exceeded_message() -> None:
    manager = make_manager()
    with patch.object(manager, "send_alert", wraps=manager.send_alert) as spy:
        manager.alert_flicker_exceeded("SPY", 6, 4)
    args, kwargs = spy.call_args
    assert "6" in args[0] and "4" in args[0]
    assert kwargs["alert_key"] == "flicker_exceeded:SPY"
