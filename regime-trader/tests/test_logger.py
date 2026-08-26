"""Tests for monitoring.logger."""

from __future__ import annotations

import json
import logging
import logging.handlers

import pytest

from monitoring.logger import BACKUP_COUNT, MAX_BYTES, StructuredLogger, get_logger


@pytest.fixture(autouse=True)
def _reset_something_else_logger():
    """conftest.py's _reset_regime_trader_loggers fixture resets the four
    standard logger names; this file also uses a non-standard name to
    test the fallback-to-main-log behavior, so it needs its own reset."""
    yield
    logger = logging.getLogger("regime_trader.something_else")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def _read_json_lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().strip().splitlines()]


@pytest.mark.parametrize(
    "name,expected_file",
    [("main", "main.log"), ("trades", "trades.log"), ("alerts", "alerts.log"), ("regime", "regime.log")],
)
def test_get_logger_routes_to_correct_file(tmp_path, name, expected_file) -> None:
    logger = get_logger(name, log_dir=tmp_path)
    logger.info("hello")

    assert (tmp_path / expected_file).exists()
    assert logger.name == f"regime_trader.{name}"


def test_get_logger_unknown_name_falls_back_to_main(tmp_path) -> None:
    logger = get_logger("something_else", log_dir=tmp_path)
    logger.info("hello")

    assert (tmp_path / "main.log").exists()


def test_get_logger_is_idempotent_no_duplicate_handlers(tmp_path) -> None:
    logger1 = get_logger("main", log_dir=tmp_path)
    logger2 = get_logger("main", log_dir=tmp_path)

    assert logger1 is logger2
    rotating_handlers = [h for h in logger1.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert len(rotating_handlers) == 1  # other tooling (e.g. pytest's own log capture) may add unrelated handlers


def test_get_logger_uses_rotating_file_handler_with_expected_limits(tmp_path) -> None:
    logger = get_logger("main", log_dir=tmp_path)
    rotating_handlers = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]

    assert len(rotating_handlers) == 1
    handler = rotating_handlers[0]
    assert handler.maxBytes == MAX_BYTES == 10 * 1024 * 1024
    assert handler.backupCount == BACKUP_COUNT == 30


def test_get_logger_does_not_propagate_to_root(tmp_path) -> None:
    logger = get_logger("main", log_dir=tmp_path)
    assert logger.propagate is False


def test_structured_logger_writes_json_lines_to_each_file(tmp_path) -> None:
    sl = StructuredLogger(log_dir=tmp_path)

    sl.log_trade("SPY", "buy", {"qty": 100, "price": 520.3})
    sl.log_regime_change("SPY", "BEAR", "BULL")
    sl.log_alert("circuit_breaker", "Daily DD halt", severity="critical")
    sl.log_error("boom", ValueError("bad"))

    trades = _read_json_lines(tmp_path / "trades.log")
    assert trades[0]["event"] == "trade"
    assert trades[0]["symbol"] == "SPY"
    assert trades[0]["qty"] == 100

    regime = _read_json_lines(tmp_path / "regime.log")
    assert regime[0]["event"] == "regime_change"
    assert regime[0]["old_regime"] == "BEAR"
    assert regime[0]["new_regime"] == "BULL"

    alerts = _read_json_lines(tmp_path / "alerts.log")
    assert alerts[0]["event"] == "alert"
    assert alerts[0]["severity"] == "critical"

    main = _read_json_lines(tmp_path / "main.log")
    assert main[0]["event"] == "error"
    assert main[0]["exception"] == "ValueError: bad"


def test_structured_logger_every_entry_has_a_timestamp(tmp_path) -> None:
    sl = StructuredLogger(log_dir=tmp_path)
    sl.log_info("startup")

    entry = _read_json_lines(tmp_path / "main.log")[0]
    assert "timestamp" in entry
    assert "T" in entry["timestamp"]  # ISO 8601


def test_structured_logger_context_defaults_to_none(tmp_path) -> None:
    sl = StructuredLogger(log_dir=tmp_path)
    sl.log_info("startup")

    entry = _read_json_lines(tmp_path / "main.log")[0]
    for field in ("regime", "probability", "equity", "positions", "daily_pnl"):
        assert entry[field] is None


def test_structured_logger_context_is_auto_attached_to_every_entry(tmp_path) -> None:
    sl = StructuredLogger(log_dir=tmp_path)
    sl.set_context(regime="BULL", probability=0.9, equity=100_000.0, positions=["SPY"], daily_pnl=340.0)

    sl.log_trade("SPY", "buy", {})
    sl.log_alert("test", "msg")

    for path in (tmp_path / "trades.log", tmp_path / "alerts.log"):
        entry = _read_json_lines(path)[0]
        assert entry["regime"] == "BULL"
        assert entry["probability"] == 0.9
        assert entry["equity"] == 100_000.0
        assert entry["positions"] == ["SPY"]
        assert entry["daily_pnl"] == 340.0


def test_structured_logger_context_ignores_unknown_fields(tmp_path) -> None:
    sl = StructuredLogger(log_dir=tmp_path)
    sl.set_context(regime="BULL", not_a_real_field="should be dropped")

    sl.log_info("startup")

    entry = _read_json_lines(tmp_path / "main.log")[0]
    assert "not_a_real_field" not in entry
    assert entry["regime"] == "BULL"


def test_structured_logger_log_error_without_exception(tmp_path) -> None:
    sl = StructuredLogger(log_dir=tmp_path)
    sl.log_error("something went wrong")

    entry = _read_json_lines(tmp_path / "main.log")[0]
    assert entry["message"] == "something went wrong"
    assert "exception" not in entry
