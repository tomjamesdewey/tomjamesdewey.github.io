"""Structured logging for regime-trader.

Every entry is a single JSON line, written to one of four rotating log
files — main.log, trades.log, alerts.log, regime.log — so each concern
can be tailed/grepped independently. Rotation is size-based (10MB per
file); ``backupCount=30`` keeps roughly a month of history alongside the
active file (an exact "10MB AND 30 calendar days" hybrid would need a
custom handler; this is the standard-library approximation of that
policy). Every entry carries the latest known regime/probability/equity/
positions/daily_pnl context automatically, via ``set_context``, so a
single log line is self-sufficient for post-hoc debugging without
cross-referencing other files.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

DEFAULT_LOG_DIR = Path("logs")
MAX_BYTES = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 30  # ~30 days of rotated history at typical log volume

#: The four dedicated log streams and the file each writes to.
LOG_FILES: dict[str, str] = {
    "main": "main.log",
    "trades": "trades.log",
    "alerts": "alerts.log",
    "regime": "regime.log",
}

#: Context fields auto-attached to every structured log entry.
CONTEXT_FIELDS = ("regime", "probability", "equity", "positions", "daily_pnl")


def get_logger(name: str = "main", level: str = "INFO", log_dir: str | Path = DEFAULT_LOG_DIR) -> logging.Logger:
    """Get a rotating-file logger writing JSON lines to ``LOG_FILES[name]``
    (falls back to main.log for any other ``name``). Idempotent: calling
    this again for the same name+log_dir returns the same logger without
    stacking a duplicate handler. Checks for our own handler specifically
    (by target file path) rather than "any handler at all", since other
    tooling (e.g. a test runner's log capture) may have already attached
    an unrelated handler to this logger name."""
    log_dir = Path(log_dir)
    filename = LOG_FILES.get(name, LOG_FILES["main"])
    target_path = str((log_dir / filename).resolve())

    logger = logging.getLogger(f"regime_trader.{name}")
    logger.setLevel(level)
    already_configured = any(
        isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", None) == target_path
        for h in logger.handlers
    )
    if not already_configured:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(log_dir / filename, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
        handler.setFormatter(logging.Formatter("%(message)s"))  # the message is already a JSON line
        logger.addHandler(handler)
        logger.propagate = False
    return logger


class StructuredLogger:
    """Emits structured JSON-line events across the four rotating log
    files, auto-attaching the latest known system context to every entry."""

    def __init__(self, name: str = "regime-trader", level: str = "INFO", log_dir: str | Path = DEFAULT_LOG_DIR) -> None:
        """Initialize the four underlying rotating-file loggers."""
        self.main_logger = get_logger("main", level, log_dir)
        self.trades_logger = get_logger("trades", level, log_dir)
        self.alerts_logger = get_logger("alerts", level, log_dir)
        self.regime_logger = get_logger("regime", level, log_dir)
        self._context: dict[str, Any] = {field: None for field in CONTEXT_FIELDS}

    def set_context(self, **fields: Any) -> None:
        """Update the context (regime/probability/equity/positions/daily_pnl)
        auto-attached to every subsequent log entry. Unknown keys are ignored."""
        self._context.update({k: v for k, v in fields.items() if k in self._context})

    def _payload(self, event: str, **extra: Any) -> dict:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **self._context,
            **extra,
        }

    def _emit(self, target: logging.Logger, level: int, event: str, **extra: Any) -> None:
        target.log(level, json.dumps(self._payload(event, **extra), default=str))

    def log_info(self, event: str, **extra: Any) -> None:
        """Log a general application event to main.log."""
        self._emit(self.main_logger, logging.INFO, event, **extra)

    def log_trade(self, symbol: str, action: str, details: dict[str, Any]) -> None:
        """Log a trade event with structured context, to trades.log."""
        self._emit(self.trades_logger, logging.INFO, "trade", symbol=symbol, action=action, **details)

    def log_regime_change(self, symbol: str, old_regime: str, new_regime: str) -> None:
        """Log a regime change event, to regime.log."""
        self._emit(
            self.regime_logger,
            logging.WARNING,
            "regime_change",
            symbol=symbol,
            old_regime=old_regime,
            new_regime=new_regime,
        )

    def log_alert(self, alert_type: str, message: str, severity: str = "info", **extra: Any) -> None:
        """Log an alert event, to alerts.log."""
        level = {"info": logging.INFO, "warning": logging.WARNING, "critical": logging.CRITICAL}.get(
            severity, logging.INFO
        )
        self._emit(self.alerts_logger, level, "alert", alert_type=alert_type, message=message, severity=severity, **extra)

    def log_error(self, message: str, exc: Optional[Exception] = None) -> None:
        """Log an error, optionally including exception details, to main.log."""
        extra: dict[str, Any] = {"message": message}
        if exc is not None:
            extra["exception"] = f"{type(exc).__name__}: {exc}"
        self._emit(self.main_logger, logging.ERROR, "error", **extra)
