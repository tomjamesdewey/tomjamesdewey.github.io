"""Structured logging for regime-trader."""

from __future__ import annotations

import logging
from typing import Any, Optional


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get a configured structured logger for the given module name."""
    ...


class StructuredLogger:
    """Wraps a standard logger to emit structured (JSON-friendly) log records."""

    def __init__(self, name: str, level: str = "INFO") -> None:
        """Initialize the underlying logger."""
        ...

    def log_trade(self, symbol: str, action: str, details: dict[str, Any]) -> None:
        """Log a trade event with structured context."""
        ...

    def log_regime_change(self, symbol: str, old_regime: str, new_regime: str) -> None:
        """Log a regime change event."""
        ...

    def log_error(self, message: str, exc: Optional[Exception] = None) -> None:
        """Log an error, optionally including exception details."""
        ...
