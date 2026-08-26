"""Alpaca API wrapper.

A thin, retrying wrapper around alpaca-py's ``TradingClient`` and
``StockHistoricalDataClient``. Owns credential handling (never hardcoded —
callers load them from ``.env``, see ``main.py``), paper/live endpoint
selection with an explicit confirmation gate for live trading, connection
health-checking, and exponential-backoff retry — which the rest of the
``broker``/``data`` package reuses via ``call_with_retry`` rather than each
re-implementing its own retry logic.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, TypeVar

from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"

#: Exact phrase an operator must type to arm live trading.
LIVE_TRADING_CONFIRMATION_PHRASE = "YES I UNDERSTAND THE RISKS"

DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE_SECONDS = 1.0

T = TypeVar("T")

#: Transient errors worth retrying with backoff (network issues, and any
#: Alpaca 5xx/429 surfaced as APIError — 4xx client errors like bad auth
#: or a rejected order are not retried, since retrying won't fix them).
_RETRYABLE_EXCEPTIONS = (RequestException, ConnectionError, TimeoutError)


class LiveTradingNotConfirmedError(RuntimeError):
    """Raised when live trading was requested but not explicitly confirmed."""


def _is_retryable_api_error(exc: APIError) -> bool:
    try:
        status_code = exc.status_code
    except Exception:  # noqa: BLE001 - a malformed error body must never break retry classification
        return False
    return status_code is not None and (status_code == 429 or 500 <= status_code < 600)


class AlpacaClient:
    """Wraps the Alpaca trading and historical-data API clients."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        *,
        confirm_live_trading: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
        run_health_check: bool = True,
    ) -> None:
        """Initialize the underlying Alpaca trading and data clients.

        If ``paper`` is False, live trading must be explicitly armed: pass
        ``confirm_live_trading="YES I UNDERSTAND THE RISKS"`` (e.g. from a
        CLI flag or already-confirmed config), or leave it unset to be
        prompted interactively on stdin. Anything else raises
        ``LiveTradingNotConfirmedError`` rather than silently trading live.
        """
        if not paper:
            self._require_live_trading_confirmation(confirm_live_trading)

        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.base_url = PAPER_BASE_URL if paper else LIVE_BASE_URL
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds

        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)

        if run_health_check:
            self._health_check()

    @staticmethod
    def _require_live_trading_confirmation(confirm_live_trading: Optional[str]) -> None:
        if confirm_live_trading == LIVE_TRADING_CONFIRMATION_PHRASE:
            return
        if confirm_live_trading is not None:
            raise LiveTradingNotConfirmedError(
                "Live trading confirmation phrase did not match. "
                f"Expected exactly: {LIVE_TRADING_CONFIRMATION_PHRASE!r}"
            )
        print("\N{WARNING SIGN}  LIVE TRADING MODE. Type 'YES I UNDERSTAND THE RISKS' to confirm.")
        response = input("> ").strip()
        if response != LIVE_TRADING_CONFIRMATION_PHRASE:
            raise LiveTradingNotConfirmedError("Live trading not confirmed; aborting startup.")

    # ------------------------------------------------------------------
    # Retry infrastructure, reused by order_executor/position_tracker/market_data
    # ------------------------------------------------------------------

    def call_with_retry(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call ``fn(*args, **kwargs)``, retrying transient failures with
        exponential backoff (``backoff_base_seconds * 2**attempt``)."""
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except APIError as exc:
                if not _is_retryable_api_error(exc):
                    raise
                last_exc = exc
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc

            if attempt < self.max_retries - 1:
                delay = self.backoff_base_seconds * (2**attempt)
                logger.warning(
                    "Alpaca API call failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt + 1,
                    self.max_retries,
                    last_exc,
                    delay,
                )
                time.sleep(delay)

        raise ConnectionError(f"Alpaca API unreachable after {self.max_retries} attempts") from last_exc

    def _health_check(self) -> None:
        """Verify connectivity on startup, with the same retry/backoff as
        any other call — raises if Alpaca is unreachable after all retries."""
        self.call_with_retry(self.trading_client.get_clock)
        logger.info("Alpaca connection healthy (%s)", "paper" if self.paper else "LIVE")

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        """Fetch current account information (equity, buying power, etc.)."""
        account = self.call_with_retry(self.trading_client.get_account)
        return account.model_dump()

    def get_positions(self) -> list[dict]:
        """Fetch all currently open positions."""
        positions = self.call_with_retry(self.trading_client.get_all_positions)
        return [p.model_dump() for p in positions]

    def get_order_history(self, status: str = "all", limit: int = 100) -> list[dict]:
        """Fetch recent orders. ``status`` is one of 'open', 'closed', 'all'."""
        request = GetOrdersRequest(status=QueryOrderStatus(status), limit=limit)
        orders = self.call_with_retry(self.trading_client.get_orders, filter=request)
        return [o.model_dump() for o in orders]

    def get_clock(self) -> dict:
        """Fetch current market clock (open/closed status, next open/close)."""
        clock = self.call_with_retry(self.trading_client.get_clock)
        return clock.model_dump()

    def is_market_open(self) -> bool:
        """Check whether the market is currently open."""
        return bool(self.get_clock()["is_open"])

    def get_available_margin(self) -> float:
        """Buying power currently available for new positions."""
        return float(self.get_account()["buying_power"])
