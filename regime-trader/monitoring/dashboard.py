"""Terminal-based live dashboard using rich.

Renders a single bordered panel with named sections — REGIME, PORTFOLIO,
POSITIONS, RECENT SIGNALS, RISK STATUS, SYSTEM — refreshed on an interval,
with color-coded (green/yellow/red) risk indicators.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Optional

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

DEFAULT_REFRESH_SECONDS = 5

#: Ratio of current-to-threshold drawdown above which a risk indicator
#: turns yellow ("approaching") vs. red ("breached", ratio >= 1.0).
WARNING_RATIO = 0.70

#: How many recent signals the RECENT SIGNALS section keeps.
MAX_RECENT_SIGNALS = 10


def _borderless_table(columns: int) -> Table:
    table = Table(box=None, show_header=False, expand=True, padding=(0, 1, 0, 0))
    for _ in range(columns):
        table.add_column(ratio=1)
    return table


def _section(title: str, table: Table) -> Table:
    """Wrap one section's table with a bold title row above it."""
    wrapper = Table(box=None, show_header=False, expand=True, padding=(0, 0))
    wrapper.add_column()
    wrapper.add_row(f"[bold cyan]{title}[/bold cyan]")
    wrapper.add_row(table)
    return wrapper


def _risk_indicator(label: str, current_pct: float, threshold_pct: float) -> str:
    """"<label>: current%/threshold% <icon>", green/yellow/red by proximity
    to the threshold."""
    ratio = abs(current_pct) / threshold_pct if threshold_pct else 0.0
    if ratio >= 1.0:
        icon, color = "\N{CROSS MARK}", "red"
    elif ratio >= WARNING_RATIO:
        icon, color = "\N{WARNING SIGN}", "yellow"
    else:
        icon, color = "\N{WHITE HEAVY CHECK MARK}", "green"
    return f"[{color}]{label}: {abs(current_pct):.1%}/{threshold_pct:.0%} {icon}[/{color}]"


def _status_icon(ok: bool) -> str:
    return "[green]\N{WHITE HEAVY CHECK MARK}[/green]" if ok else "[red]\N{CROSS MARK}[/red]"


class Dashboard:
    """Renders a live-refreshing terminal dashboard of bot state."""

    def __init__(self, refresh_seconds: int = DEFAULT_REFRESH_SECONDS, console: Optional[Console] = None) -> None:
        """Initialize the dashboard console and refresh interval."""
        self.refresh_seconds = refresh_seconds
        self.console = console or Console()
        self.recent_signals: deque[dict[str, Any]] = deque(maxlen=MAX_RECENT_SIGNALS)

    def record_signal(self, timestamp: str, symbol: str, description: str) -> None:
        """Track a signal for display in the RECENT SIGNALS section."""
        self.recent_signals.appendleft({"timestamp": timestamp, "symbol": symbol, "description": description})

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def render_regime_status(self, regime_by_symbol: dict[str, Any]) -> Table:
        """Render a table of current regime state per symbol."""
        table = _borderless_table(3)
        for symbol, info in regime_by_symbol.items():
            table.add_row(
                f"{symbol}: [bold]{info['label']}[/bold] ({info['probability']:.0%})",
                f"Stability: {info['consecutive_bars']} bars",
                f"Flicker: {info['flicker_rate']}/{info['flicker_window']}",
            )
        if not regime_by_symbol:
            table.add_row("No regime data yet", "", "")
        return _section("REGIME", table)

    def render_account_summary(self, account: dict[str, Any]) -> Table:
        """Render a summary panel of account equity, P&L, and exposure."""
        table = _borderless_table(2)
        daily_pnl = account.get("daily_pnl", 0.0)
        daily_pnl_pct = account.get("daily_pnl_pct", 0.0)
        color = "green" if daily_pnl >= 0 else "red"
        sign = "+" if daily_pnl >= 0 else "-"
        table.add_row(
            f"Equity: ${account.get('equity', 0.0):,.2f}",
            f"Daily: [{color}]{sign}${abs(daily_pnl):,.2f} ({sign}{abs(daily_pnl_pct):.2%})[/{color}]",
        )
        table.add_row(
            f"Allocation: {account.get('allocation_pct', 0.0):.0%}",
            f"Leverage: {account.get('leverage', 1.0):.2f}x",
        )
        return _section("PORTFOLIO", table)

    def render_positions(self, positions: list[dict[str, Any]]) -> Table:
        """Render a table of current open positions."""
        table = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1, 0, 0))
        for col in ("Symbol", "Dir", "Price", "P&L", "Stop", "Held"):
            table.add_column(col)
        for p in positions:
            pnl_pct = p.get("unrealized_pnl_pct", 0.0)
            pnl_color = "green" if pnl_pct >= 0 else "red"
            stop = p.get("stop_level")
            table.add_row(
                p["symbol"],
                p.get("direction", "LONG"),
                f"${p['current_price']:.2f}",
                f"[{pnl_color}]{pnl_pct:+.1%}[/{pnl_color}]",
                f"${stop:.2f}" if stop is not None else "n/a",
                p.get("holding_period", "n/a"),
            )
        if not positions:
            table.add_row("No open positions", "", "", "", "", "")
        return _section("POSITIONS", table)

    def render_recent_signals(self) -> Table:
        """Render a table of the most recently generated signals."""
        table = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1, 0, 0))
        for col in ("Time", "Symbol", "Signal"):
            table.add_column(col)
        for sig in self.recent_signals:
            table.add_row(sig["timestamp"], sig["symbol"], sig["description"])
        if not self.recent_signals:
            table.add_row("No recent signals", "", "")
        return _section("RECENT SIGNALS", table)

    def render_risk_status(self, risk: dict[str, Any]) -> Table:
        """Render color-coded daily/peak drawdown indicators."""
        table = _borderless_table(2)
        table.add_row(
            _risk_indicator("Daily DD", risk.get("daily_dd_pct", 0.0), risk.get("daily_dd_halt", 1.0)),
            _risk_indicator("From Peak", risk.get("peak_dd_pct", 0.0), risk.get("max_dd_from_peak", 1.0)),
        )
        return _section("RISK STATUS", table)

    def render_system_status(self, system: dict[str, Any]) -> Table:
        """Render data-feed/API/HMM-freshness/mode status."""
        table = _borderless_table(4)
        api_latency = system.get("api_latency_ms")
        api_str = f"API: {_status_icon(system.get('api_ok', True))}" + (
            f" {api_latency:.0f}ms" if api_latency is not None else ""
        )
        table.add_row(
            f"Data: {_status_icon(system.get('data_feed_ok', True))}",
            api_str,
            f"HMM: {system.get('hmm_age', 'n/a')}",
            system.get("mode", ""),
        )
        return _section("SYSTEM", table)

    # ------------------------------------------------------------------
    # Assembly / live refresh
    # ------------------------------------------------------------------

    def render(self, state: dict[str, Any]) -> Panel:
        """Assemble the full dashboard panel from a state dict with keys
        'regimes', 'portfolio', 'positions', 'risk', 'system'."""
        group = Group(
            self.render_regime_status(state.get("regimes", {})),
            self.render_account_summary(state.get("portfolio", {})),
            self.render_positions(state.get("positions", [])),
            self.render_recent_signals(),
            self.render_risk_status(state.get("risk", {})),
            self.render_system_status(state.get("system", {})),
        )
        return Panel(group, title="regime-trader", border_style="cyan")

    def run(self, state_provider: Callable[[], dict[str, Any]]) -> None:
        """Start the live-refreshing dashboard loop: calls
        ``state_provider()`` every ``refresh_seconds`` and re-renders.
        Blocking — runs until interrupted (Ctrl-C)."""
        with Live(self.render(state_provider()), console=self.console, screen=False) as live:
            try:
                while True:
                    time.sleep(self.refresh_seconds)
                    live.update(self.render(state_provider()))
            except KeyboardInterrupt:
                pass
