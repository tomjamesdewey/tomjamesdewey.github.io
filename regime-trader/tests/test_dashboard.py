"""Tests for monitoring.dashboard.Dashboard."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from monitoring.dashboard import MAX_RECENT_SIGNALS, Dashboard, _risk_indicator


def _null_console() -> Console:
    return Console(file=io.StringIO(), width=100)


def _render_text(renderable, console: Console | None = None) -> str:
    console = console or _null_console()
    console.print(renderable)
    return console.file.getvalue()


SAMPLE_STATE = {
    "regimes": {"SPY": {"label": "BULL", "probability": 0.72, "consecutive_bars": 14, "flicker_rate": 1, "flicker_window": 20}},
    "portfolio": {"equity": 105230.0, "daily_pnl": 340.0, "daily_pnl_pct": 0.0032, "allocation_pct": 0.95, "leverage": 1.25},
    "positions": [
        {"symbol": "SPY", "direction": "LONG", "current_price": 520.30, "unrealized_pnl_pct": 0.012, "stop_level": 508.0, "holding_period": "3h"}
    ],
    "risk": {"daily_dd_pct": 0.003, "daily_dd_halt": 0.03, "peak_dd_pct": 0.012, "max_dd_from_peak": 0.10},
    "system": {"data_feed_ok": True, "api_ok": True, "api_latency_ms": 23, "hmm_age": "2d ago", "mode": "PAPER"},
}


def test_render_regime_status_shows_label_probability_stability_flicker() -> None:
    dashboard = Dashboard()
    text = _render_text(dashboard.render_regime_status(SAMPLE_STATE["regimes"]))

    assert "REGIME" in text
    assert "BULL" in text
    assert "72%" in text
    assert "14 bars" in text
    assert "1/20" in text


def test_render_regime_status_handles_no_data() -> None:
    dashboard = Dashboard()
    text = _render_text(dashboard.render_regime_status({}))
    assert "No regime data yet" in text


def test_render_account_summary_positive_and_negative_pnl() -> None:
    dashboard = Dashboard()
    positive = _render_text(dashboard.render_account_summary(SAMPLE_STATE["portfolio"]))
    assert "$105,230.00" in positive
    assert "+$340.00" in positive

    negative_account = {**SAMPLE_STATE["portfolio"], "daily_pnl": -200.0, "daily_pnl_pct": -0.002}
    negative = _render_text(dashboard.render_account_summary(negative_account))
    assert "-$200.00" in negative


def test_render_positions_shows_row_data() -> None:
    dashboard = Dashboard()
    text = _render_text(dashboard.render_positions(SAMPLE_STATE["positions"]))

    assert "SPY" in text
    assert "520.30" in text
    assert "508.00" in text
    assert "3h" in text


def test_render_positions_handles_empty() -> None:
    dashboard = Dashboard()
    text = _render_text(dashboard.render_positions([]))
    assert "No open positions" in text


def test_record_signal_appears_in_recent_signals() -> None:
    dashboard = Dashboard()
    dashboard.record_signal("14:30", "SPY", "Rebalance 60%->95% | Low vol")

    text = _render_text(dashboard.render_recent_signals())

    assert "14:30" in text
    assert "SPY" in text
    assert "Rebalance" in text


def test_recent_signals_caps_at_max_and_shows_newest_first() -> None:
    dashboard = Dashboard()
    for i in range(MAX_RECENT_SIGNALS + 5):
        dashboard.record_signal(f"t{i}", "SPY", f"signal {i}")

    assert len(dashboard.recent_signals) == MAX_RECENT_SIGNALS
    assert dashboard.recent_signals[0]["description"] == f"signal {MAX_RECENT_SIGNALS + 4}"  # most recent first


@pytest.mark.parametrize(
    "current,threshold,expected_color",
    [(0.005, 0.03, "green"), (0.022, 0.03, "yellow"), (0.035, 0.03, "red")],
)
def test_risk_indicator_color_thresholds(current, threshold, expected_color) -> None:
    result = _risk_indicator("Daily DD", current, threshold)
    assert f"[{expected_color}]" in result


def test_render_risk_status_shows_both_indicators() -> None:
    dashboard = Dashboard()
    text = _render_text(dashboard.render_risk_status(SAMPLE_STATE["risk"]))

    assert "Daily DD" in text
    assert "From Peak" in text


def test_render_system_status_shows_all_fields() -> None:
    dashboard = Dashboard()
    text = _render_text(dashboard.render_system_status(SAMPLE_STATE["system"]))

    assert "23ms" in text
    assert "2d ago" in text
    assert "PAPER" in text


def test_render_system_status_shows_failure_icons_when_down() -> None:
    dashboard = Dashboard()
    down_system = {**SAMPLE_STATE["system"], "data_feed_ok": False, "api_ok": False}
    text = _render_text(dashboard.render_system_status(down_system))
    assert "\N{CROSS MARK}" in text


def test_render_assembles_full_panel_without_raising() -> None:
    dashboard = Dashboard()
    dashboard.record_signal("14:30", "SPY", "Rebalance")
    panel = dashboard.render(SAMPLE_STATE)

    text = _render_text(panel)
    for section in ("REGIME", "PORTFOLIO", "POSITIONS", "RECENT SIGNALS", "RISK STATUS", "SYSTEM"):
        assert section in text


def test_render_handles_missing_state_keys_gracefully() -> None:
    dashboard = Dashboard()
    panel = dashboard.render({})  # nothing populated
    _render_text(panel)  # must not raise


def test_run_refreshes_via_state_provider_until_interrupted(monkeypatch) -> None:
    dashboard = Dashboard(refresh_seconds=0, console=_null_console())
    call_count = {"n": 0}

    def state_provider():
        call_count["n"] += 1
        return SAMPLE_STATE

    def fake_sleep(_seconds):
        if call_count["n"] >= 3:
            raise KeyboardInterrupt

    monkeypatch.setattr("monitoring.dashboard.time.sleep", fake_sleep)

    dashboard.run(state_provider)  # must not raise, must stop on KeyboardInterrupt

    assert call_count["n"] >= 3
