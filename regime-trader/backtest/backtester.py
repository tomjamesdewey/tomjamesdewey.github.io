"""Walk-forward allocation backtester.

This is an ALLOCATION-BASED walk-forward backtester: it does not track
individual trade entries/exits. At each out-of-sample bar it decides a
target *portfolio allocation* (a fraction of equity, possibly >1.0 when
leveraged) from the currently detected volatility regime, and only
rebalances into that target when it has drifted meaningfully from the
allocation currently held. This is how real systematic strategies work.

Rolling windows (fixed-size, sliding forward by ``step_size``):
    IS (in-sample):  ``train_window`` bars -> HMM training + model selection
    OOS (out-of-sample): ``test_window`` bars -> walked bar-by-bar, causally

For each window: a fresh HMM is trained on the window's IS feature slice
(BIC model selection, see ``core.hmm_engine``), vol-rankings are derived
from that model's ``regime_info`` and used to build a fresh
``StrategyOrchestrator`` (see ``core.regime_strategies`` — buckets by
volatility, not by label). The OOS segment is then walked one bar at a
time, using only data up to and including that bar (the forward-algorithm
filtered inference from ``HMMEngine.predict_regime_filtered`` guarantees
no look-ahead bias) to decide that bar's target allocation.

The per-bar target-allocation *decisions* across every window are then fed,
as one continuous series, into ``simulate_allocation_series`` — the single
place the exact cash/shares allocation math lives, also reused by the
benchmark generators in ``backtest.performance``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import StrategyOrchestrator
from core.risk_manager import RiskManager
from data.feature_engineering import FeatureEngineer

#: Bars between a rebalance decision (at bar N's close) and its execution
#: (at bar N+1's open).
FILL_DELAY_BARS = 1


@dataclass
class BacktestResult:
    """Results of a backtest run."""

    equity_curve: pd.Series
    trades: pd.DataFrame
    regime_history: pd.DataFrame
    metadata: dict = field(default_factory=dict)


def _clone_hmm_engine(template: HMMEngine) -> HMMEngine:
    """A fresh, unfit HMMEngine sharing ``template``'s configuration."""
    return HMMEngine(
        n_candidates=list(template.n_candidates),
        n_init=template.n_init,
        covariance_type=template.covariance_type,
        min_train_bars=template.min_train_bars,
        stability_bars=template.stability_bars,
        flicker_window=template.flicker_window,
        flicker_threshold=template.flicker_threshold,
        min_confidence=template.min_confidence,
    )


def generate_walk_forward_windows(
    n_rows: int, train_window: int, test_window: int, step_size: int
) -> list[tuple[int, int, int, int]]:
    """Positional ``(is_start, is_end, oos_start, oos_end)`` tuples tiling
    ``n_rows`` rows of an already-valid (NaN-free) feature matrix.

    The IS window is fixed-size and slides forward by ``step_size`` each
    iteration (a genuine *rolling*, not expanding, window — this backtester
    always trains fresh on exactly the last ``train_window`` bars). The
    final OOS segment is truncated to whatever data remains.
    """
    windows: list[tuple[int, int, int, int]] = []
    is_start = 0
    while True:
        is_end = is_start + train_window
        oos_start = is_end
        if oos_start >= n_rows:
            break
        oos_end = min(oos_start + test_window, n_rows)
        windows.append((is_start, is_end, oos_start, oos_end))
        if oos_end >= n_rows:
            break
        is_start += step_size
    return windows


def simulate_allocation_series(
    bars: pd.DataFrame,
    target_allocation: pd.Series,
    initial_capital: float,
    slippage_pct: float,
    rebalance_threshold: float,
    fill_delay_bars: int = FILL_DELAY_BARS,
) -> tuple[pd.Series, pd.DataFrame]:
    """Mechanically simulate cash/shares given a pre-decided target
    allocation at each bar. This is the single place the exact allocation
    math lives, reused by both the walk-forward backtest and the
    benchmark generators in ``backtest.performance``::

        equity = cash + shares * current_price
        target_shares = int(equity * target_allocation / current_price)
        delta = target_shares - current_shares
        cash -= delta * price
        shares = target_shares

    A rebalance decided at bar N's close only executes ``fill_delay_bars``
    bars later, at that bar's open (default 1: bar N+1's open) — and only
    if the new target has drifted from the currently held target by more
    than ``rebalance_threshold`` (prevents churn from minor probability
    fluctuations). When ``target_allocation`` exceeds 1.0 (leverage),
    ``cash`` legitimately goes negative — that's margin debt, and
    ``equity = cash + shares * price`` still holds because share value
    exceeds it.
    """
    index = target_allocation.index
    cash = initial_capital
    shares = 0.0
    held_target = 0.0
    pending: list[tuple[pd.Timestamp, float]] = []

    equity_curve = pd.Series(index=index, dtype=float)
    trade_records: list[dict] = []

    for t, ts in enumerate(index):
        price_open = float(bars.loc[ts, "open"])
        price_close = float(bars.loc[ts, "close"])

        still_pending: list[tuple[pd.Timestamp, float]] = []
        for decided_at, target in pending:
            decided_idx = index.get_loc(decided_at)
            if t - decided_idx >= fill_delay_bars:
                equity_at_open = cash + shares * price_open
                target_shares = int(equity_at_open * target / price_open)
                delta = target_shares - shares
                if delta != 0:
                    fill_price = price_open * (1 + slippage_pct if delta > 0 else 1 - slippage_pct)
                    cash -= delta * fill_price
                    trade_records.append(
                        {
                            "timestamp": ts,
                            "decided_at": decided_at,
                            "target_allocation": target,
                            "shares_delta": delta,
                            "fill_price": fill_price,
                            "equity_before": equity_at_open,
                        }
                    )
                shares = target_shares
            else:
                still_pending.append((decided_at, target))
        pending = still_pending

        new_target = float(target_allocation.loc[ts])
        if abs(new_target - held_target) > rebalance_threshold:
            pending.append((ts, new_target))
            held_target = new_target

        equity_curve.loc[ts] = cash + shares * price_close

    trades = pd.DataFrame(trade_records)
    if not trades.empty:
        trades["equity_after"] = trades["timestamp"].map(equity_curve)
    return equity_curve, trades


class Backtester:
    """Walk-forward backtester for the regime-based allocation strategy."""

    def __init__(
        self,
        hmm_engine: HMMEngine,
        strategy: StrategyOrchestrator,
        risk_manager: Optional[RiskManager],
        initial_capital: float,
        slippage_pct: float,
        train_window: int,
        test_window: int,
        step_size: int,
    ) -> None:
        """``hmm_engine`` and ``strategy`` are used as *configuration
        templates*, not pre-fit objects: a fresh ``HMMEngine`` (same
        constructor kwargs) is trained on each walk-forward window's
        in-sample data, and a fresh ``StrategyOrchestrator`` (same
        ``strategy.config``) is built from that window's fitted
        ``regime_info``. ``risk_manager`` is accepted for interface
        compatibility with later phases but is not yet consulted here —
        drawdown halts/position caps are ``core.risk_manager``'s job once
        implemented.
        """
        self._hmm_template = hmm_engine
        self._strategy_config = strategy.config
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.feature_engineer = FeatureEngineer()

    def run(self, price_data: dict[str, pd.DataFrame]) -> BacktestResult:
        """Run the full walk-forward backtest across all symbols.

        Each symbol is simulated as its own single-asset sleeve (the
        allocation math is inherently single-asset), allocated an equal
        share of ``initial_capital``; sleeve equity curves are summed into
        the returned portfolio-level equity curve.
        """
        symbols = list(price_data.keys())
        if not symbols:
            raise ValueError("price_data must contain at least one symbol")
        capital_per_symbol = self.initial_capital / len(symbols)

        sleeve_equity: dict[str, pd.Series] = {}
        all_trades = []
        all_regime_history = []
        window_counts: dict[str, int] = {}

        for symbol in symbols:
            equity, trades, regime_history, n_windows = self._run_symbol(
                symbol, price_data[symbol], capital_per_symbol
            )
            sleeve_equity[symbol] = equity
            window_counts[symbol] = n_windows
            if not trades.empty:
                trades = trades.copy()
                trades["symbol"] = symbol
                all_trades.append(trades)
            if not regime_history.empty:
                regime_history = regime_history.copy()
                regime_history["symbol"] = symbol
                all_regime_history.append(regime_history)

        equity_curve = pd.concat(sleeve_equity.values(), axis=1).sum(axis=1)
        trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        regime_history_df = (
            pd.concat(all_regime_history) if all_regime_history else pd.DataFrame()
        )

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades_df,
            regime_history=regime_history_df,
            metadata={
                "symbols": symbols,
                "windows_per_symbol": window_counts,
                "initial_capital": self.initial_capital,
                "train_window": self.train_window,
                "test_window": self.test_window,
                "step_size": self.step_size,
            },
        )

    def _run_window(
        self,
        train_data: dict[str, pd.DataFrame],
        test_data: dict[str, pd.DataFrame],
    ) -> BacktestResult:
        """Run a single walk-forward train/test window in isolation.

        Thin convenience wrapper around the same per-symbol machinery
        ``run`` uses, exposed for callers (e.g. stress tests) that want to
        evaluate one window's data without the full multi-window loop.
        """
        combined = {
            symbol: pd.concat([train_data[symbol], test_data[symbol]]).sort_index()
            for symbol in train_data
        }
        return self.run(combined)

    def _run_symbol(
        self, symbol: str, bars: pd.DataFrame, initial_capital: float
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, int]:
        features = self.feature_engineer.build_feature_set(bars)
        windows = generate_walk_forward_windows(
            len(features), self.train_window, self.test_window, self.step_size
        )
        if not windows:
            raise ValueError(
                f"{symbol}: not enough history for even one walk-forward window "
                f"(need >= {self.train_window + self.test_window} valid feature rows, "
                f"have {len(features)})"
            )

        target_allocations: dict[pd.Timestamp, float] = {}
        regime_rows: list[dict] = []

        for is_start, is_end, oos_start, oos_end in windows:
            engine = _clone_hmm_engine(self._hmm_template)
            engine.fit(features.iloc[is_start:is_end])
            orchestrator = StrategyOrchestrator(self._strategy_config, engine.regime_info)

            for t in range(oos_start, oos_end):
                causal_window = features.iloc[is_start : t + 1]
                regime_state = engine.predict_regime_filtered(causal_window)[-1]
                is_flickering = engine.is_flickering()

                ts = features.index[t]
                bars_so_far = bars.loc[:ts]
                signals = orchestrator.generate_signals(
                    [symbol], {symbol: bars_so_far}, regime_state, is_flickering
                )
                target = signals[0].position_size_pct * signals[0].leverage if signals else 0.0
                target_allocations[ts] = target

                regime_rows.append(
                    {
                        "timestamp": ts,
                        "regime_id": regime_state.state_id,
                        "label": regime_state.label,
                        "probability": regime_state.probability,
                        "is_confirmed": regime_state.is_confirmed,
                        "consecutive_bars": regime_state.consecutive_bars,
                        "is_flickering": is_flickering,
                        "target_allocation": target,
                    }
                )

        target_series = pd.Series(target_allocations).sort_index()
        oos_bars = bars.loc[target_series.index]

        equity_curve, trades = simulate_allocation_series(
            oos_bars,
            target_series,
            initial_capital,
            self.slippage_pct,
            self._strategy_config.rebalance_threshold,
        )
        regime_history = pd.DataFrame(regime_rows).set_index("timestamp")
        return equity_curve, trades, regime_history, len(windows)
