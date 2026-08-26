"""Performance analytics: Sharpe, drawdown, regime breakdown, and benchmarks."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from backtest.backtester import simulate_allocation_series

TRADING_DAYS_PER_YEAR = 252

#: Candidate allocation values used by the random-entry benchmark, matching
#: the range of position_size_pct * leverage the real strategies can emit
#: (flat, high-vol/no-trend, low/mid-vol, low-vol-leveraged).
DEFAULT_RANDOM_BENCHMARK_ALLOCATIONS = (0.0, 0.60, 0.95, 1.1875)

CONFIDENCE_BUCKET_EDGES = (0.0, 0.50, 0.60, 0.70, 1.0)
CONFIDENCE_BUCKET_LABELS = ("<50%", "50-60%", "60-70%", "70%+")


@dataclass
class PerformanceMetrics:
    """Summary performance metrics for a backtest or live equity curve."""

    total_return_pct: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    total_trades: int
    avg_holding_period_days: float


@dataclass
class WorstCaseStats:
    """Worst-case realized stress statistics."""

    worst_day_pct: float
    worst_week_pct: float
    worst_month_pct: float
    max_consecutive_losing_trades: int
    longest_underwater_days: int


def compute_returns(equity_curve: pd.Series) -> pd.Series:
    """Simple per-bar returns of an equity curve."""
    return equity_curve.pct_change().dropna()


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Drawdown from the running peak, as a (non-positive) fraction."""
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1.0


def _drawdown_episodes(equity_curve: pd.Series) -> list[dict]:
    """Contiguous underwater episodes, each with its trough depth and duration."""
    drawdown = compute_drawdown_series(equity_curve)
    episodes: list[dict] = []
    start_idx: Optional[int] = None
    trough = 0.0

    for i, dd in enumerate(drawdown.to_numpy()):
        if dd < 0:
            if start_idx is None:
                start_idx = i
                trough = dd
            else:
                trough = min(trough, dd)
        elif start_idx is not None:
            episodes.append(
                {
                    "start": drawdown.index[start_idx],
                    "end": drawdown.index[i],
                    "trough_pct": trough,
                    "duration_days": i - start_idx,
                }
            )
            start_idx = None

    if start_idx is not None:
        episodes.append(
            {
                "start": drawdown.index[start_idx],
                "end": drawdown.index[-1],
                "trough_pct": trough,
                "duration_days": len(drawdown) - start_idx,
            }
        )
    return episodes


def compute_max_drawdown(equity_curve: pd.Series) -> tuple[float, int]:
    """(max_drawdown_pct [<=0], duration_in_trading_days) of the deepest episode."""
    episodes = _drawdown_episodes(equity_curve)
    if not episodes:
        return 0.0, 0
    worst = min(episodes, key=lambda e: e["trough_pct"])
    return float(worst["trough_pct"]), int(worst["duration_days"])


def compute_longest_underwater(equity_curve: pd.Series) -> int:
    """Longest continuous underwater streak, in trading days (may differ from
    the max-drawdown episode's own duration)."""
    episodes = _drawdown_episodes(equity_curve)
    if not episodes:
        return 0
    return int(max(e["duration_days"] for e in episodes))


def compute_cagr(equity_curve: pd.Series) -> float:
    n_days = len(equity_curve) - 1
    if n_days <= 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = n_days / TRADING_DAYS_PER_YEAR
    if total_return <= 0:
        return -1.0
    return float(total_return ** (1 / years) - 1.0)


def compute_annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def compute_sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    std = excess.std(ddof=1)
    if not std or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def compute_sortino_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1)
    if not downside_std or np.isnan(downside_std):
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def compute_calmar_ratio(cagr: float, max_drawdown_pct: float) -> float:
    if max_drawdown_pct == 0:
        return 0.0
    return cagr / abs(max_drawdown_pct)


def compute_trade_pnls(equity_curve: pd.Series, trades: pd.DataFrame) -> pd.DataFrame:
    """P&L earned holding the allocation each trade established, from that
    trade's fill to the next trade's fill (or to the end of the equity
    curve for the last trade). Uses ``equity_curve`` for the P&L, so this
    is exact for a single-symbol backtest; for a multi-symbol portfolio it
    approximates each trade's P&L with the *aggregate* portfolio's move
    over that holding window.
    """
    columns = ["timestamp", "exit_timestamp", "pnl_pct", "holding_days"]
    if trades.empty:
        return pd.DataFrame(columns=columns)

    ordered = trades.sort_values("timestamp").reset_index(drop=True)
    records = []
    for i in range(len(ordered)):
        entry_ts = ordered.loc[i, "timestamp"]
        exit_ts = (
            ordered.loc[i + 1, "timestamp"] if i + 1 < len(ordered) else equity_curve.index[-1]
        )
        entry_equity = equity_curve.loc[entry_ts]
        exit_equity = equity_curve.loc[exit_ts]
        pnl_pct = float(exit_equity / entry_equity - 1.0) if entry_equity else 0.0
        holding_days = int(
            equity_curve.index.get_loc(exit_ts) - equity_curve.index.get_loc(entry_ts)
        )
        records.append(
            {
                "timestamp": entry_ts,
                "exit_timestamp": exit_ts,
                "pnl_pct": pnl_pct,
                "holding_days": holding_days,
            }
        )
    return pd.DataFrame(records)


def _augment_trades_with_regime(trades: pd.DataFrame, regime_history: pd.DataFrame) -> pd.DataFrame:
    """Join each trade to the regime that was active when it was *decided*."""
    if trades.empty or regime_history.empty:
        return trades

    right = regime_history.reset_index().rename(
        columns={"timestamp": "decided_at", "label": "regime_name", "probability": "regime_probability"}
    )
    join_keys = ["decided_at"]
    if "symbol" in trades.columns and "symbol" in right.columns:
        join_keys.append("symbol")

    return trades.merge(right[join_keys + ["regime_name", "regime_probability"]], on=join_keys, how="left")


class PerformanceAnalyzer:
    """Computes performance metrics and regime-level breakdowns."""

    def __init__(self, risk_free_rate: float) -> None:
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self, equity_curve: pd.Series, trades: Optional[pd.DataFrame] = None
    ) -> PerformanceMetrics:
        """Compute summary performance metrics from an equity curve (and,
        for the trade-based stats, an optional trade log)."""
        returns = compute_returns(equity_curve)
        cagr = compute_cagr(equity_curve)
        max_dd, max_dd_days = compute_max_drawdown(equity_curve)

        trade_pnls = (
            compute_trade_pnls(equity_curve, trades)
            if trades is not None and not trades.empty
            else pd.DataFrame(columns=["pnl_pct", "holding_days"])
        )
        wins = trade_pnls.loc[trade_pnls["pnl_pct"] > 0, "pnl_pct"]
        losses = trade_pnls.loc[trade_pnls["pnl_pct"] <= 0, "pnl_pct"]

        win_rate = float(len(wins) / len(trade_pnls)) if len(trade_pnls) else 0.0
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
        avg_holding = float(trade_pnls["holding_days"].mean()) if len(trade_pnls) else 0.0

        return PerformanceMetrics(
            total_return_pct=float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0),
            cagr=cagr,
            annualized_volatility=compute_annualized_volatility(returns),
            sharpe_ratio=compute_sharpe_ratio(returns, self.risk_free_rate),
            sortino_ratio=compute_sortino_ratio(returns, self.risk_free_rate),
            calmar_ratio=compute_calmar_ratio(cagr, max_dd),
            max_drawdown_pct=max_dd,
            max_drawdown_duration_days=max_dd_days,
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            total_trades=len(trade_pnls),
            avg_holding_period_days=avg_holding,
        )

    def compute_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        return compute_drawdown_series(equity_curve)

    def compute_worst_case_stats(
        self, equity_curve: pd.Series, trades: Optional[pd.DataFrame] = None
    ) -> WorstCaseStats:
        """Worst single day/week/month, longest losing-trade streak, and
        longest continuous time spent underwater."""
        daily_returns = compute_returns(equity_curve)
        weekly_returns = equity_curve.resample("W").last().pct_change().dropna()
        monthly_returns = equity_curve.resample("ME").last().pct_change().dropna()

        max_consecutive_losses = 0
        if trades is not None and not trades.empty:
            trade_pnls = compute_trade_pnls(equity_curve, trades)
            streak = 0
            for pnl in trade_pnls["pnl_pct"]:
                if pnl <= 0:
                    streak += 1
                    max_consecutive_losses = max(max_consecutive_losses, streak)
                else:
                    streak = 0

        return WorstCaseStats(
            worst_day_pct=float(daily_returns.min()) if len(daily_returns) else 0.0,
            worst_week_pct=float(weekly_returns.min()) if len(weekly_returns) else 0.0,
            worst_month_pct=float(monthly_returns.min()) if len(monthly_returns) else 0.0,
            max_consecutive_losing_trades=max_consecutive_losses,
            longest_underwater_days=compute_longest_underwater(equity_curve),
        )

    # ------------------------------------------------------------------
    # Breakdowns
    # ------------------------------------------------------------------

    def compute_regime_breakdown(
        self, equity_curve: pd.Series, regime_history: pd.DataFrame, trades: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Regime | % Time In | Return Contribution | Avg Trade P&L | Win Rate | Sharpe."""
        if regime_history.empty:
            return pd.DataFrame(
                columns=["regime", "pct_time_in", "return_contribution_pct", "avg_trade_pnl_pct", "win_rate", "sharpe"]
            )

        returns = compute_returns(equity_curve)
        labels = regime_history["label"].reindex(returns.index).ffill()

        trade_pnls_by_regime: dict[str, pd.Series] = {}
        if trades is not None and not trades.empty:
            augmented = _augment_trades_with_regime(trades, regime_history)
            trade_pnls = compute_trade_pnls(equity_curve, trades)
            if "regime_name" in augmented.columns:
                trade_pnls = trade_pnls.merge(
                    augmented[["timestamp", "regime_name"]], on="timestamp", how="left"
                )
                for regime_name, group in trade_pnls.groupby("regime_name"):
                    trade_pnls_by_regime[regime_name] = group["pnl_pct"]

        rows = []
        for label, group_returns in returns.groupby(labels):
            pct_time_in = len(group_returns) / len(returns)
            return_contribution = float((1.0 + group_returns).prod() - 1.0)
            regime_trade_pnls = trade_pnls_by_regime.get(label, pd.Series(dtype=float))
            avg_trade_pnl = float(regime_trade_pnls.mean()) if len(regime_trade_pnls) else float("nan")
            win_rate = float((regime_trade_pnls > 0).mean()) if len(regime_trade_pnls) else float("nan")
            rows.append(
                {
                    "regime": label,
                    "pct_time_in": pct_time_in,
                    "return_contribution_pct": return_contribution,
                    "avg_trade_pnl_pct": avg_trade_pnl,
                    "win_rate": win_rate,
                    "sharpe": compute_sharpe_ratio(group_returns, self.risk_free_rate),
                }
            )
        return pd.DataFrame(rows).sort_values("pct_time_in", ascending=False).reset_index(drop=True)

    def compute_confidence_breakdown(
        self, equity_curve: pd.Series, trades: pd.DataFrame, regime_history: pd.DataFrame
    ) -> pd.DataFrame:
        """Confidence | Trades | Sharpe | Win Rate | Avg P&L, bucketed at
        <50%, 50-60%, 60-70%, 70%+. A trade-level ("per-trade") Sharpe-like
        statistic (mean/std of trade P&Ls, not an annualized daily-bar
        Sharpe) is used since trades are irregularly spaced.
        """
        columns = ["confidence_bucket", "trades", "sharpe", "win_rate", "avg_pnl_pct"]
        if trades.empty:
            return pd.DataFrame(columns=columns)

        augmented = _augment_trades_with_regime(trades, regime_history)
        trade_pnls = compute_trade_pnls(equity_curve, trades)
        if "regime_probability" not in augmented.columns:
            return pd.DataFrame(columns=columns)

        trade_pnls = trade_pnls.merge(
            augmented[["timestamp", "regime_probability"]], on="timestamp", how="left"
        )
        trade_pnls["confidence_bucket"] = pd.cut(
            trade_pnls["regime_probability"],
            bins=CONFIDENCE_BUCKET_EDGES,
            labels=CONFIDENCE_BUCKET_LABELS,
            include_lowest=True,
        )

        rows = []
        for bucket in CONFIDENCE_BUCKET_LABELS:
            group = trade_pnls.loc[trade_pnls["confidence_bucket"] == bucket, "pnl_pct"]
            if len(group) == 0:
                rows.append({"confidence_bucket": bucket, "trades": 0, "sharpe": 0.0, "win_rate": 0.0, "avg_pnl_pct": 0.0})
                continue
            std = group.std(ddof=1)
            sharpe_like = float(group.mean() / std * np.sqrt(len(group))) if std and not np.isnan(std) else 0.0
            rows.append(
                {
                    "confidence_bucket": bucket,
                    "trades": int(len(group)),
                    "sharpe": sharpe_like,
                    "win_rate": float((group > 0).mean()),
                    "avg_pnl_pct": float(group.mean()),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------

    def generate_buy_and_hold(self, bars: pd.DataFrame, initial_capital: float) -> pd.Series:
        """Hold the asset for the entire period."""
        shares = initial_capital / float(bars["close"].iloc[0])
        return shares * bars["close"]

    def generate_sma_trend_benchmark(
        self, bars: pd.DataFrame, initial_capital: float, sma_window: int = 200, slippage_pct: float = 0.0
    ) -> pd.Series:
        """Long when price is above its ``sma_window``-bar SMA, cash below."""
        sma = bars["close"].rolling(window=sma_window, min_periods=sma_window).mean()
        target_allocation = (bars["close"] > sma).astype(float)
        target_allocation = target_allocation.dropna()
        valid_bars = bars.loc[target_allocation.index]
        equity_curve, _ = simulate_allocation_series(
            valid_bars, target_allocation, initial_capital, slippage_pct, rebalance_threshold=0.0
        )
        return equity_curve

    def generate_random_benchmark(
        self,
        bars: pd.DataFrame,
        initial_capital: float,
        slippage_pct: float,
        rebalance_threshold: float,
        trade_frequency_bars: int,
        allocation_choices: tuple[float, ...] = DEFAULT_RANDOM_BENCHMARK_ALLOCATIONS,
        seed: int = 0,
    ) -> pd.Series:
        """Random allocation changes at the same frequency, using the same
        position-sizing rules (slippage, rebalance threshold) as the real
        strategy — one random realization."""
        rng = np.random.RandomState(seed)
        n = len(bars)
        change_points = set(range(0, n, max(1, trade_frequency_bars)))
        values = []
        current = 0.0
        for i in range(n):
            if i in change_points:
                current = float(rng.choice(allocation_choices))
            values.append(current)
        target_allocation = pd.Series(values, index=bars.index)
        equity_curve, _ = simulate_allocation_series(
            bars, target_allocation, initial_capital, slippage_pct, rebalance_threshold
        )
        return equity_curve

    def run_random_benchmark_monte_carlo(
        self,
        bars: pd.DataFrame,
        initial_capital: float,
        slippage_pct: float,
        rebalance_threshold: float,
        trade_frequency_bars: int,
        allocation_choices: tuple[float, ...] = DEFAULT_RANDOM_BENCHMARK_ALLOCATIONS,
        n_seeds: int = 100,
    ) -> pd.DataFrame:
        """Run the random-entry benchmark across ``n_seeds`` seeds and
        return one metrics row per seed (report mean/std across rows)."""
        rows = []
        for seed in range(n_seeds):
            equity_curve = self.generate_random_benchmark(
                bars,
                initial_capital,
                slippage_pct,
                rebalance_threshold,
                trade_frequency_bars,
                allocation_choices,
                seed=seed,
            )
            metrics = self.compute_metrics(equity_curve)
            rows.append({"seed": seed, **asdict(metrics)})
        return pd.DataFrame(rows)

    def compare_to_benchmark(self, equity_curve: pd.Series, benchmark_curve: pd.Series) -> pd.DataFrame:
        """Compare strategy performance against a benchmark equity curve."""
        strategy_metrics = self.compute_metrics(equity_curve)
        benchmark_metrics = self.compute_metrics(benchmark_curve)
        return pd.DataFrame(
            {"strategy": asdict(strategy_metrics), "benchmark": asdict(benchmark_metrics)}
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(
        self,
        metrics: PerformanceMetrics,
        regime_breakdown: Optional[pd.DataFrame] = None,
        confidence_breakdown: Optional[pd.DataFrame] = None,
        worst_case: Optional[WorstCaseStats] = None,
        benchmark_comparison: Optional[pd.DataFrame] = None,
        console: Optional[Console] = None,
    ) -> None:
        """Render rich-formatted tables of the backtest results to the terminal."""
        console = console or Console()

        summary = Table(title="Performance Summary")
        summary.add_column("Metric")
        summary.add_column("Value", justify="right")
        summary.add_row("Total Return", f"{metrics.total_return_pct:.2%}")
        summary.add_row("CAGR", f"{metrics.cagr:.2%}")
        summary.add_row("Annualized Volatility", f"{metrics.annualized_volatility:.2%}")
        summary.add_row("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
        summary.add_row("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
        summary.add_row("Calmar Ratio", f"{metrics.calmar_ratio:.2f}")
        summary.add_row("Max Drawdown", f"{metrics.max_drawdown_pct:.2%}")
        summary.add_row("Max Drawdown Duration", f"{metrics.max_drawdown_duration_days} days")
        summary.add_row("Win Rate", f"{metrics.win_rate:.2%}")
        summary.add_row("Avg Win", f"{metrics.avg_win_pct:.2%}")
        summary.add_row("Avg Loss", f"{metrics.avg_loss_pct:.2%}")
        summary.add_row("Profit Factor", f"{metrics.profit_factor:.2f}")
        summary.add_row("Total Trades", str(metrics.total_trades))
        summary.add_row("Avg Holding Period", f"{metrics.avg_holding_period_days:.1f} days")
        console.print(summary)

        if worst_case is not None:
            worst = Table(title="Worst-Case Stats")
            worst.add_column("Metric")
            worst.add_column("Value", justify="right")
            worst.add_row("Worst Day", f"{worst_case.worst_day_pct:.2%}")
            worst.add_row("Worst Week", f"{worst_case.worst_week_pct:.2%}")
            worst.add_row("Worst Month", f"{worst_case.worst_month_pct:.2%}")
            worst.add_row("Max Consecutive Losing Trades", str(worst_case.max_consecutive_losing_trades))
            worst.add_row("Longest Time Underwater", f"{worst_case.longest_underwater_days} days")
            console.print(worst)

        if regime_breakdown is not None and not regime_breakdown.empty:
            regime_table = Table(title="Regime Breakdown")
            for col in ["Regime", "% Time In", "Return Contribution", "Avg Trade P&L", "Win Rate", "Sharpe"]:
                regime_table.add_column(col)
            for _, row in regime_breakdown.iterrows():
                regime_table.add_row(
                    str(row["regime"]),
                    f"{row['pct_time_in']:.1%}",
                    f"{row['return_contribution_pct']:.2%}",
                    "n/a" if pd.isna(row["avg_trade_pnl_pct"]) else f"{row['avg_trade_pnl_pct']:.2%}",
                    "n/a" if pd.isna(row["win_rate"]) else f"{row['win_rate']:.1%}",
                    f"{row['sharpe']:.2f}",
                )
            console.print(regime_table)

        if confidence_breakdown is not None and not confidence_breakdown.empty:
            conf_table = Table(title="Confidence-Bucketed Trades")
            for col in ["Confidence", "Trades", "Sharpe", "Win Rate", "Avg P&L"]:
                conf_table.add_column(col)
            for _, row in confidence_breakdown.iterrows():
                conf_table.add_row(
                    str(row["confidence_bucket"]),
                    str(row["trades"]),
                    f"{row['sharpe']:.2f}",
                    f"{row['win_rate']:.1%}",
                    f"{row['avg_pnl_pct']:.2%}",
                )
            console.print(conf_table)

        if benchmark_comparison is not None and not benchmark_comparison.empty:
            bench_table = Table(title="Strategy vs. Benchmark")
            bench_table.add_column("Metric")
            bench_table.add_column("Strategy", justify="right")
            bench_table.add_column("Benchmark", justify="right")
            for metric_name in benchmark_comparison.index:
                bench_table.add_row(
                    metric_name,
                    str(benchmark_comparison.loc[metric_name, "strategy"]),
                    str(benchmark_comparison.loc[metric_name, "benchmark"]),
                )
            console.print(bench_table)

    def export_csvs(
        self,
        out_dir: str | Path,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        regime_history: pd.DataFrame,
        benchmark_comparison: Optional[pd.DataFrame] = None,
    ) -> None:
        """Write equity_curve.csv, trade_log.csv, regime_history.csv, and
        (if provided) benchmark_comparison.csv to ``out_dir``."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        equity_curve.rename("equity").to_csv(out_dir / "equity_curve.csv", header=True)
        trades.to_csv(out_dir / "trade_log.csv", index=False, quoting=csv.QUOTE_MINIMAL)
        regime_history.to_csv(out_dir / "regime_history.csv")
        if benchmark_comparison is not None:
            benchmark_comparison.to_csv(out_dir / "benchmark_comparison.csv")
