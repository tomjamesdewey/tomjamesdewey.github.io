"""Entry point for regime-trader.

Loads configuration and credentials, wires up the core engine, broker,
data, monitoring, and risk components, and starts the trading loop.

Currently implements the ``backtest`` subcommand (Phase 4: walk-forward
backtesting, performance reporting, and stress testing)::

    python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
    python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31 --compare
    python main.py backtest --stress-test
"""

from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from dotenv import load_dotenv
from rich.console import Console

from backtest.backtester import Backtester
from backtest.performance import PerformanceAnalyzer
from backtest.stress_test import StressTester
from core.hmm_engine import HMMEngine
from core.regime_strategies import StrategyConfig, StrategyOrchestrator

DEFAULT_CONFIG_PATH = "config/settings.yaml"
REQUIRED_CSV_COLUMNS = ("open", "high", "low", "close", "volume")


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load and parse settings.yaml into a configuration dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_hmm_template(config: dict) -> HMMEngine:
    """An unfit HMMEngine carrying settings.yaml's ``hmm`` section as
    configuration — used as a template by Backtester (a fresh engine is
    trained per walk-forward window; see backtest/backtester.py)."""
    hmm_cfg = config["hmm"]
    return HMMEngine(
        n_candidates=list(hmm_cfg["n_candidates"]),
        n_init=hmm_cfg["n_init"],
        covariance_type=hmm_cfg["covariance_type"],
        min_train_bars=hmm_cfg["min_train_bars"],
        stability_bars=hmm_cfg["stability_bars"],
        flicker_window=hmm_cfg["flicker_window"],
        flicker_threshold=hmm_cfg["flicker_threshold"],
        min_confidence=hmm_cfg["min_confidence"],
    )


def build_strategy_template(config: dict) -> StrategyOrchestrator:
    """A StrategyOrchestrator carrying settings.yaml's ``strategy`` section
    as configuration — used as a template by Backtester (a fresh
    orchestrator is built per window from that window's fitted
    regime_info; see backtest/backtester.py)."""
    strat_cfg = config["strategy"]
    strategy_config = StrategyConfig(
        low_vol_allocation=strat_cfg["low_vol_allocation"],
        mid_vol_allocation_trend=strat_cfg["mid_vol_allocation_trend"],
        mid_vol_allocation_no_trend=strat_cfg["mid_vol_allocation_no_trend"],
        high_vol_allocation=strat_cfg["high_vol_allocation"],
        low_vol_leverage=strat_cfg["low_vol_leverage"],
        rebalance_threshold=strat_cfg["rebalance_threshold"],
        uncertainty_size_mult=strat_cfg["uncertainty_size_mult"],
        min_confidence=config["hmm"]["min_confidence"],
    )
    return StrategyOrchestrator(strategy_config, {})


def build_backtester(config: dict) -> Backtester:
    """Wire together a Backtester from settings.yaml's ``backtest`` section."""
    backtest_cfg = config["backtest"]
    return Backtester(
        hmm_engine=build_hmm_template(config),
        strategy=build_strategy_template(config),
        risk_manager=None,
        initial_capital=backtest_cfg["initial_capital"],
        slippage_pct=backtest_cfg["slippage_pct"],
        train_window=backtest_cfg["train_window"],
        test_window=backtest_cfg["test_window"],
        step_size=backtest_cfg["step_size"],
    )


def _load_price_data_from_csv(
    symbols: list[str], start: str, end: str, data_dir: str
) -> dict[str, pd.DataFrame]:
    """Load ``{data_dir}/{symbol}.csv`` files (date-indexed OHLCV) as an
    offline alternative to live Alpaca data — used by ``--data-dir`` and by
    the test suite, since this environment has no Alpaca credentials."""
    price_data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = Path(data_dir) / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No CSV found for {symbol} at {path}")
        bars = pd.read_csv(path, index_col=0, parse_dates=True)
        bars.columns = [c.lower() for c in bars.columns]
        missing = [c for c in REQUIRED_CSV_COLUMNS if c not in bars.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        bars = bars.sort_index().loc[start:end]
        price_data[symbol] = bars
    return price_data


def _load_price_data_from_alpaca(
    symbols: list[str], start: str, end: str, config: dict
) -> dict[str, pd.DataFrame]:
    """Fetch historical bars from Alpaca via broker.alpaca_client /
    data.market_data. Requires ALPACA_API_KEY/ALPACA_SECRET_KEY (see
    .env.example) and that those modules' fetch methods are implemented.
    """
    from broker.alpaca_client import AlpacaClient
    from data.market_data import MarketDataClient

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set. Copy .env.example "
            "to .env and fill in your Alpaca credentials, or pass --data-dir "
            "to backtest against local CSV files instead."
        )
    paper = os.environ.get("ALPACA_PAPER", "true").lower() == "true"

    client = AlpacaClient(api_key, secret_key, paper=paper)
    market_data = MarketDataClient(client)
    timeframe = config["broker"]["timeframe"]

    price_data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        bars = market_data.get_historical_bars(
            symbol, timeframe, datetime.fromisoformat(start), datetime.fromisoformat(end)
        )
        if not isinstance(bars, pd.DataFrame) or bars.empty:
            raise RuntimeError(
                "data.market_data.MarketDataClient.get_historical_bars() did not "
                "return data — that class is not implemented yet in this phase. "
                "Pass --data-dir to backtest against local CSV files instead."
            )
        price_data[symbol] = bars
    return price_data


def load_price_data(
    symbols: list[str], start: str, end: str, config: dict, data_dir: Optional[str] = None
) -> dict[str, pd.DataFrame]:
    """Load historical OHLCV bars for ``symbols`` between ``start`` and ``end``."""
    if data_dir:
        return _load_price_data_from_csv(symbols, start, end, data_dir)
    return _load_price_data_from_alpaca(symbols, start, end, config)


def build_app(config: dict) -> None:
    """Construct and wire together all application components."""
    ...


def cmd_backtest(args: argparse.Namespace, config: dict) -> None:
    """Run the Phase 4 walk-forward backtest, print a performance report,
    write CSV output, and (optionally) run benchmark comparisons and
    stress tests."""
    console = Console()
    symbols = args.symbols or config["broker"]["symbols"]
    end = args.end or date.today().isoformat()
    start = args.start or (date.today() - timedelta(days=6 * 365)).isoformat()

    console.print(f"[bold]Loading price data[/bold] for {symbols} from {start} to {end}...")
    price_data = load_price_data(symbols, start, end, config, data_dir=args.data_dir)

    backtester = build_backtester(config)
    console.print("[bold]Running walk-forward backtest...[/bold]")
    result = backtester.run(price_data)

    analyzer = PerformanceAnalyzer(risk_free_rate=config["backtest"]["risk_free_rate"])
    metrics = analyzer.compute_metrics(result.equity_curve, result.trades)
    worst_case = analyzer.compute_worst_case_stats(result.equity_curve, result.trades)
    regime_breakdown = analyzer.compute_regime_breakdown(
        result.equity_curve, result.regime_history, result.trades
    )
    confidence_breakdown = analyzer.compute_confidence_breakdown(
        result.equity_curve, result.trades, result.regime_history
    )

    benchmark_comparison = None
    if args.compare:
        console.print("[bold]Running benchmark comparisons...[/bold]")
        first_symbol = symbols[0]
        bars = price_data[first_symbol].loc[result.equity_curve.index[0] : result.equity_curve.index[-1]]
        buy_and_hold = analyzer.generate_buy_and_hold(bars, config["backtest"]["initial_capital"])
        sma_trend = analyzer.generate_sma_trend_benchmark(
            price_data[first_symbol], config["backtest"]["initial_capital"]
        )
        random_mc = analyzer.run_random_benchmark_monte_carlo(
            bars,
            config["backtest"]["initial_capital"],
            config["backtest"]["slippage_pct"],
            config["strategy"]["rebalance_threshold"],
            trade_frequency_bars=max(1, len(bars) // max(1, len(result.trades))),
        )
        benchmark_comparison = analyzer.compare_to_benchmark(
            result.equity_curve, buy_and_hold.reindex(result.equity_curve.index).ffill()
        )
        console.print(
            f"200-SMA trend benchmark final equity: {sma_trend.iloc[-1]:,.2f} | "
            f"Random-entry benchmark (n={len(random_mc)}): "
            f"mean return {random_mc['total_return_pct'].mean():.2%} "
            f"(std {random_mc['total_return_pct'].std():.2%})"
        )

    analyzer.print_report(metrics, regime_breakdown, confidence_breakdown, worst_case, benchmark_comparison, console)

    out_dir = Path(args.output_dir)
    analyzer.export_csvs(out_dir, result.equity_curve, result.trades, result.regime_history, benchmark_comparison)
    console.print(f"[bold green]Wrote equity_curve.csv, trade_log.csv, regime_history.csv to {out_dir}[/bold green]")

    if args.stress_test:
        console.print("[bold]Running stress tests...[/bold]")
        stress_tester = StressTester(backtester, circuit_breaker_dd_threshold=config["risk"]["max_dd_from_peak"])

        crash = stress_tester.crash_injection_test(price_data, baseline_result=result)
        console.print(
            f"Crash injection ({crash.n_simulations} sims): mean max DD "
            f"{crash.mean_max_drawdown_pct:.2%}, worst {crash.worst_max_drawdown_pct:.2%}, "
            f"circuit breaker fired in {crash.pct_circuit_breaker_fired:.1%} of runs"
        )

        gap = stress_tester.gap_risk_test(price_data, baseline_result=result)
        console.print(
            f"Gap risk ({gap.n_simulations} sims): expected loss {gap.expected_loss_pct:.2%} "
            f"vs. actual mean loss {gap.actual_mean_loss_pct:.2%} (worst {gap.actual_worst_loss_pct:.2%})"
        )

        shuffle = stress_tester.regime_misclassification_test(price_data, baseline_result=result)
        contained_str = "contained" if shuffle.contained else "NOT contained"
        console.print(
            f"Regime misclassification ({shuffle.n_shuffles} shuffles): baseline max DD "
            f"{shuffle.baseline_max_drawdown_pct:.2%}, shuffled worst "
            f"{shuffle.worst_shuffled_max_drawdown_pct:.2%} — risk is {contained_str}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="regime-trader")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to settings.yaml")

    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest_parser = subparsers.add_parser("backtest", help="Run the walk-forward backtest")
    backtest_parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to backtest (default: settings.yaml broker.symbols)")
    backtest_parser.add_argument("--start", default=None, help="Start date, YYYY-MM-DD (default: 6 years ago)")
    backtest_parser.add_argument("--end", default=None, help="End date, YYYY-MM-DD (default: today)")
    backtest_parser.add_argument("--compare", action="store_true", help="Run benchmark comparisons (buy-and-hold, 200-SMA, random-entry)")
    backtest_parser.add_argument("--stress-test", action="store_true", dest="stress_test", help="Run crash/gap/regime-misclassification stress tests")
    backtest_parser.add_argument("--data-dir", default=None, dest="data_dir", help="Load OHLCV bars from {data_dir}/{symbol}.csv instead of Alpaca")
    backtest_parser.add_argument("--output-dir", default="backtest_output", dest="output_dir", help="Directory to write CSV output to")

    return parser


def main() -> None:
    """Parse CLI arguments and start regime-trader."""
    load_dotenv()
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "backtest":
        cmd_backtest(args, config)


if __name__ == "__main__":
    main()
