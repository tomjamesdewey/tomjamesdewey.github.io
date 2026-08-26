"""Entry point for regime-trader.

Loads configuration and credentials, wires up the core engine, broker,
data, monitoring, and risk components, and starts the trading loop.

Subcommands::

    python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31 [--compare] [--stress-test]
    python main.py train-only [--symbols SPY ...]
    python main.py run [--dry-run]
    python main.py dashboard

Design note on bar cadence: the HMM is trained on, and regime/signal
decisions are made from, ``broker.timeframe`` (daily) bars — that's the
granularity the model actually understands. The live loop instead wakes
on ``live.bar_timeframe`` (default 5-minute) bars purely as an
operational heartbeat: every tick it re-checks circuit breakers, trailing
stops, and the dashboard, and once per new calendar day it re-fetches the
daily history and re-runs the regime/signal pipeline. This keeps
risk-management responsive intraday without pretending the HMM has
5-minute-resolution opinions about volatility regimes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import signal as signal_module
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from backtest.backtester import Backtester
from backtest.performance import PerformanceAnalyzer
from backtest.stress_test import StressTester
from broker.alpaca_client import AlpacaClient
from broker.order_executor import OrderExecutor
from broker.position_tracker import PositionTracker
from core.hmm_engine import HMMEngine
from core.regime_strategies import STOP_LOSS_ATR_MULTIPLE, StrategyConfig, StrategyOrchestrator
from core.regime_strategies import Signal as StrategySignal
from core.risk_manager import RiskConfig, RiskManager
from data.feature_engineering import ATR_WINDOW, FeatureEngineer, average_true_range
from data.market_data import MarketDataClient

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config/settings.yaml"
REQUIRED_CSV_COLUMNS = ("open", "high", "low", "close", "volume")

#: Calendar days of history fetched to (re)train/refresh a symbol's daily
#: bars — generous enough to cover feature warm-up (~452 bars) plus
#: min_train_bars with room to spare.
TRAINING_LOOKBACK_DAYS = 1460

#: A live bar older than this multiple of the live bar_timeframe is
#: treated as a dropped/stale data feed — new signals are paused (existing
#: stops are left untouched) until fresh bars resume.
STALE_FEED_MULTIPLE = 3


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


def build_risk_manager(config: dict, lock_file_path: Optional[str] = None) -> RiskManager:
    """Wire together a RiskManager from settings.yaml's ``risk`` section."""
    risk_cfg = config["risk"]
    risk_config = RiskConfig(
        max_risk_per_trade=risk_cfg["max_risk_per_trade"],
        max_exposure=risk_cfg["max_exposure"],
        max_leverage=risk_cfg["max_leverage"],
        max_single_position=risk_cfg["max_single_position"],
        max_concurrent=risk_cfg["max_concurrent"],
        max_daily_trades=risk_cfg["max_daily_trades"],
        daily_dd_reduce=risk_cfg["daily_dd_reduce"],
        daily_dd_halt=risk_cfg["daily_dd_halt"],
        weekly_dd_reduce=risk_cfg["weekly_dd_reduce"],
        weekly_dd_halt=risk_cfg["weekly_dd_halt"],
        max_dd_from_peak=risk_cfg["max_dd_from_peak"],
        gap_stop_multiple=risk_cfg["gap_stop_multiple"],
        overnight_gap_risk_pct=risk_cfg["overnight_gap_risk_pct"],
        min_position_usd=risk_cfg["min_position_usd"],
        max_correlation_reduce=risk_cfg["max_correlation_reduce"],
        max_correlation_reject=risk_cfg["max_correlation_reject"],
        correlation_window_days=risk_cfg["correlation_window_days"],
        max_sector_exposure=risk_cfg["max_sector_exposure"],
        max_spread_pct=risk_cfg["max_spread_pct"],
        duplicate_window_seconds=risk_cfg["duplicate_window_seconds"],
        flicker_rate_threshold=risk_cfg["flicker_rate_threshold"],
    )
    return RiskManager(risk_config, lock_file_path=lock_file_path or config["live"]["lock_file_path"])


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
    .env.example) and network access to Alpaca.
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
                f"Alpaca returned no bars for {symbol} between {start} and {end}. "
                "Check the symbol and date range, or pass --data-dir to backtest "
                "against local CSV files instead."
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


@dataclass
class TradingState:
    """Everything needed to resume a session after a restart, and to power
    ``python main.py dashboard`` for a separate, already-running instance.
    Written to ``live.state_snapshot_path`` on every housekeeping tick."""

    last_regime_date: dict = field(default_factory=dict)
    hmm_last_trained: dict = field(default_factory=dict)
    trades_today: int = 0
    daily_start_equity: Optional[float] = None
    weekly_start_equity: Optional[float] = None
    peak_equity: Optional[float] = None
    circuit_breaker_state: dict = field(default_factory=dict)
    open_positions: list = field(default_factory=list)
    session_started_at: Optional[str] = None
    last_updated_at: Optional[str] = None
    dry_run: bool = False


class TradingSession:
    """Orchestrates startup, the live per-bar loop, and graceful shutdown.

    Every collaborator (Alpaca client, market data, order executor,
    position tracker, risk manager) is injected, so the orchestration
    logic here — bar handling, signal validation/submission, trailing
    stops, circuit breakers, state persistence — can be unit tested with
    mocked collaborators without touching a real Alpaca connection. Only
    ``run()``'s outer blocking WebSocket loop can't be exercised that way.
    """

    def __init__(
        self,
        config: dict,
        client: AlpacaClient,
        market_data: MarketDataClient,
        order_executor: OrderExecutor,
        position_tracker: PositionTracker,
        risk_manager: RiskManager,
        symbols: list[str],
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.client = client
        self.market_data = market_data
        self.order_executor = order_executor
        self.position_tracker = position_tracker
        self.risk_manager = risk_manager
        self.symbols = symbols
        self.dry_run = dry_run

        live_cfg = config["live"]
        self.timeframe = config["broker"]["timeframe"]
        self.live_bar_timeframe = live_cfg["bar_timeframe"]
        self.model_stale_days = live_cfg["model_stale_days"]
        self.model_dir = Path(live_cfg["model_dir"])
        self.state_snapshot_path = Path(live_cfg["state_snapshot_path"])
        self.max_market_wait_seconds = live_cfg["max_market_wait_seconds"]
        self.dashboard_refresh_seconds = config["monitoring"]["dashboard_refresh_seconds"]
        self.strategy_config = build_strategy_template(config).config

        self.feature_engineer = FeatureEngineer()
        self.engines: dict[str, HMMEngine] = {}
        self.orchestrators: dict[str, StrategyOrchestrator] = {}
        self.bars_history: dict[str, pd.DataFrame] = {}
        self.hmm_last_trained: dict[str, datetime] = {}
        self._last_regime_date: dict[str, date] = {}
        self._last_regime_state: dict[str, object] = {}
        self._last_bar_at: dict[str, datetime] = {}
        self._last_dashboard_refresh = 0.0
        self._session_started_at = datetime.now(timezone.utc)
        self._running = False

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def startup(self) -> bool:
        """Steps 1-8 of the startup sequence. Returns False if the market
        is closed and startup should abort (caller decides what that means)."""
        logger.info("Starting up regime-trader (%s)...", "paper" if self.client.paper else "LIVE trading")
        account = self.client.get_account()
        logger.info("Account verified: equity=%s buying_power=%s", account["equity"], account["buying_power"])

        if not self.check_market_hours():
            logger.info("Market is closed; aborting startup.")
            return False

        for symbol in self.symbols:
            bars = self._fetch_daily_history(symbol)
            self.bars_history[symbol] = bars
            self.engines[symbol] = self.load_or_train_hmm(symbol, bars)
            self.orchestrators[symbol] = StrategyOrchestrator(self.strategy_config, self.engines[symbol].regime_info)

        self.position_tracker.sync_with_alpaca()
        self._restore_state_snapshot()

        logger.info("System online: symbols=%s dry_run=%s", self.symbols, self.dry_run)
        return True

    def check_market_hours(self) -> bool:
        """Wait for the market to open (capped at max_market_wait_seconds),
        or report closed if it doesn't open within that window."""
        clock = self.client.get_clock()
        if clock["is_open"]:
            return True

        next_open = clock.get("next_open")
        wait_seconds = 0.0
        if isinstance(next_open, datetime):
            wait_seconds = max(0.0, (next_open - datetime.now(timezone.utc)).total_seconds())
        wait_seconds = min(wait_seconds, self.max_market_wait_seconds)

        if wait_seconds <= 0:
            return False

        logger.info("Market is closed; waiting up to %.0fs for open (next_open=%s)", wait_seconds, next_open)
        time.sleep(wait_seconds)
        return self.client.is_market_open()

    def load_or_train_hmm(self, symbol: str, bars: pd.DataFrame) -> HMMEngine:
        """Load a saved model if younger than model_stale_days, else fit
        fresh (on ``bars``) and save it."""
        path = self.model_dir / f"{symbol}_hmm.pkl"
        if path.exists():
            engine = build_hmm_template(self.config)
            engine.load_model(path)
            trained_at = engine.training_metadata.get("training_date")
            if trained_at is not None:
                age_days = (datetime.now(timezone.utc) - trained_at).days
                if age_days <= self.model_stale_days:
                    logger.info("Loaded HMM for %s (trained %s, %d day(s) old)", symbol, trained_at.date(), age_days)
                    self.hmm_last_trained[symbol] = trained_at
                    return engine
                logger.info("HMM for %s is %d day(s) old (> %d); retraining", symbol, age_days, self.model_stale_days)
        else:
            logger.info("No saved HMM for %s; training fresh", symbol)

        return self._train_and_save_hmm(symbol, bars)

    def _train_and_save_hmm(self, symbol: str, bars: pd.DataFrame) -> HMMEngine:
        features = self.feature_engineer.build_feature_set(bars)
        engine = build_hmm_template(self.config)
        engine.fit(features)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        engine.save_model(self.model_dir / f"{symbol}_hmm.pkl")
        self.hmm_last_trained[symbol] = engine.training_metadata["training_date"]
        logger.info("Trained HMM for %s: n_regimes=%d labels=%s", symbol, engine.n_regimes, engine.state_labels)
        return engine

    def retrain_if_due(self, symbol: str, bars: pd.DataFrame) -> None:
        """Weekly (in practice: model_stale_days-driven) retrain check,
        called once per newly-processed trading day."""
        trained_at = self.hmm_last_trained.get(symbol)
        if trained_at is None or (datetime.now(timezone.utc) - trained_at).days > self.model_stale_days:
            self.engines[symbol] = self._train_and_save_hmm(symbol, bars)
            self.orchestrators[symbol] = StrategyOrchestrator(self.strategy_config, self.engines[symbol].regime_info)

    def _fetch_daily_history(self, symbol: str) -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=TRAINING_LOOKBACK_DAYS)
        return self.market_data.get_historical_bars(symbol, self.timeframe, start, end)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def on_bar(self, symbol: str, bar: pd.Series) -> None:
        """Called on every live_bar_timeframe bar close. Once per new
        calendar day this re-runs the daily regime/signal pipeline;
        every tick it does lightweight housekeeping. Any unhandled
        exception is logged with a traceback, state is saved, and a
        CRITICAL log line stands in for an operator alert (monitoring.alerts
        is not implemented in this phase)."""
        try:
            self._last_bar_at[symbol] = _as_utc_datetime(bar.name)
            today = self._last_bar_at[symbol].date()

            if self._last_regime_date.get(symbol) != today:
                try:
                    self._process_new_trading_day(symbol)
                    self._last_regime_date[symbol] = today
                except Exception:
                    logger.exception(
                        "HMM/regime processing failed for %s; holding current regime this cycle.", symbol
                    )

            self._housekeeping_tick(symbol)
        except Exception:
            logger.critical("ALERT: unhandled error processing bar for %s", symbol, exc_info=True)
            self._save_state_snapshot()

    def _process_new_trading_day(self, symbol: str) -> None:
        bars = self._fetch_daily_history(symbol)
        self.bars_history[symbol] = bars
        self.retrain_if_due(symbol, bars)

        features = self.feature_engineer.build_feature_set(bars)
        if features.empty:
            logger.warning("No valid features yet for %s (still warming up); skipping regime update", symbol)
            return

        engine = self.engines[symbol]
        regime_state = engine.predict_regime_filtered(features)[-1]
        is_flickering = engine.is_flickering()
        self._last_regime_state[symbol] = regime_state
        self.position_tracker.update_current_regime(symbol, regime_state.label)

        orchestrator = self.orchestrators[symbol]
        signals = orchestrator.generate_signals([symbol], {symbol: bars}, regime_state, is_flickering)
        for sig in signals:
            self.process_signal(symbol, sig)

        self.update_trailing_stop(symbol, bars)

    def process_signal(self, symbol: str, sig: StrategySignal) -> None:
        if self._is_data_feed_stale(symbol):
            logger.warning("Data feed for %s looks stale; pausing new signals (existing stops untouched).", symbol)
            return

        portfolio_state = self.position_tracker.get_portfolio_state()
        decision = self.risk_manager.validate_signal(sig, portfolio_state)

        if not decision.approved:
            logger.warning("Signal rejected for %s: %s", symbol, decision.rejection_reason)
            return
        if decision.modifications:
            logger.info("Signal for %s modified by risk manager: %s", symbol, decision.modifications)

        if self.dry_run:
            logger.info(
                "[DRY RUN] Would submit order for %s: direction=%s size=%.1f%% leverage=%.2fx",
                symbol,
                decision.modified_signal.direction,
                decision.modified_signal.position_size_pct * 100,
                decision.modified_signal.leverage,
            )
            return

        trade_id = str(uuid.uuid4())
        self.position_tracker.register_order_submitted(symbol, decision.modified_signal.direction)
        try:
            result = self.order_executor.submit_bracket_order(decision.modified_signal, trade_id=trade_id)
            logger.info("Order submitted for %s (trade_id=%s): status=%s", symbol, trade_id, result.status)
        except Exception:
            logger.exception("Order submission failed for %s", symbol)

    def update_trailing_stop(self, symbol: str, bars: pd.DataFrame) -> None:
        """Tighten (never widen — enforced by OrderExecutor.modify_stop
        itself) the position's stop toward a fresh ATR-based level."""
        position = self.position_tracker.get_position(symbol)
        if position is None or len(bars) < ATR_WINDOW + 1:
            return
        atr = average_true_range(bars["high"], bars["low"], bars["close"], ATR_WINDOW).iloc[-1]
        if pd.isna(atr):
            return
        candidate_stop = float(bars["close"].iloc[-1]) - STOP_LOSS_ATR_MULTIPLE * float(atr)

        if self.dry_run:
            logger.info("[DRY RUN] Would attempt to tighten stop for %s to %.2f", symbol, candidate_stop)
            return
        if self.order_executor.modify_stop(symbol, candidate_stop):
            self.position_tracker.update_stop_level(symbol, candidate_stop)

    def check_circuit_breakers(self, symbol: Optional[str] = None) -> None:
        portfolio_state = self.position_tracker.get_portfolio_state()
        previous = self.risk_manager.circuit_breaker.check()
        regime_state = self._last_regime_state.get(symbol) if symbol else None
        new_state = self.risk_manager.circuit_breaker.update(
            portfolio_state, regime_label=regime_state.label if regime_state else None
        )

        newly_halted = (new_state.daily_halt_active and not previous.daily_halt_active) or (
            new_state.weekly_halt_active and not previous.weekly_halt_active
        )
        if newly_halted:
            logger.warning("Circuit breaker halt triggered — closing all positions.")
            if not self.dry_run:
                self.order_executor.close_all_positions()

    def _is_data_feed_stale(self, symbol: str) -> bool:
        last_bar_at = self._last_bar_at.get(symbol)
        if last_bar_at is None:
            return False
        max_age = STALE_FEED_MULTIPLE * _timeframe_to_timedelta(self.live_bar_timeframe)
        return datetime.now(timezone.utc) - last_bar_at > max_age

    def _housekeeping_tick(self, symbol: str) -> None:
        self.check_circuit_breakers(symbol)
        self._maybe_refresh_dashboard()
        self._save_state_snapshot()

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def _maybe_refresh_dashboard(self) -> None:
        now = time.monotonic()
        if now - self._last_dashboard_refresh < self.dashboard_refresh_seconds:
            return
        self._last_dashboard_refresh = now
        self.render_dashboard()

    def render_dashboard(self, console: Optional[Console] = None) -> None:
        console = console or Console()
        portfolio_state = self.position_tracker.get_portfolio_state()

        summary = Table(title="regime-trader — live status")
        summary.add_column("Metric")
        summary.add_column("Value", justify="right")
        summary.add_row("Mode", "DRY RUN" if self.dry_run else ("PAPER" if self.client.paper else "LIVE"))
        summary.add_row("Equity", f"${portfolio_state.equity:,.2f}")
        summary.add_row("Daily P&L", f"{portfolio_state.daily_pnl_pct:.2%}")
        summary.add_row("Weekly P&L", f"{portfolio_state.weekly_pnl_pct:.2%}")
        summary.add_row("Drawdown from Peak", f"{portfolio_state.drawdown_from_peak_pct:.2%}")
        summary.add_row("Circuit Breaker", "HALTED" if portfolio_state.circuit_breaker_status.trading_halted else "normal")
        summary.add_row("Open Positions", str(len(portfolio_state.positions)))
        console.print(summary)

        if self.position_tracker.positions:
            positions_table = Table(title="Open Positions")
            for col in ("Symbol", "Qty", "Entry", "Current", "Unrealized P&L", "Regime (entry -> now)"):
                positions_table.add_column(col)
            for symbol, tracked in self.position_tracker.positions.items():
                positions_table.add_row(
                    symbol,
                    f"{tracked.quantity:g}",
                    f"{tracked.entry_price:.2f}",
                    f"{tracked.current_price:.2f}",
                    f"{tracked.unrealized_pnl_pct:.2%}",
                    f"{tracked.regime_at_entry} -> {tracked.regime_current}",
                )
            console.print(positions_table)

    # ------------------------------------------------------------------
    # State persistence (crash recovery + `dashboard` subcommand)
    # ------------------------------------------------------------------

    def _build_state_snapshot(self) -> TradingState:
        return TradingState(
            last_regime_date={s: d.isoformat() for s, d in self._last_regime_date.items()},
            hmm_last_trained={s: t.isoformat() for s, t in self.hmm_last_trained.items()},
            trades_today=self.position_tracker.trades_today,
            daily_start_equity=self.position_tracker._daily_start_equity,
            weekly_start_equity=self.position_tracker._weekly_start_equity,
            peak_equity=self.position_tracker._peak_equity,
            circuit_breaker_state=dataclasses.asdict(self.risk_manager.circuit_breaker.check()),
            open_positions=[
                {
                    "symbol": symbol,
                    "quantity": p.quantity,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "unrealized_pnl_pct": p.unrealized_pnl_pct,
                    "regime_at_entry": p.regime_at_entry,
                    "regime_current": p.regime_current,
                }
                for symbol, p in self.position_tracker.positions.items()
            ],
            session_started_at=self._session_started_at.isoformat(),
            last_updated_at=datetime.now(timezone.utc).isoformat(),
            dry_run=self.dry_run,
        )

    def _save_state_snapshot(self) -> None:
        try:
            state = self._build_state_snapshot()
            self.state_snapshot_path.write_text(json.dumps(dataclasses.asdict(state), indent=2, default=str))
        except Exception:
            logger.exception("Failed to save state snapshot to %s", self.state_snapshot_path)

    def _restore_state_snapshot(self) -> None:
        if not self.state_snapshot_path.exists():
            return
        try:
            raw = json.loads(self.state_snapshot_path.read_text())
        except Exception:
            logger.exception("Failed to read state snapshot at %s; ignoring", self.state_snapshot_path)
            return

        logger.info(
            "Recovered previous session state from %s (last updated %s)",
            self.state_snapshot_path,
            raw.get("last_updated_at"),
        )
        if raw.get("daily_start_equity") is not None:
            self.position_tracker._daily_start_equity = raw["daily_start_equity"]
        if raw.get("weekly_start_equity") is not None:
            self.position_tracker._weekly_start_equity = raw["weekly_start_equity"]
        if raw.get("peak_equity") is not None:
            self.position_tracker._peak_equity = raw["peak_equity"]
        self.position_tracker.trades_today = raw.get("trades_today", self.position_tracker.trades_today)
        for symbol, iso_date in (raw.get("last_regime_date") or {}).items():
            try:
                self._last_regime_date[symbol] = date.fromisoformat(iso_date)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Run / shutdown
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Blocking: start up, subscribe to bars, and run until a
        SIGINT/SIGTERM (or the stream itself) stops it."""
        if not self.startup():
            return
        self._running = True

        def _handle_shutdown_signal(signum, _frame) -> None:
            logger.info("Received signal %s; shutting down...", signum)
            self._running = False
            self.market_data.stop_stream()

        signal_module.signal(signal_module.SIGINT, _handle_shutdown_signal)
        signal_module.signal(signal_module.SIGTERM, _handle_shutdown_signal)

        self.market_data.subscribe_bars(self.symbols, self.live_bar_timeframe, self.on_bar)
        try:
            self.market_data.run_stream()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Close WebSocket connections, leave positions/stops alone, save
        state, print a session summary."""
        logger.info("Shutting down: closing WebSocket connections (positions and stops left in place)...")
        try:
            self.market_data.stop_stream()
        except Exception:
            logger.exception("Error stopping market data stream")
        try:
            self.position_tracker.stop_streaming()
        except Exception:
            logger.exception("Error stopping position tracker stream")

        self._save_state_snapshot()
        self._print_session_summary()

    def _print_session_summary(self, console: Optional[Console] = None) -> None:
        console = console or Console()
        portfolio_state = self.position_tracker.get_portfolio_state()
        console.print(
            f"[bold]Session summary[/bold]: started {self._session_started_at.isoformat()}, "
            f"equity=${portfolio_state.equity:,.2f}, trades_today={self.position_tracker.trades_today}, "
            f"open_positions={len(portfolio_state.positions)}"
        )


def _as_utc_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return pd.Timestamp(value).to_pydatetime().replace(tzinfo=timezone.utc)


def _timeframe_to_timedelta(timeframe: str) -> timedelta:
    digits = "".join(c for c in timeframe if c.isdigit()) or "1"
    unit = "".join(c for c in timeframe if c.isalpha()).lower()
    amount = int(digits)
    if unit.startswith("min"):
        return timedelta(minutes=amount)
    if unit.startswith("hour"):
        return timedelta(hours=amount)
    if unit.startswith("day"):
        return timedelta(days=amount)
    if unit.startswith("week"):
        return timedelta(weeks=amount)
    raise ValueError(f"Unrecognized timeframe: {timeframe!r}")


def build_trading_session(config: dict, symbols: list[str], dry_run: bool = False, max_retries: int = 3) -> TradingSession:
    """Construct and wire together all live-trading components."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set. Copy .env.example to .env and fill them in."
        )
    paper = config["broker"]["paper_trading"]
    confirm_live = None if paper else os.environ.get("ALPACA_LIVE_TRADING_CONFIRMATION")

    client = AlpacaClient(api_key, secret_key, paper=paper, confirm_live_trading=confirm_live, max_retries=max_retries)
    market_data = MarketDataClient(client)
    order_executor = OrderExecutor(client)
    risk_manager = build_risk_manager(config)
    position_tracker = PositionTracker(client, circuit_breaker=risk_manager.circuit_breaker)

    return TradingSession(
        config, client, market_data, order_executor, position_tracker, risk_manager, symbols, dry_run=dry_run
    )


def build_app(config: dict, symbols: Optional[list[str]] = None, dry_run: bool = False) -> TradingSession:
    """Construct and wire together all application components."""
    return build_trading_session(config, symbols or config["broker"]["symbols"], dry_run=dry_run)


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


def cmd_run(args: argparse.Namespace, config: dict) -> None:
    """Start the live/paper trading loop (Phase 7 startup -> main loop -> shutdown)."""
    symbols = args.symbols or config["broker"]["symbols"]
    session = build_trading_session(config, symbols, dry_run=args.dry_run)
    session.run()


def cmd_train_only(args: argparse.Namespace, config: dict) -> None:
    """Train (or retrain) the HMM for each symbol against fresh history, then exit."""
    console = Console()
    symbols = args.symbols or config["broker"]["symbols"]
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=TRAINING_LOOKBACK_DAYS)).isoformat()

    console.print(f"[bold]Loading price data[/bold] for {symbols} from {start} to {end}...")
    price_data = load_price_data(symbols, start, end, config, data_dir=args.data_dir)

    fe = FeatureEngineer()
    model_dir = Path(config["live"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        console.print(f"[bold]Training HMM for {symbol}...[/bold]")
        features = fe.build_feature_set(price_data[symbol])
        engine = build_hmm_template(config)
        engine.fit(features)
        engine.save_model(model_dir / f"{symbol}_hmm.pkl")
        console.print(
            f"  n_regimes={engine.n_regimes} labels={engine.state_labels} "
            f"bic={engine.training_metadata['bic']:.1f} -> {model_dir / f'{symbol}_hmm.pkl'}"
        )


def cmd_dashboard(args: argparse.Namespace, config: dict) -> None:
    """Print the last-known status of a running (or previously run) instance
    from its state_snapshot.json — a lightweight way to check on a session
    without attaching to its process."""
    console = Console()
    path = Path(args.state_file or config["live"]["state_snapshot_path"])
    if not path.exists():
        console.print(f"[yellow]No state snapshot found at {path}. Is an instance running?[/yellow]")
        return

    raw = json.loads(path.read_text())

    summary = Table(title="regime-trader — last known status")
    summary.add_column("Field")
    summary.add_column("Value")
    summary.add_row("Mode", "DRY RUN" if raw.get("dry_run") else "live/paper")
    summary.add_row("Session started", str(raw.get("session_started_at")))
    summary.add_row("Last updated", str(raw.get("last_updated_at")))
    summary.add_row("Trades today", str(raw.get("trades_today")))
    summary.add_row("Peak equity", f"${raw['peak_equity']:,.2f}" if raw.get("peak_equity") else "n/a")
    cb = raw.get("circuit_breaker_state") or {}
    halted = cb.get("daily_halt_active") or cb.get("weekly_halt_active") or cb.get("peak_halt_active")
    summary.add_row("Circuit breaker", "HALTED" if halted else "normal")
    console.print(summary)

    positions = raw.get("open_positions") or []
    if positions:
        positions_table = Table(title="Open Positions (as of last snapshot)")
        for col in ("Symbol", "Qty", "Entry", "Current", "Unrealized P&L", "Regime (entry -> now)"):
            positions_table.add_column(col)
        for p in positions:
            positions_table.add_row(
                p["symbol"],
                f"{p['quantity']:g}",
                f"{p['entry_price']:.2f}",
                f"{p['current_price']:.2f}",
                f"{p['unrealized_pnl_pct']:.2%}",
                f"{p['regime_at_entry']} -> {p['regime_current']}",
            )
        console.print(positions_table)
    else:
        console.print("No open positions as of the last snapshot.")


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

    run_parser = subparsers.add_parser("run", help="Start the live/paper trading loop")
    run_parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to trade (default: settings.yaml broker.symbols)")
    run_parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Run the full pipeline (signals, risk checks) without submitting orders")

    train_parser = subparsers.add_parser("train-only", help="Train the HMM for each symbol and exit")
    train_parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to train (default: settings.yaml broker.symbols)")
    train_parser.add_argument("--data-dir", default=None, dest="data_dir", help="Load OHLCV bars from {data_dir}/{symbol}.csv instead of Alpaca")

    dashboard_parser = subparsers.add_parser("dashboard", help="Show the last known status of a running instance")
    dashboard_parser.add_argument("--state-file", default=None, dest="state_file", help="Path to state_snapshot.json (default: settings.yaml live.state_snapshot_path)")

    return parser


def main() -> None:
    """Parse CLI arguments and start regime-trader."""
    load_dotenv()
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "backtest":
        cmd_backtest(args, config)
    elif args.command == "run":
        cmd_run(args, config)
    elif args.command == "train-only":
        cmd_train_only(args, config)
    elif args.command == "dashboard":
        cmd_dashboard(args, config)


if __name__ == "__main__":
    main()
