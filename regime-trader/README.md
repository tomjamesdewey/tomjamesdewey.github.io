# regime-trader

An HMM regime-based trading bot: detects market volatility/trend regimes
with a Gaussian Hidden Markov Model and allocates a portfolio accordingly,
trading through Alpaca (paper or live).

## Status

Phase 1: project scaffolding. Modules are stubs only — no trading logic is
implemented yet.

## Project layout

```
regime-trader/
├── config/          # settings.yaml, credentials.yaml.example
├── core/            # HMM engine, regime strategies, risk manager, signal generator
├── broker/          # Alpaca client, order executor, position tracker
├── data/            # Market data fetching, feature engineering
├── monitoring/       # Logging, terminal dashboard, alerts
├── backtest/        # Walk-forward backtester, performance analytics, stress tests
├── tests/           # Unit tests
└── main.py          # Entry point
```

## Setup

1. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your Alpaca API credentials.
3. Copy `config/credentials.yaml.example` to `config/credentials.yaml` and
   fill in real values (this file is gitignored).
4. Review and adjust `config/settings.yaml` for your desired parameters.

## Usage

```
python main.py
```
