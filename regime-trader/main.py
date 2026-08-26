"""Entry point for regime-trader.

Loads configuration and credentials, wires up the core engine, broker,
data, monitoring, and risk components, and starts the trading loop.
"""

from __future__ import annotations

import argparse


def load_config(config_path: str) -> dict:
    """Load and parse settings.yaml into a configuration dict."""
    ...


def build_app(config: dict) -> None:
    """Construct and wire together all application components."""
    ...


def main() -> None:
    """Parse CLI arguments and start regime-trader."""
    ...


if __name__ == "__main__":
    main()
