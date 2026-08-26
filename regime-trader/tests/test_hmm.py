"""Tests for core.hmm_engine.HMMEngine."""

from __future__ import annotations

import pytest


def test_model_selection_picks_valid_candidate() -> None:
    """select_model should choose one of the configured n_candidates state counts."""
    ...


def test_predict_regime_returns_confidence_in_range() -> None:
    """predict_regime should return a confidence between 0 and 1."""
    ...


def test_stability_filter_suppresses_single_bar_switches() -> None:
    """apply_stability_filter should not switch regime on a single-bar blip."""
    ...


def test_flicker_detection_flags_excessive_switching() -> None:
    """detect_flicker should return True when switches exceed flicker_threshold."""
    ...
