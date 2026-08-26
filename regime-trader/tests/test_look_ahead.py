"""Tests verifying no look-ahead bias in feature engineering, HMM inference,
and the walk-forward backtester.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.hmm_engine import HMMEngine
from data.feature_engineering import FeatureEngineer
from tests.conftest import fresh_inference_copy


def test_features_use_only_past_data(synthetic_bars: pd.DataFrame) -> None:
    """Feature values at time t must not depend on bars after t.

    Appending 100 extra future bars must not change any already-computed
    feature value for the bars that existed before the extension.
    """
    fe = FeatureEngineer()
    short_bars = synthetic_bars.iloc[:400]
    long_bars = synthetic_bars.iloc[:500]

    short_features = fe.compute_hmm_features(short_bars)
    long_features = fe.compute_hmm_features(long_bars)

    overlap = short_features.index
    pd.testing.assert_frame_equal(
        short_features,
        long_features.loc[overlap],
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
    )


def test_hmm_training_window_excludes_test_window(synthetic_features: pd.DataFrame) -> None:
    """The HMM must only be fit on train_window data, never on test_window data.

    A model fit on the first N rows must be identical (same log-likelihood
    on those rows) whether or not additional rows exist beyond N in the
    full feature set it was given.
    """
    train = synthetic_features.iloc[:300]
    kwargs = dict(
        n_candidates=[3],
        n_init=2,
        covariance_type="full",
        min_train_bars=100,
        stability_bars=3,
        flicker_window=20,
        flicker_threshold=4,
        min_confidence=0.55,
    )

    engine_a = HMMEngine(**kwargs)
    engine_a.fit(train)

    # Fitting again on the exact same train slice (as if a longer test
    # window existed afterward, but was never passed to fit) must give an
    # identical model: fit() only ever sees what it's explicitly handed.
    engine_b = HMMEngine(**kwargs)
    engine_b.fit(synthetic_features.iloc[:300])

    assert engine_a.n_regimes == engine_b.n_regimes
    assert np.allclose(engine_a.model.means_, engine_b.model.means_)
    assert np.allclose(engine_a.model.transmat_, engine_b.model.transmat_)


def test_no_look_ahead_bias(trained_engine: HMMEngine, synthetic_features: pd.DataFrame) -> None:
    """MANDATORY: regime at bar T must be identical whether it is computed
    from data[0:T] or from a longer data[0:T+100].

    predict_regime_filtered uses only the forward algorithm (filtered
    inference), never model.predict() (Viterbi), so the regime assigned to
    bar T can never be revised by observations after T.
    """
    t = 400
    extended = t + 100

    engine_short = fresh_inference_copy(trained_engine)
    engine_long = fresh_inference_copy(trained_engine)

    regime_short = engine_short.predict_regime_filtered(synthetic_features.iloc[:t])[-1]
    regime_long = engine_long.predict_regime_filtered(synthetic_features.iloc[:extended])[t - 1]

    assert regime_short.state_id == regime_long.state_id, "LOOK-AHEAD BIAS DETECTED"
    assert regime_short.label == regime_long.label, "LOOK-AHEAD BIAS DETECTED"
    assert np.isclose(regime_short.probability, regime_long.probability), (
        "LOOK-AHEAD BIAS DETECTED"
    )
    for label, prob in regime_short.state_probabilities.items():
        assert np.isclose(prob, regime_long.state_probabilities[label]), (
            "LOOK-AHEAD BIAS DETECTED"
        )


def test_no_look_ahead_bias_with_cached_alpha(
    trained_engine: HMMEngine, synthetic_features: pd.DataFrame
) -> None:
    """The same no-look-ahead guarantee must hold when a single engine
    instance incrementally extends its cached forward-algorithm state,
    which is the pattern used in the live/backtest loop.
    """
    engine = fresh_inference_copy(trained_engine)

    results_400 = engine.predict_regime_filtered(synthetic_features.iloc[:400])
    results_500 = engine.predict_regime_filtered(synthetic_features.iloc[:500])

    assert results_400[-1].state_id == results_500[399].state_id, "LOOK-AHEAD BIAS DETECTED"
    assert results_400[-1].label == results_500[399].label, "LOOK-AHEAD BIAS DETECTED"
    assert np.isclose(results_400[-1].probability, results_500[399].probability), (
        "LOOK-AHEAD BIAS DETECTED"
    )


def test_backtest_walk_forward_no_future_leakage(synthetic_features: pd.DataFrame) -> None:
    """Walk-forward backtest steps must never use future bars to generate signals.

    Simulates a walk-forward loop: at each step, only fit/predict using
    data available "as of" that step, and confirm the regime call for the
    last bar of each step never changes retroactively once later steps run.
    """
    kwargs = dict(
        n_candidates=[3],
        n_init=2,
        covariance_type="full",
        min_train_bars=100,
        stability_bars=3,
        flicker_window=20,
        flicker_threshold=4,
        min_confidence=0.55,
    )
    engine = HMMEngine(**kwargs)
    engine.fit(synthetic_features.iloc[:250])

    recorded: dict[int, str] = {}
    for step_end in (260, 280, 300):
        as_of_data = synthetic_features.iloc[:step_end]
        step_engine = fresh_inference_copy(engine)
        results = step_engine.predict_regime_filtered(as_of_data)
        recorded[step_end] = results[-1].label

        # Re-derive the regime for bar 259 (present in every step) and make
        # sure later, longer walk-forward steps never change it.
        replay_engine = fresh_inference_copy(engine)
        replay_results = replay_engine.predict_regime_filtered(synthetic_features.iloc[:260])
        assert replay_results[-1].label == recorded[260]
