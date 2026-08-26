"""Tests for core.hmm_engine.HMMEngine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.hmm_engine import HMMEngine, RegimeState
from tests.conftest import TEST_HMM_KWARGS, fresh_inference_copy


def test_model_selection_picks_valid_candidate(trained_engine: HMMEngine) -> None:
    """select_model should choose one of the configured n_candidates state counts."""
    assert trained_engine.n_regimes in TEST_HMM_KWARGS["n_candidates"]


def test_candidate_bic_scores_are_logged_and_best_is_selected(trained_engine: HMMEngine) -> None:
    """All candidate BIC scores should be recorded, and the selected model
    should be the one with the lowest BIC among them."""
    assert len(trained_engine.candidate_results) >= 1
    bics = [c["bic"] for c in trained_engine.candidate_results]
    selected_bic = trained_engine.training_metadata["bic"]
    assert selected_bic == min(bics)


def test_labels_assigned_match_regime_count(trained_engine: HMMEngine) -> None:
    """Assigned labels should be a full permutation of the label set for n_regimes."""
    from core.hmm_engine import REGIME_LABELS_BY_COUNT

    expected = set(REGIME_LABELS_BY_COUNT[trained_engine.n_regimes])
    assert set(trained_engine.state_labels.values()) == expected


def test_labels_sorted_by_ascending_mean_return(trained_engine: HMMEngine) -> None:
    """The state labeled CRASH/BEAR-most should have a lower mean return than
    the state labeled BULL/EUPHORIA-most (labels sorted ascending by return)."""
    from core.hmm_engine import REGIME_LABELS_BY_COUNT

    labels_ascending = REGIME_LABELS_BY_COUNT[trained_engine.n_regimes]
    returns_in_label_order = [
        trained_engine.regime_info[state].expected_return
        for label in labels_ascending
        for state, lbl in trained_engine.state_labels.items()
        if lbl == label
    ]
    assert returns_in_label_order == sorted(returns_in_label_order)


def test_predict_regime_returns_confidence_in_range(
    trained_engine: HMMEngine, synthetic_features: pd.DataFrame
) -> None:
    """predict_regime should return a confidence (probability) between 0 and 1."""
    engine = fresh_inference_copy(trained_engine)
    state = engine.predict_regime(synthetic_features)
    assert isinstance(state, RegimeState)
    assert 0.0 <= state.probability <= 1.0
    assert abs(sum(state.state_probabilities.values()) - 1.0) < 1e-6


def test_stability_filter_suppresses_single_bar_switches(trained_engine: HMMEngine) -> None:
    """apply_stability_filter (via the filtered-prediction loop) should not
    switch the displayed regime on a single-bar blip that reverts immediately."""
    engine = fresh_inference_copy(trained_engine)
    labels = list(engine.state_labels.values())
    a, b = labels[0], labels[1]
    raw_sequence = [a, a, a, b, a, a, a]  # single-bar blip to b, then back to a

    displayed = []
    for raw_label in raw_sequence:
        label, _, _, _ = engine._advance_stability_filter(raw_label)
        displayed.append(label)

    assert displayed == [a, a, a, a, a, a, a]


def test_stability_filter_confirms_after_persistence(trained_engine: HMMEngine) -> None:
    """A raw regime change that persists for stability_bars bars should be confirmed."""
    engine = fresh_inference_copy(trained_engine)
    labels = list(engine.state_labels.values())
    a, b = labels[0], labels[1]
    raw_sequence = [a, a] + [b] * engine.stability_bars

    displayed = []
    for raw_label in raw_sequence:
        label, is_confirmed, _, _ = engine._advance_stability_filter(raw_label)
        displayed.append((label, is_confirmed))

    assert displayed[-1] == (b, True)
    assert displayed[-2][0] == a  # still showing old regime just before confirmation


def test_flicker_detection_flags_excessive_switching(trained_engine: HMMEngine) -> None:
    """is_flickering should return True when confirmed switches exceed flicker_threshold
    within the trailing flicker_window bars."""
    engine = fresh_inference_copy(trained_engine)
    labels = list(engine.state_labels.values())
    if len(labels) < 2:
        return
    a, b = labels[0], labels[1]

    # Alternate regimes every stability_bars bars, well within one flicker_window,
    # producing more confirmed changes than flicker_threshold allows.
    n_switches_needed = engine.flicker_threshold + 1
    raw_sequence: list[str] = [a] * engine.stability_bars
    current, other = b, a
    for _ in range(n_switches_needed):
        raw_sequence += [current] * engine.stability_bars
        current, other = other, current

    assert len(raw_sequence) <= engine.flicker_window

    for raw_label in raw_sequence:
        _, _, _, changed_now = engine._advance_stability_filter(raw_label)
        engine._processed_bars += 1
        if changed_now:
            engine._change_event_positions.append(engine._processed_bars - 1)

    assert engine.get_regime_flicker_rate() > engine.flicker_threshold
    assert engine.is_flickering() is True


def test_transition_matrix_rows_sum_to_one(trained_engine: HMMEngine) -> None:
    """get_transition_matrix should return a valid stochastic matrix."""
    matrix = trained_engine.get_transition_matrix()
    row_sums = matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_save_and_load_model_round_trip(
    trained_engine: HMMEngine, synthetic_features: pd.DataFrame, tmp_path
) -> None:
    """A model saved with save_model and reloaded with load_model should
    produce identical predictions."""
    path = tmp_path / "model.pkl"
    trained_engine.save_model(path)

    reloaded = HMMEngine(**TEST_HMM_KWARGS)
    reloaded.load_model(path)

    original = fresh_inference_copy(trained_engine).predict_regime(synthetic_features)
    restored = reloaded.predict_regime(synthetic_features)

    assert restored.label == original.label
    assert restored.state_id == original.state_id
    assert abs(restored.probability - original.probability) < 1e-9


def test_fit_raises_with_too_few_bars() -> None:
    """fit() should raise if there is not enough clean training data."""
    engine = HMMEngine(**TEST_HMM_KWARGS)
    too_few = pd.DataFrame(
        np.random.randn(10, 3), columns=["return_1", "return_5", "return_20"]
    )
    try:
        engine.fit(too_few)
        assert False, "expected ValueError for insufficient training data"
    except ValueError:
        pass
