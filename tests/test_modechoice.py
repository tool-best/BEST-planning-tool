"""Tests for data generation rules, estimation, prediction, and sensitivity."""

import numpy as np
import pandas as pd
import pytest

from modechoice import config
from modechoice.generate_data import generate
from modechoice.mnl import estimate, predicted_shares, apply_adjustments, sensitivity


@pytest.fixture(scope="module")
def data():
    return generate(n=1000, seed=42)


@pytest.fixture(scope="module")
def fitted(data):
    return estimate(data)


# ---- generation rules ----
def test_row_count_and_reproducible():
    a = generate(n=500, seed=1)
    b = generate(n=500, seed=1)
    assert len(a) == 500
    pd.testing.assert_frame_equal(a, b)


def test_walk_and_bike_availability_gates(data):
    # Walk available iff distance < 3 mi; Bike iff distance < 6 mi.
    assert (data["av_Walk"] == (data["distance_mi"] < config.WALK_MAX_MI).astype(int)).all()
    assert (data["av_Bike"] == (data["distance_mi"] < config.BIKE_MAX_MI).astype(int)).all()


def test_bike_walk_have_no_cost_or_parking(data):
    for m in ("Bike", "Walk"):
        assert (data[f"cost_{m}"] == 0).all()
        assert (data[f"park_{m}"] == 0).all()


def test_parking_only_on_allowed_modes(data):
    for m in config.MODES:
        if not config.HAS_PARK[m]:
            assert (data[f"park_{m}"] == 0).all()
    assert (data["park_Drive"] > 0).all()


def test_no_one_chooses_an_unavailable_mode(data):
    for _, r in data.iterrows():
        assert r[f"av_{r['chosen_mode']}"] == 1


# ---- estimation ----
def test_signs_and_significance(fitted):
    t = fitted.table
    for p in ("b_time", "b_cost", "b_park"):
        assert t.loc[p, "coef"] < 0                 # disutilities are negative
        assert abs(t.loc[p, "t_stat"]) > 2          # and precisely estimated


def test_recovers_true_params_within_tolerance(fitted):
    # Level-of-service coefficients should be in the right neighborhood.
    assert fitted.params[0] == pytest.approx(config.TRUE_PARAMS["b_time"], abs=0.02)
    assert fitted.params[1] == pytest.approx(config.TRUE_PARAMS["b_cost"], abs=0.06)


def test_predicted_matches_observed_marginals(data, fitted):
    obs = data["chosen_mode"].value_counts(normalize=True).reindex(config.MODES)
    pred = predicted_shares(data, fitted.params)
    assert np.allclose(obs.values, pred.values, atol=0.01)


def test_shares_sum_to_one(data, fitted):
    assert predicted_shares(data, fitted.params).sum() == pytest.approx(1.0)


# ---- sensitivity behaves sensibly ----
def test_raising_drive_cost_lowers_drive_share(data, fitted):
    s = sensitivity(data, fitted.params, "Drive", "park", 0.50)
    assert s.loc["Drive", "abs_change"] < 0
    assert s.loc["PublicTransit", "abs_change"] > 0   # substitutes gain


def test_zero_adjustment_is_a_noop(data, fitted):
    base = predicted_shares(data, fitted.params)
    same = predicted_shares(apply_adjustments(data, {("Drive", "cost"): 0.0}), fitted.params)
    assert np.allclose(base.values, same.values)


def test_sensitivity_shares_still_sum_to_one(data, fitted):
    s = sensitivity(data, fitted.params, "PublicTransit", "time", -0.30)
    assert s["new"].sum() == pytest.approx(1.0)


# ---- web export payload ----
def test_export_payload_is_consistent(data, fitted):
    from modechoice.export_web import build_payload
    p = build_payload()
    assert len(p["modes"]) == len(p["basePred"]) == len(p["observed"])
    assert p["n"] == len(data)
    assert abs(sum(p["basePred"]) - 1.0) < 1e-6
    # ASC vector is aligned to modes, reference (Drive) fixed at 0
    assert p["coef"]["asc"][p["modes"].index("Drive")] == 0.0
