"""Tests. Mechanics tests use small hand-built fixtures; data tests use the real SQLite DB
and are skipped if it has not been downloaded."""
import numpy as np
import pandas as pd
import pytest

from sfl.config import DB_PATH, SEED
from sfl.evaluation import calibration_table, classification_metrics
from sfl.features import FEATURES, TEAM_FEATURES, build_game_features, rolling_team_features
from sfl.simulation import simulate_season

needs_db = pytest.mark.skipif(not DB_PATH.exists(), reason="run `python -m sfl.data` first")


@pytest.fixture(scope="module")
def real():
    from sfl.pipeline import load_data
    pbp, sched, teams, _ = load_data()
    return pbp, sched, teams


def _toy_team_games():
    days = pd.date_range("2024-09-01", periods=6, freq="7D")
    rows = []
    for i, d in enumerate(days):
        rows.append({"team": "AAA", "game_id": f"g{i}", "gameday": d, "off_epa": float(i),
                     "def_epa": 0.0, "off_sr": 0.5, "def_sr": 0.5, "off_ppd": 2.0, "def_ppd": 2.0,
                     "off_to_rate": 0.01, "def_to_rate": 0.01, "off_ypp": 5.0, "def_ypp": 5.0,
                     "win": float(i % 2), "margin": float(i)})
    return pd.DataFrame(rows)


def test_rolling_features_use_only_prior_games():
    out = rolling_team_features(_toy_team_games())
    assert np.isnan(out.loc[0, "roll_off_epa"])
    # Game 3 sees games 0-2 only: mean(0, 1, 2) = 1
    assert out.loc[3, "roll_off_epa"] == pytest.approx(1.0)
    # Form window of 4 at game 5 sees games 1-4 -> margins mean(1..4) = 2.5
    assert out.loc[5, "form_margin"] == pytest.approx(2.5)


@needs_db
def test_no_lookahead_on_real_data(real):
    """Corrupting every game on/after a cutoff date must not change features before it."""
    pbp, sched, _ = real
    base = build_game_features(pbp, sched)
    cutoff = pd.Timestamp("2023-11-01")
    future_ids = set(sched.loc[pd.to_datetime(sched["gameday"]) >= cutoff, "game_id"])
    pbp2, sched2 = pbp.copy(), sched.copy()
    pbp2.loc[pbp2["game_id"].isin(future_ids), "epa"] = 99.0
    m = sched2["game_id"].isin(future_ids)
    sched2.loc[m, "home_score"] = 0
    sched2.loc[m, "away_score"] = 99
    sched2.loc[m, "result"] = -99
    pert = build_game_features(pbp2, sched2)

    on_or_before = base["gameday"] <= cutoff  # games on cutoff day use only earlier games
    a = base.loc[on_or_before, FEATURES].reset_index(drop=True)
    b = pert.loc[pert["gameday"] <= cutoff, FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    later = base["gameday"] > cutoff
    assert not np.allclose(base.loc[later, FEATURES].to_numpy(),
                           pert.loc[pert["gameday"] > cutoff, FEATURES].to_numpy())


@needs_db
def test_feature_shapes(real):
    pbp, sched, _ = real
    games = build_game_features(pbp, sched)
    assert games["game_id"].is_unique
    assert len(FEATURES) == len(TEAM_FEATURES) + 2 == 14
    assert games[FEATURES].shape == (len(games), 14)
    assert not games[FEATURES].isna().any().any()
    # Only games where a team has no prior game (2021 week 1) are dropped.
    dropped = set(sched["game_id"]) - set(games["game_id"])
    assert dropped == set(sched.loc[(sched["season"] == 2021) & (sched["week"] == 1), "game_id"])


def test_calibration_bins_sum_to_one():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.random(500), [0.0, 1.0, 0.5]])
    y = (rng.random(len(p)) < p).astype(int)
    cal = calibration_table(y, p, n_bins=10)
    assert len(cal) == 10
    assert cal["frac"].sum() == pytest.approx(1.0)
    assert cal["n"].sum() == len(p)


def test_simulation_reproducible_with_seed_42():
    start = pd.Series([3.0, 2.0, 4.0, 1.0], index=["A", "B", "C", "D"])
    home, away = ["A", "C", "B", "D"], ["B", "D", "C", "A"]
    p = np.array([0.6, 0.55, 0.4, 0.7])
    div = pd.Series({"A": "X", "B": "X", "C": "Y", "D": "Y"})
    s1, w1 = simulate_season(start, home, away, p, div, n_sims=10_000, seed=SEED)
    s2, w2 = simulate_season(start, home, away, p, div, n_sims=10_000, seed=SEED)
    _, w3 = simulate_season(start, home, away, p, div, n_sims=10_000, seed=SEED + 1)
    np.testing.assert_array_equal(w1, w2)
    pd.testing.assert_frame_equal(s1, s2)
    assert not np.array_equal(w1, w3)
    # Each sim distributes exactly one win per game.
    np.testing.assert_allclose(w1.sum(axis=1), start.sum() + len(p))
    assert s1.groupby(div.values)["p_division"].sum().tolist() == pytest.approx([1.0, 1.0])


@needs_db
def test_models_beat_mean_baseline_log_loss():
    from sfl.pipeline import run_pipeline
    m = run_pipeline(log_mlflow=False)["metrics"].set_index("model")
    base = m.loc["mean_baseline", "log_loss"]
    for name in ("logreg", "xgboost", "spread_ridge"):
        assert m.loc[name, "log_loss"] < base, name


def test_classification_metrics_perfect_and_constant():
    y = np.array([1, 0, 1, 0])
    perfect = classification_metrics(y, y.astype(float))
    assert perfect["accuracy"] == 1.0 and perfect["brier"] == pytest.approx(0.0, abs=1e-9)
    const = classification_metrics(y, np.full(4, 0.5))
    assert const["log_loss"] == pytest.approx(np.log(2))
