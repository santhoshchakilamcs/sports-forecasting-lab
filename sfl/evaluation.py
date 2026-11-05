"""Metrics: classification scores, calibration, against-the-spread, and feature drift."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import brier_score_loss, log_loss


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width probability bins: mean predicted vs actual win rate.

    ``frac`` is each bin's share of games and sums to 1.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "n": int(m.sum()),
            "frac": float(m.mean()) if len(p) else 0.0,
            "mean_pred": float(p[m].mean()) if m.any() else np.nan,
            "actual_rate": float(y[m].mean()) if m.any() else np.nan,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(cal: pd.DataFrame) -> float:
    c = cal[cal["n"] > 0]
    return float((c["frac"] * (c["mean_pred"] - c["actual_rate"]).abs()).sum())


def against_the_spread(result: np.ndarray, spread_line: np.ndarray,
                       model_margin: np.ndarray) -> dict[str, float]:
    """How often the model's side covers the closing line.

    Lines use nflverse convention: positive = home favored, comparable to ``result``
    (home minus away). The model takes the home side when its predicted margin exceeds the
    line, otherwise the away side. Pushes and exact model/line ties are excluded.
    """
    result, line, mm = map(lambda a: np.asarray(a, dtype=float), (result, spread_line, model_margin))
    push = result == line
    no_pick = mm == line
    pick_home = mm > line
    home_cover = result > line
    valid = ~push & ~no_pick
    wins = int(np.sum((pick_home == home_cover) & valid))
    n = int(valid.sum())
    return {
        "ats_n": n,
        "ats_wins": wins,
        "ats_losses": n - wins,
        "ats_pushes": int(push.sum()),
        "ats_hit_rate": wins / n if n else float("nan"),
        "ats_pvalue_vs_50": float(binomtest(wins, n, 0.5).pvalue) if n else float("nan"),
    }


def drift_report(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                 threshold: float = 0.25) -> pd.DataFrame:
    """Standardized mean difference of each feature (test vs train). A flag, never a failure."""
    mu_tr, sd_tr, mu_te = train[features].mean(), train[features].std(ddof=1), test[features].mean()
    smd = (mu_te - mu_tr) / sd_tr.replace(0, np.nan)
    return pd.DataFrame({
        "feature": features, "train_mean": mu_tr.values, "test_mean": mu_te.values,
        "std_mean_diff": smd.values, "drift_flag": (smd.abs() > threshold).values,
    })
