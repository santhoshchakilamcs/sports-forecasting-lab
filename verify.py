"""Re-run the full pipeline from data/sports.db and print the verified metrics.

Every number quoted in README.md is copied from this script's output
(also saved to reports/verify_output.txt and reports/metrics.json).
The JSON includes test-game predictions for the database-free public dashboard.

Usage:  python verify.py [--no-mlflow]
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import warnings
from contextlib import redirect_stdout

import pandas as pd

from sfl.config import (DB_PATH, FORM_WINDOW, REPORTS_DIR, ROLLING_WINDOW, SIM_CUTOFF_WEEK,
                        TEST_SEASON, TRAIN_SEASONS)
from sfl.pipeline import metrics_to_json, run_pipeline


def report(r: dict) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("=" * 78)
    print("SPORTS FORECASTING LAB - VERIFIED RESULTS (re-computed from data/sports.db)")
    print("=" * 78)
    print("\n[Data provenance - metadata table]")
    print(r["metadata"].to_string(index=False))
    print("\n[Row counts recomputed from DB]")
    for k, v in r["data_counts"].items():
        print(f"  {k:<16} {v}")
    print(f"\n[Split] train seasons {TRAIN_SEASONS}: {r['n_train']} games "
          f"({r['n_train_ties']} ties excluded from win/loss metrics); "
          f"test season {TEST_SEASON}: {r['n_test']} games ({r['n_test_ties']} ties)")
    print(f"  Features: rolling {ROLLING_WINDOW} prior games, form over {FORM_WINDOW} prior games")

    m = r["metrics"].copy()
    print("\n[Test-season metrics, 2024 (REG + POST)]")
    cols = ["model", "accuracy", "log_loss", "brier", "ece"]
    print(m[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n[Against the closing spread, 2024 (pushes excluded)]")
    ats = m.dropna(subset=["ats_n"])[["model", "ats_n", "ats_wins", "ats_losses", "ats_pushes",
                                      "ats_hit_rate", "ats_pvalue_vs_50"]].copy()
    for c in ("ats_n", "ats_wins", "ats_losses", "ats_pushes"):
        ats[c] = ats[c].astype(int)
    print(ats.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n[Margin MAE vs actual home point differential, 2024]")
    print(m.dropna(subset=["margin_mae"])[["model", "margin_mae"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    for name, cal in r["calibration"].items():
        print(f"\n[Calibration: {name}] (frac sums to {cal['frac'].sum():.4f})")
        print(cal.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    d = r["drift"]
    print(f"\n[Drift check: train vs test feature means] flagged |SMD|>0.25: "
          f"{int(d['drift_flag'].sum())} of {len(d)} (informational flag, not a failure)")
    print(d.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    s = r["sim_metrics"]
    print(f"\n[Monte Carlo: {TEST_SEASON} regular season after week {SIM_CUTOFF_WEEK}] "
          f"model={s['sim_model']} sims={s['sim_n']} seed={s['sim_seed']} "
          f"remaining_games={s['sim_remaining_games']}")
    print(f"  MAE expected wins vs actual final wins : {s['sim_mae_exp_wins']:.3f}")
    print(f"  MAE naive win-pace extrapolation        : {s['pace_mae_wins']:.3f}")
    print(f"  MAE coin-flip remaining games           : {s['coinflip_mae_wins']:.3f}")
    print(f"  Actual final wins inside sim P10-P90    : {s['sim_actual_within_p10_p90']:.4f}")
    print(r["simulation"].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if "mlflow_run_id" in r:
        print(f"\n[MLflow] parent run id: {r['mlflow_run_id']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = parser.parse_args()
    if not DB_PATH.exists():
        sys.exit(f"{DB_PATH} missing. Run `python -m sfl.data` to download real nflverse data.")

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    warnings.filterwarnings("ignore")
    r = run_pipeline(log_mlflow=not args.no_mlflow)

    buf = io.StringIO()
    with redirect_stdout(buf):
        report(r)
    text = buf.getvalue()
    print(text)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "verify_output.txt").write_text(text)
    (REPORTS_DIR / "metrics.json").write_text(metrics_to_json(r))


if __name__ == "__main__":
    main()
