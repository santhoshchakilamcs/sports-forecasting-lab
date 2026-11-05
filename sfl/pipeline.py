"""End-to-end pipeline: SQLite -> features -> models -> evaluation -> simulation (-> MLflow)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from sfl.config import (DB_PATH, MLFLOW_EXPERIMENT, MLFLOW_URI, N_SIMS, SEED, SIM_CUTOFF_WEEK,
                        TEST_SEASON, TRAIN_SEASONS)
from sfl.data import read_table
from sfl.evaluation import (against_the_spread, calibration_table, classification_metrics,
                            drift_report, expected_calibration_error)
from sfl.features import FEATURES, build_game_features, team_game_stats
from sfl.models import (LOGREG_PARAMS, RIDGE_PARAMS, XGB_PARAMS, SpreadModel, make_logreg,
                        make_xgb, market_sigma, prob_to_margin)
from sfl.simulation import current_wins, remaining_game_features, simulate_season, split_at_cutoff

MODEL_NAMES = ["mean_baseline", "logreg", "xgboost", "spread_ridge", "market_spread"]


def load_data(db_path=DB_PATH):
    return (read_table("pbp", db_path), read_table("schedules", db_path),
            read_table("teams", db_path), read_table("metadata", db_path))


def split(games: pd.DataFrame):
    train = games[games["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    test = games[games["season"] == TEST_SEASON].reset_index(drop=True)
    return train, test


def fit_models(train: pd.DataFrame) -> dict:
    tr = train.dropna(subset=["home_win"])
    X, y = tr[FEATURES], tr["home_win"].astype(int)
    return {
        "mean_rate": float(y.mean()),
        "logreg": make_logreg().fit(X, y),
        "xgboost": make_xgb().fit(X, y),
        "spread_ridge": SpreadModel().fit(train[FEATURES], train["result"]),
        "market_sigma": market_sigma(train["result"], train["spread_line"]),
    }


def predict(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Home win probability and implied home margin for every model and benchmark."""
    from scipy.stats import norm

    X, sig = df[FEATURES], models["market_sigma"]
    out = df.copy()
    out["p_mean_baseline"] = models["mean_rate"]
    out["p_logreg"] = models["logreg"].predict_proba(X)[:, 1]
    out["p_xgboost"] = models["xgboost"].predict_proba(X)[:, 1]
    out["p_spread_ridge"] = models["spread_ridge"].predict_proba(X)
    out["p_market_spread"] = norm.cdf(df["spread_line"] / sig)
    out["margin_spread_ridge"] = models["spread_ridge"].predict_margin(X)
    out["margin_logreg"] = prob_to_margin(out["p_logreg"].to_numpy(), sig)
    out["margin_xgboost"] = prob_to_margin(out["p_xgboost"].to_numpy(), sig)
    return out


def evaluate(pred: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    decided = pred.dropna(subset=["home_win"])
    rows, cals = [], {}
    for name in MODEL_NAMES:
        p = decided[f"p_{name}"].to_numpy()
        row = {"model": name, **classification_metrics(decided["home_win"], p)}
        if name != "mean_baseline":
            cal = calibration_table(decided["home_win"], p)
            cals[name] = cal
            row["ece"] = expected_calibration_error(cal)
        if name in ("logreg", "xgboost", "spread_ridge"):
            row.update(against_the_spread(pred["result"], pred["spread_line"],
                                          pred[f"margin_{name}"]))
            row["margin_mae"] = float(np.mean(np.abs(pred[f"margin_{name}"] - pred["result"])))
        if name == "market_spread":
            row["margin_mae"] = float(np.mean(np.abs(pred["spread_line"] - pred["result"])))
        rows.append(row)
    return pd.DataFrame(rows), cals


def run_simulation(models: dict, pbp, sched, teams, model_name: str = "logreg",
                   cutoff_week: int = SIM_CUTOFF_WEEK, n_sims: int = N_SIMS, seed: int = SEED):
    """Simulate TEST_SEASON weeks after ``cutoff_week`` and compare to actual final records."""
    played, _ = split_at_cutoff(sched, TEST_SEASON, cutoff_week)
    rem = remaining_game_features(pbp, sched, TEST_SEASON, cutoff_week)
    p = predict(models, rem)[f"p_{model_name}"].to_numpy()
    team_list = sorted(teams["team_abbr"])
    start = current_wins(played, team_list)
    divisions = teams.set_index("team_abbr")["team_division"]
    summary, _ = simulate_season(start, rem["home_team"], rem["away_team"], p, divisions,
                                 n_sims=n_sims, seed=seed)

    reg = sched[(sched["season"] == TEST_SEASON) & (sched["game_type"] == "REG")]
    actual = current_wins(reg, team_list)
    games_played = pd.Series(0, index=team_list)
    for t in pd.concat([played["home_team"], played["away_team"]]):
        games_played[t] += 1
    n_total = 17
    summary["actual_final_wins"] = summary["team"].map(actual)
    pace = start / games_played * n_total
    coin = start + 0.5 * (n_total - games_played)
    sim_metrics = {
        "sim_model": model_name, "sim_cutoff_week": cutoff_week, "sim_n": n_sims, "sim_seed": seed,
        "sim_remaining_games": len(rem),
        "sim_mae_exp_wins": float(np.mean(np.abs(summary["exp_wins"] - summary["actual_final_wins"]))),
        "pace_mae_wins": float(np.mean(np.abs(summary["team"].map(pace) - summary["actual_final_wins"]))),
        "coinflip_mae_wins": float(np.mean(np.abs(summary["team"].map(coin) - summary["actual_final_wins"]))),
        "sim_actual_within_p10_p90": float(np.mean(
            (summary["actual_final_wins"] >= summary["p10_wins"])
            & (summary["actual_final_wins"] <= summary["p90_wins"]))),
    }
    return summary.sort_values("exp_wins", ascending=False).reset_index(drop=True), sim_metrics


def _log_mlflow(results: dict, models: dict) -> None:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    common = {"train_seasons": ",".join(map(str, TRAIN_SEASONS)), "test_season": TEST_SEASON,
              "n_features": len(FEATURES), "features": ",".join(FEATURES),
              "n_train_games": results["n_train"], "n_test_games": results["n_test"]}
    metrics = results["metrics"].set_index("model")

    def _metrics(name):
        return {k: float(v) for k, v in metrics.loc[name].items()
                if isinstance(v, (int, float, np.floating, np.integer)) and pd.notna(v)}

    with mlflow.start_run(run_name="pipeline") as parent, tempfile.TemporaryDirectory() as tmp:
        mlflow.log_params({**common, "sim_cutoff_week": SIM_CUTOFF_WEEK, "n_sims": N_SIMS,
                           "seed": SEED})
        mlflow.log_metrics({k: v for k, v in results["sim_metrics"].items()
                            if isinstance(v, (int, float)) and k not in ("sim_n", "sim_seed",
                                                                         "sim_cutoff_week")})
        drift = results["drift"]
        mlflow.log_metric("n_drift_flags", int(drift["drift_flag"].sum()))
        for name, df in {"drift.csv": drift, "metrics.csv": results["metrics"],
                         "simulation.csv": results["simulation"]}.items():
            df.to_csv(Path(tmp) / name, index=False)
            mlflow.log_artifact(str(Path(tmp) / name))

        specs = {
            "logreg": (LOGREG_PARAMS, lambda: mlflow.sklearn.log_model(models["logreg"], artifact_path="model")),
            "xgboost": (XGB_PARAMS, lambda: mlflow.xgboost.log_model(models["xgboost"], artifact_path="model")),
            "spread_ridge": ({**RIDGE_PARAMS, "sigma": models["spread_ridge"].sigma},
                             lambda: mlflow.sklearn.log_model(models["spread_ridge"].pipe, artifact_path="model")),
            "mean_baseline": ({"mean_rate": models["mean_rate"]}, None),
            "market_spread": ({"market_sigma": models["market_sigma"]}, None),
        }
        for name, (params, log_model) in specs.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params({**common, **params, "model": name})
                mlflow.log_metrics(_metrics(name))
                if name in results["calibration"]:
                    path = Path(tmp) / f"calibration_{name}.csv"
                    results["calibration"][name].to_csv(path, index=False)
                    mlflow.log_artifact(str(path))
                if log_model:
                    log_model()
        results["mlflow_run_id"] = parent.info.run_id


def run_pipeline(db_path=DB_PATH, log_mlflow: bool = True, sim_model: str = "logreg") -> dict:
    pbp, sched, teams, meta = load_data(db_path)
    games = build_game_features(pbp, sched)
    train, test = split(games)
    models = fit_models(train)
    pred_test = predict(models, test)
    metrics, cals = evaluate(pred_test)
    drift = drift_report(train, test, FEATURES)
    sim_summary, sim_metrics = run_simulation(models, pbp, sched, teams, model_name=sim_model)

    results = {
        "metadata": meta,
        "data_counts": {"schedule_games": len(sched), "pbp_plays": len(pbp),
                        "team_games": len(team_game_stats(pbp, sched)),
                        "modeled_games": len(games)},
        "n_train": len(train), "n_test": len(test),
        "n_train_ties": int(train["home_win"].isna().sum()),
        "n_test_ties": int(test["home_win"].isna().sum()),
        "models": models, "games": games, "pred_test": pred_test,
        "metrics": metrics, "calibration": cals, "drift": drift,
        "simulation": sim_summary, "sim_metrics": sim_metrics,
    }
    if log_mlflow:
        _log_mlflow(results, models)
    return results


def metrics_to_json(results: dict) -> str:
    return json.dumps({
        "data_counts": results["data_counts"],
        "n_train": results["n_train"], "n_test": results["n_test"],
        "metrics": results["metrics"].to_dict(orient="records"),
        "calibration": {k: v.to_dict(orient="records") for k, v in results["calibration"].items()},
        "drift": results["drift"].to_dict(orient="records"),
        "sim_metrics": results["sim_metrics"],
        "simulation": results["simulation"].to_dict(orient="records"),
    }, indent=2, default=float)
