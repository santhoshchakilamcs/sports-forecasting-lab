"""Leakage-free game features.

Pipeline:
1. ``team_game_stats``  - one row per (game, team) with that game's box-score efficiency.
2. ``rolling_team_features`` - for each team, rolling means over *prior* games only
   (``shift(1)`` before ``rolling``), so a game's own result never enters its features.
3. ``build_game_features`` - one row per game: home-minus-away feature differences,
   rest, and home-field indicator, plus targets (result, home_win) and the market line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sfl.config import FORM_WINDOW, ROLLING_WINDOW

EFFICIENCY_STATS = [
    "off_epa", "def_epa", "off_sr", "def_sr", "off_ppd", "def_ppd",
    "off_to_rate", "def_to_rate", "off_ypp", "def_ypp",
]
FORM_STATS = ["win", "margin"]
TEAM_FEATURES = [f"roll_{s}" for s in EFFICIENCY_STATS] + [f"form_{s}" for s in FORM_STATS]
FEATURES = [f"diff_{f}" for f in TEAM_FEATURES] + ["rest_diff", "home_field"]


def team_game_stats(pbp: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Per-(game, team) offensive and defensive efficiency for completed games."""
    plays = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()].copy()
    plays["turnover"] = ((plays["interception"] == 1) | (plays["fumble_lost"] == 1)).astype(float)

    off = plays.groupby(["game_id", "posteam"]).agg(
        off_epa=("epa", "mean"), off_sr=("success", "mean"), off_ypp=("yards_gained", "mean"),
        off_to_rate=("turnover", "mean"),
    ).reset_index().rename(columns={"posteam": "team"})
    drives = (pbp.dropna(subset=["posteam", "fixed_drive"])
              .groupby(["game_id", "posteam"])["fixed_drive"].nunique()
              .rename("drives").reset_index().rename(columns={"posteam": "team"}))
    off = off.merge(drives, on=["game_id", "team"], how="left")

    games = sched.dropna(subset=["result"])
    sides = []
    for side, opp in (("home", "away"), ("away", "home")):
        g = games[["game_id", "season", "week", "game_type", "gameday",
                   f"{side}_team", f"{opp}_team", f"{side}_score", f"{opp}_score"]].copy()
        g.columns = ["game_id", "season", "week", "game_type", "gameday",
                     "team", "opp", "points_for", "points_against"]
        g["is_home"] = int(side == "home")
        sides.append(g)
    tg = pd.concat(sides, ignore_index=True)

    tg = tg.merge(off, on=["game_id", "team"], how="left")
    opp_off = off.rename(columns={"team": "opp", "off_epa": "def_epa", "off_sr": "def_sr",
                                  "off_ypp": "def_ypp", "off_to_rate": "def_to_rate",
                                  "drives": "opp_drives"})
    tg = tg.merge(opp_off, on=["game_id", "opp"], how="left")

    tg["off_ppd"] = tg["points_for"] / tg["drives"]
    tg["def_ppd"] = tg["points_against"] / tg["opp_drives"]
    tg["margin"] = tg["points_for"] - tg["points_against"]
    tg["win"] = np.sign(tg["margin"]).map({1: 1.0, 0: 0.5, -1: 0.0})
    tg["gameday"] = pd.to_datetime(tg["gameday"])
    return tg.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)


def rolling_team_features(tg: pd.DataFrame, include_current: bool = False) -> pd.DataFrame:
    """Rolling means per team, ordered by date.

    ``include_current=False`` (default) shifts by one game so each row only sees games
    played strictly before it. ``include_current=True`` gives the team state *after*
    each game, used to snapshot information at a simulation cutoff.
    """
    tg = tg.sort_values(["team", "gameday", "game_id"]).copy()
    grp = tg.groupby("team", sort=False)
    shift = 0 if include_current else 1
    for s in EFFICIENCY_STATS:
        tg[f"roll_{s}"] = grp[s].transform(
            lambda x: x.shift(shift).rolling(ROLLING_WINDOW, min_periods=1).mean())
    for s in FORM_STATS:
        tg[f"form_{s}"] = grp[s].transform(
            lambda x: x.shift(shift).rolling(FORM_WINDOW, min_periods=1).mean())
    tg["prior_games"] = grp.cumcount() + (1 if include_current else 0)
    return tg


def assemble_matchups(sched: pd.DataFrame, home_feats: pd.DataFrame,
                      away_feats: pd.DataFrame) -> pd.DataFrame:
    """Combine per-team feature rows (indexed like ``sched``) into model features."""
    out = sched[["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team",
                 "home_score", "away_score", "result", "spread_line", "location",
                 "home_rest", "away_rest"]].copy()
    for f in TEAM_FEATURES:
        out[f"diff_{f}"] = home_feats[f].to_numpy() - away_feats[f].to_numpy()
    out["rest_diff"] = (out["home_rest"].clip(upper=14) - out["away_rest"].clip(upper=14))
    out["home_field"] = (out["location"] != "Neutral").astype(int)
    out["home_win"] = np.where(out["result"] > 0, 1.0, np.where(out["result"] < 0, 0.0, np.nan))
    return out


def build_game_features(pbp: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """One row per completed game with prior-games-only features.

    Games where either team has no prior game in the dataset (2021 week 1) are dropped.
    """
    tg = rolling_team_features(team_game_stats(pbp, sched))
    games = sched.dropna(subset=["result"]).reset_index(drop=True)
    cols = ["game_id", "team", "prior_games"] + TEAM_FEATURES
    home = games[["game_id", "home_team"]].merge(
        tg[cols], left_on=["game_id", "home_team"], right_on=["game_id", "team"], how="left")
    away = games[["game_id", "away_team"]].merge(
        tg[cols], left_on=["game_id", "away_team"], right_on=["game_id", "team"], how="left")
    out = assemble_matchups(games, home, away)
    out["min_prior_games"] = np.minimum(home["prior_games"], away["prior_games"])
    out = out[out["min_prior_games"] > 0].dropna(subset=FEATURES)
    out["gameday"] = pd.to_datetime(out["gameday"])
    return out.sort_values(["gameday", "game_id"]).reset_index(drop=True)
