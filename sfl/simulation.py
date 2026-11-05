"""Monte Carlo simulation of the remaining regular season from a week cutoff.

Features for every remaining game are frozen at the cutoff (each team's rolling state after
its last game played by then), so the simulation uses no information from after the cutoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sfl.config import N_SIMS, SEED
from sfl.features import TEAM_FEATURES, assemble_matchups, rolling_team_features, team_game_stats


def split_at_cutoff(sched: pd.DataFrame, season: int, cutoff_week: int):
    reg = sched[(sched["season"] == season) & (sched["game_type"] == "REG")]
    played = reg[reg["week"] <= cutoff_week]
    remaining = reg[reg["week"] > cutoff_week].reset_index(drop=True)
    return played, remaining


def remaining_game_features(pbp: pd.DataFrame, sched: pd.DataFrame, season: int,
                            cutoff_week: int) -> pd.DataFrame:
    played, remaining = split_at_cutoff(sched, season, cutoff_week)
    first_remaining = pd.to_datetime(remaining["gameday"]).min()
    known = sched[pd.to_datetime(sched["gameday"]) < first_remaining]
    known_pbp = pbp[pbp["game_id"].isin(known["game_id"])]
    state = (rolling_team_features(team_game_stats(known_pbp, known), include_current=True)
             .groupby("team").tail(1).set_index("team"))
    home = state.loc[remaining["home_team"], TEAM_FEATURES].reset_index(drop=True)
    away = state.loc[remaining["away_team"], TEAM_FEATURES].reset_index(drop=True)
    return assemble_matchups(remaining, home, away)


def current_wins(played: pd.DataFrame, teams: list[str]) -> pd.Series:
    w = pd.Series(0.0, index=teams)
    for _, g in played.iterrows():
        if g["result"] > 0:
            w[g["home_team"]] += 1
        elif g["result"] < 0:
            w[g["away_team"]] += 1
        else:
            w[g["home_team"]] += 0.5
            w[g["away_team"]] += 0.5
    return w


def simulate_season(start_wins: pd.Series, home_teams, away_teams, p_home,
                    divisions: pd.Series | None = None, n_sims: int = N_SIMS,
                    seed: int = SEED) -> tuple[pd.DataFrame, np.ndarray]:
    """Return per-team summary and the full (n_sims, n_teams) wins matrix."""
    teams = list(start_wins.index)
    col = {t: i for i, t in enumerate(teams)}
    n_games = len(p_home)
    H = np.zeros((n_games, len(teams)))
    A = np.zeros((n_games, len(teams)))
    H[np.arange(n_games), [col[t] for t in home_teams]] = 1
    A[np.arange(n_games), [col[t] for t in away_teams]] = 1

    rng = np.random.default_rng(seed)
    home_win = (rng.random((n_sims, n_games)) < np.asarray(p_home)).astype(float)
    wins = start_wins.to_numpy() + home_win @ H + (1 - home_win) @ A

    summary = pd.DataFrame({
        "team": teams,
        "wins_at_cutoff": start_wins.to_numpy(),
        "exp_wins": wins.mean(axis=0),
        "p10_wins": np.percentile(wins, 10, axis=0),
        "p90_wins": np.percentile(wins, 90, axis=0),
        "p_10plus_wins": (wins >= 10).mean(axis=0),
    })
    if divisions is not None:
        # Division title by most wins; ties broken uniformly at random (NFL tiebreakers not modeled).
        noisy = wins + rng.random(wins.shape) * 1e-3
        div_win = np.zeros_like(wins)
        for _, members in divisions.reindex(teams).groupby(divisions.reindex(teams)):
            idx = [col[t] for t in members.index]
            best = np.argmax(noisy[:, idx], axis=1)
            div_win[np.arange(n_sims), np.array(idx)[best]] = 1
        summary["p_division"] = div_win.mean(axis=0)
    return summary, wins
