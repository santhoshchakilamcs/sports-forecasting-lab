"""Download real NFL data (nflverse) and persist it to SQLite.

Primary source: the ``nfl_data_py`` package. Fallback: public nflverse CSVs on GitHub.
No synthetic data is ever generated; if every source fails the script exits with an error.

Usage:  python -m sfl.data
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone

import pandas as pd

from sfl.config import DATA_DIR, DB_PATH, PBP_COLUMNS, SCHEDULE_COLUMNS, SEASONS

PBP_FALLBACK_URLS = [
    "https://github.com/nflverse/nflfastR-data/raw/master/data/play_by_play_{season}.csv.gz",
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz",
]
SCHEDULE_FALLBACK_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
TEAMS_FALLBACK_URL = "https://github.com/nflverse/nflfastR-data/raw/master/teams_colors_logos.csv"


def _download_nfl_data_py(seasons: list[int]):
    import nfl_data_py as nfl

    sched = nfl.import_schedules(seasons)
    pbp = nfl.import_pbp_data(seasons, columns=PBP_COLUMNS, downcast=False, cache=False)
    teams = nfl.import_team_desc()
    return sched, pbp, teams, "nfl_data_py (nflverse)"


def _download_github_csv(seasons: list[int]):
    frames = []
    for season in seasons:
        last_err = None
        for url in PBP_FALLBACK_URLS:
            try:
                frames.append(pd.read_csv(url.format(season=season), usecols=PBP_COLUMNS,
                                          low_memory=False))
                break
            except Exception as err:  # noqa: BLE001
                last_err = err
        else:
            raise RuntimeError(f"All play-by-play sources failed for {season}: {last_err}")
    sched = pd.read_csv(SCHEDULE_FALLBACK_URL)
    sched = sched[sched["season"].isin(seasons)]
    teams = pd.read_csv(TEAMS_FALLBACK_URL)
    return sched, pd.concat(frames, ignore_index=True), teams, "nflverse GitHub CSVs"


def download(seasons: list[int] = SEASONS):
    try:
        return _download_nfl_data_py(seasons)
    except Exception as err:  # noqa: BLE001
        print(f"nfl_data_py failed ({err!r}); falling back to GitHub CSVs", file=sys.stderr)
    return _download_github_csv(seasons)


def write_db(sched: pd.DataFrame, pbp: pd.DataFrame, teams: pd.DataFrame, source: str,
             db_path=DB_PATH) -> pd.DataFrame:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sched = sched[SCHEDULE_COLUMNS].copy()
    pbp = pbp[PBP_COLUMNS].copy()
    teams = teams[["team_abbr", "team_name", "team_conf", "team_division"]].copy()
    teams = teams[teams["team_abbr"].isin(set(sched["home_team"]) | set(sched["away_team"]))]

    meta = pd.DataFrame({
        "table_name": ["schedules", "pbp", "teams"],
        "rows": [len(sched), len(pbp), len(teams)],
        "seasons": [",".join(map(str, sorted(sched["season"].unique())))] * 3,
        "source": [source] * 3,
        "downloaded_utc": [datetime.now(timezone.utc).isoformat(timespec="seconds")] * 3,
    })
    with sqlite3.connect(db_path) as con:
        sched.to_sql("schedules", con, if_exists="replace", index=False)
        pbp.to_sql("pbp", con, if_exists="replace", index=False, chunksize=20_000)
        teams.to_sql("teams", con, if_exists="replace", index=False)
        meta.to_sql("metadata", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_pbp_game ON pbp(game_id)")
    return meta


def read_table(name: str, db_path=DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found. Run `python -m sfl.data` first.")
    with sqlite3.connect(db_path) as con:
        return pd.read_sql(f"SELECT * FROM {name}", con)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sched, pbp, teams, source = download()
    meta = write_db(sched, pbp, teams, source)
    print(meta.to_string(index=False))


if __name__ == "__main__":
    main()
