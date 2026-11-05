"""Project-wide constants and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "sports.db"
REPORTS_DIR = ROOT / "reports"
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "sports-forecasting-lab"

SEASONS = [2021, 2022, 2023, 2024]
TRAIN_SEASONS = [2021, 2022, 2023]
TEST_SEASON = 2024

SEED = 42
N_SIMS = 10_000
SIM_CUTOFF_WEEK = 9  # simulate 2024 regular-season weeks 10-18 from information known after week 9

ROLLING_WINDOW = 8  # prior games used for efficiency stats
FORM_WINDOW = 4     # prior games used for recent form

# Play-by-play columns persisted to SQLite (the full feed has ~390 columns).
PBP_COLUMNS = [
    "play_id", "game_id", "season", "season_type", "week", "game_date",
    "home_team", "away_team", "posteam", "defteam", "play_type",
    "down", "yards_gained", "epa", "success", "interception", "fumble_lost",
    "drive", "fixed_drive", "fixed_drive_result", "qb_kneel", "qb_spike",
]

SCHEDULE_COLUMNS = [
    "game_id", "season", "game_type", "week", "gameday", "weekday", "gametime",
    "away_team", "away_score", "home_team", "home_score", "location", "result",
    "total", "overtime", "away_rest", "home_rest", "spread_line", "total_line",
    "div_game", "roof", "surface",
]
