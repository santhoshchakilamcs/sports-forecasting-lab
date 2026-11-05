# Sports Forecasting Lab

NFL win-probability and point-spread models built on public play-by-play data, evaluated
out-of-sample on the 2024 season against the betting market's closing line.

> **Scope and disclaimer.** This is a personal research project on public data. It is not job
> experience, not affiliated with Susquehanna or any employer, and involves no real betting,
> trading, or money. Against-the-spread numbers are a historical model-evaluation benchmark only,
> not betting advice. Nothing here scrapes live odds.

## Data

| Item | Value |
|---|---|
| Source | `nfl_data_py` 0.3.3 (nflverse). The fallback, public nflverse CSVs on GitHub, was **not** needed. |
| Seasons | 2021, 2022, 2023, 2024 (regular season + playoffs) |
| `schedules` table | 1,139 games (scores, rest days, nflverse `spread_line`) |
| `pbp` table | 198,513 plays (22 selected columns from the ~390-column feed) |
| `teams` table | 32 teams (conference / division) |
| Storage | SQLite, `data/sports.db`. Provenance is in the `metadata` table. |

There is no synthetic data. All games, scores, and lines come from nflverse. In nflverse,
`spread_line` is "the spread line for the game" and uses the same sign as `result` (positive
means the home team was favored). It is used here as the closing-spread benchmark.

## Method

**Features (no lookahead).** Each team gets one row of stats per game. Those stats are shifted by
one game, then averaged with a rolling window, so a game's features only include games played
**before** it:

- Last 8 prior games: offensive and defensive EPA per play (expected points added), success
  rate, points per drive, turnover rate, and yards per play.
- Last 4 prior games (recent form): win rate and average point margin.
- Pre-game context: rest-day difference (capped at 14 days) and a home-field flag (0 at neutral
  sites).

Each model input is the home team's value minus the away team's value. Windows carry over across
seasons. The 16 games in 2021 week 1 are dropped because those teams have no earlier games in the
dataset. Two tests check for leakage:

1. A small hand-built table confirms the rolling windows only see earlier games.
2. On the real data, every game on or after 2023-11-01 is corrupted (EPA set to 99, scores
   changed). All features for earlier games must stay exactly the same.

**Models.** Hyperparameters were set once in `sfl/models.py` and never tuned on 2024 data.

- (a) Logistic regression with standardized features.
- (b) XGBoost classifier (depth 2, 300 trees, learning rate 0.03).
- (c) Spread model: ridge regression on home point differential. Its win probability is
  Φ(predicted margin / σ), where σ is the spread of the training errors.

Two benchmarks are scored the same way:

- **Mean baseline:** always predicts the 2021–2023 home win rate.
- **Market spread:** Φ(closing line / σ), with σ fitted on 2021–2023.

**Split.** Train on 2021–2023 (838 games, 3 ties left out of win/loss metrics). Test on 2024 only
(285 games, 0 ties).

**Against the spread (ATS).** For each game, the model takes the home side if its predicted margin
is above the closing line, otherwise the away side. The classifiers' probabilities are converted to
margins with the market σ. Pushes (games landing exactly on the line) are excluded. The p-value is
a two-sided binomial test against 50%.

**Simulation.** The 2024 regular season is replayed as of the end of week 9:

- Results through week 9 are fixed.
- The 134 remaining games are simulated 10,000 times with seed 42, using logistic regression
  probabilities.
- Every team's features are frozen at the week-9 cutoff, so no later information is used.
- Division titles go to the team with the most wins. Ties are broken at random; NFL tiebreakers
  are not modeled.
- Expected wins are compared to actual 2024 final wins.

**Tracking and drift.**

- MLflow (`mlflow.db`) stores one parent run plus one child run per model and benchmark, with
  parameters, metrics, calibration tables, and the fitted models.
- The drift check compares train and test feature means using the standardized mean difference
  (SMD). Features with |SMD| > 0.25 are flagged for review; a flag does not fail the run.

## Verified results (2024 out-of-sample)

All numbers below were produced by `python verify.py` (full output in `reports/verify_output.txt`).

| Model | Accuracy | Log loss | Brier | ECE | ATS record (W-L) | ATS hit rate | ATS p-value vs 50% | Margin MAE |
|---|---|---|---|---|---|---|---|---|
| Mean baseline | 0.5474 | 0.6887 | 0.2478 | – | – | – | – | – |
| Logistic regression | 0.6456 | 0.6315 | 0.2206 | 0.0413 | 141-140 | 0.5018 | 1.0000 | 10.406 |
| XGBoost | 0.6316 | 0.6466 | 0.2269 | 0.0568 | 132-149 | 0.4698 | 0.3399 | 10.788 |
| Spread model (ridge) | 0.6351 | 0.6288 | 0.2192 | 0.0564 | 143-138 | 0.5089 | 0.8115 | 10.272 |
| Market closing spread (benchmark) | 0.7053 | 0.5931 | 0.2028 | 0.0788 | – | – | – | 9.704 |

ATS: 281 graded games and 4 pushes. ECE (expected calibration error) is the game-weighted average
gap between predicted probability and actual win rate across 10 bins. Full 10-bin calibration
tables are in the verify output and the dashboard.

**Reading the results:**

- All three models beat the mean baseline on accuracy, log loss, and Brier score.
- The spread model has the best log loss and Brier score. Logistic regression has the best
  accuracy and calibration.
- XGBoost did not beat the linear models on this small sample (838 training games).
- **None of the models beats the closing line.** The market benchmark is better on every metric
  except ECE.
- No model's ATS hit rate is statistically different from 50%: all p-values are ≥ 0.34, and
  XGBoost is below 50%. There is no evidence of an edge over the market.

**Drift check:** 0 of 14 features flagged. The largest shift is rest-day difference (SMD −0.105).

**Monte Carlo (2024, after week 9, 10,000 sims, seed 42):**

| Method | MAE vs actual final wins |
|---|---|
| Simulated expected wins (logistic regression) | 1.433 |
| Naive win-pace extrapolation | 1.316 |
| Coin-flip remaining games | 1.844 |

Actual final wins fell inside the simulated 10th–90th percentile range for 71.88% of teams
(the target for that range is 80%). The simulation beats coin flips but **not** simple win-pace
extrapolation.

## Limitations

- One test season (285 games) leaves wide uncertainty on every metric.
- Features exclude injuries, quarterback changes, weather, and roster moves.
- The market σ is a single global value.
- Division tiebreakers in the simulation are random rather than NFL rules.

## Project layout

```
sfl/config.py       seasons, split, seeds, windows
sfl/data.py         download (nfl_data_py -> GitHub CSV fallback) -> SQLite
sfl/features.py     per-team game stats, prior-games-only rolling features
sfl/models.py       logistic regression, XGBoost, ridge spread model
sfl/evaluation.py   accuracy / log loss / Brier, calibration bins, ATS, drift
sfl/simulation.py   cutoff-frozen features, Monte Carlo season simulation
sfl/pipeline.py     end-to-end orchestration + MLflow logging
verify.py           re-runs everything from data/sports.db, prints verified metrics
streamlit_app.py    dashboard: weekly predictions, model vs line, calibration, simulation
tests/              pytest suite (7 tests)
```

## Reproduce

Requires Python 3.10–3.12. `nfl_data_py` needs NumPy < 2, so `pyarrow` is pinned to 18.1.0.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m sfl.data              # download real nflverse data -> data/sports.db
python verify.py                # full pipeline + MLflow logging + verified metrics
pytest -q                       # 7 tests
streamlit run streamlit_app.py  # dashboard
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
