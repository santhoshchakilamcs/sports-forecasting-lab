"""Model definitions. Hyperparameters are fixed up front; the 2024 test season is never used for tuning."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from sfl.config import SEED

LOGREG_PARAMS = {"C": 1.0, "max_iter": 2000}
XGB_PARAMS = {
    "n_estimators": 300, "max_depth": 2, "learning_rate": 0.03, "subsample": 0.8,
    "colsample_bytree": 0.8, "min_child_weight": 5, "reg_lambda": 1.0,
    "random_state": SEED, "n_jobs": 1, "eval_metric": "logloss",
}
RIDGE_PARAMS = {"alpha": 1.0}


def make_logreg() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(**LOGREG_PARAMS))])


def make_xgb() -> XGBClassifier:
    return XGBClassifier(**XGB_PARAMS)


@dataclass
class SpreadModel:
    """Ridge regression on home point differential.

    Win probability is derived as ``P(margin > 0) = Phi(pred / sigma)`` where ``sigma`` is the
    standard deviation of training residuals.
    """
    pipe: Pipeline = field(default_factory=lambda: Pipeline(
        [("scale", StandardScaler()), ("reg", Ridge(**RIDGE_PARAMS))]))
    sigma: float = float("nan")

    def fit(self, X: pd.DataFrame, margin: pd.Series) -> "SpreadModel":
        self.pipe.fit(X, margin)
        self.sigma = float(np.std(margin - self.pipe.predict(X), ddof=1))
        return self

    def predict_margin(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return norm.cdf(self.predict_margin(X) / self.sigma)


def market_sigma(result: pd.Series, spread_line: pd.Series) -> float:
    """Std of (result - closing line) on training data; maps a line to a win probability."""
    return float(np.std(result - spread_line, ddof=1))


def prob_to_margin(p: np.ndarray, sigma: float) -> np.ndarray:
    """Invert the normal margin model so classifier probabilities can be compared to a line."""
    return sigma * norm.ppf(np.clip(p, 1e-6, 1 - 1e-6))
