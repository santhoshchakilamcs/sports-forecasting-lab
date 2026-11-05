"""Research-environment checks linking the hosted snapshot to the original pipeline."""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "reports" / "metrics.json"


def test_snapshot_predictions_reproduce_reported_metrics():
    """Published scores and calibration must agree with the actual published predictions."""
    from sfl.pipeline import evaluate

    snapshot = json.loads(SNAPSHOT.read_text())
    pred = pd.DataFrame(snapshot["pred_test"])
    assert len(pred) == snapshot["n_test"]
    assert pred["game_id"].is_unique
    assert set(pred["season"]) == {snapshot["test_season"]}
    metrics, calibration = evaluate(pred)
    pd.testing.assert_frame_equal(metrics, pd.DataFrame(snapshot["metrics"]),
                                  check_exact=False, atol=1e-12, rtol=1e-12)
    for name, table in calibration.items():
        pd.testing.assert_frame_equal(table, pd.DataFrame(snapshot["calibration"][name]),
                                      check_exact=False, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(not (ROOT / "data" / "sports.db").exists(), reason="Real DB not downloaded")
def test_committed_snapshot_matches_fresh_pipeline():
    from sfl.pipeline import metrics_to_json, run_pipeline

    fresh = metrics_to_json(run_pipeline(log_mlflow=False))
    assert json.loads(fresh) == json.loads(SNAPSHOT.read_text())
