"""Deployment checks using real, verified predictions, with no downloaded DB required."""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "reports" / "metrics.json"


def reject_nonstandard_constant(value):
    raise ValueError(f"Non-standard JSON number: {value}")


@pytest.fixture(scope="module")
def snapshot():
    return json.loads(SNAPSHOT.read_text(), parse_constant=reject_nonstandard_constant)


def test_dashboard_runs_without_database_and_switches_weeks(tmp_path, snapshot):
    """Emulate the deployed app: just the entrypoint and its checked-in JSON snapshot."""
    shutil.copy(ROOT / "streamlit_app.py", tmp_path / "streamlit_app.py")
    (tmp_path / "reports").mkdir()
    shutil.copy(SNAPSHOT, tmp_path / "reports" / "metrics.json")
    assert not (tmp_path / "data").exists()

    app = AppTest.from_file(str(tmp_path / "streamlit_app.py"), default_timeout=60).run()
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 4
    assert len(app.dataframe) == 5
    assert len(app.get("vega_lite_chart")) == 1
    pred = pd.DataFrame(snapshot["pred_test"])
    assert len(app.dataframe[0].value) == (pred["week"] == pred["week"].min()).sum()

    for week in sorted(pred["week"].unique()):
        app.selectbox[0].set_value(int(week)).run()
        assert not app.exception
        expected = pred.loc[pred["week"] == week]
        assert app.dataframe[0].value["matchup"].tolist() == (
            expected["away_team"] + " @ " + expected["home_team"]).tolist()
        assert len(app.get("vega_lite_chart")) == 1


def test_missing_snapshot_stops_with_clear_error(tmp_path):
    shutil.copy(ROOT / "streamlit_app.py", tmp_path / "streamlit_app.py")
    app = AppTest.from_file(str(tmp_path / "streamlit_app.py"), default_timeout=60).run()
    assert not app.exception
    assert "No replacement data is generated" in app.error[0].value
    assert len(app.dataframe) == 0
