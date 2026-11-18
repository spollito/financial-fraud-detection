import json
import subprocess
import sys

import joblib
import numpy as np
import pytest

from fraud_detection.features import CATEGORICAL, RAW_FEATURES
from fraud_detection.ingestion import TARGET, TIME, load_dataset
from fraud_detection.training import train_baseline


def test_preprocessor_and_threshold_are_independent_of_test_rows(fixture_data):
    frame, _ = load_dataset(fixture_data)
    frame.loc[frame[TIME] >= 36 * 7200, "ProductCD"] = "validation_or_test_only"
    settings = {
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "max_fpr": 0.01,
        "review_cost": 5,
        "missed_fraud_cost": 250,
    }
    model, original = train_baseline(frame, **settings, evaluate_test=True)
    test_mask = frame[TIME] >= 48 * 7200
    assert test_mask.any()
    changed = frame.copy()
    changed.loc[test_mask, "ProductCD"] = "unseen_test_category"
    changed.loc[test_mask, TARGET] = 1 - changed.loc[test_mask, TARGET]
    _, new = train_baseline(changed, **settings, evaluate_test=True)
    categories = model.named_steps["encode"].named_transformers_["categorical"].categories_
    assert "validation_or_test_only" not in categories[CATEGORICAL.index("ProductCD")]
    assert original["threshold"] == new["threshold"]
    assert original["validation"] == new["validation"]


@pytest.mark.parametrize("include_test", [False, True])
def test_cli_reports_and_model_serialization(tmp_path, fixture_data, include_test):
    results = tmp_path / "run"
    command = [sys.executable, "-m", "fraud_detection.cli"]
    data_args = ["--data", str(fixture_data), "--dataset", "synthetic"]
    subprocess.run(
        [*command, "inspect", *data_args, "--output", str(tmp_path / "eda")],
        check=True,
    )
    args = [*command, "train", *data_args, "--output", str(results)]
    if include_test:
        args.append("--evaluate-test")
    subprocess.run(args, check=True)
    eda = json.loads((tmp_path / "eda" / "eda.json").read_text(encoding="utf-8"))
    report = json.loads((results / "baseline.json").read_text(encoding="utf-8"))
    assert eda["scope"] == "oldest partition only"
    assert eda["class_distribution"]["rows"] == report["partitions"]["train"]["rows"]
    assert eda["synthetic_only"] and report["synthetic_only"]
    assert report["quality"]["rows"] == 120
    assert ("test" in report) == include_test
    assert 0 <= report["validation"]["average_precision"] <= 1
    model_path = results / "baseline.joblib"
    model = joblib.load(model_path)
    frame, _ = load_dataset(fixture_data)
    scores = model.predict_proba(frame[list(RAW_FEATURES)])
    assert scores.shape == (120, 2)
    np.testing.assert_allclose(scores.sum(axis=1), 1)
    assert model_path.stat().st_size == report["artifact_bytes"]
    duplicate = subprocess.run(args, capture_output=True, text=True)
    assert duplicate.returncode == 2
    assert "already exists" in duplicate.stderr
