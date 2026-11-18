import numpy as np
import pandas as pd
import pytest

from fraud_detection.evaluation import score_partition, temporal_split
from fraud_detection.features import CATEGORICAL, NUMERICAL, make_features
from fraud_detection.ingestion import AMOUNT, ID, TARGET, TIME, load_dataset


def test_temporal_split_never_divides_transaction_time_ties(fixture_data):
    frame, _ = load_dataset(fixture_data)
    train, validation, test = temporal_split(frame, 0.61, 0.20)
    assert train[TIME].max() < validation[TIME].min()
    assert validation[TIME].max() < test[TIME].min()
    assert sum(map(len, (train, validation, test))) == len(frame)
    assert pd.concat([train, validation, test])[ID].is_unique


def test_temporal_split_fails_on_unsplittable_data():
    frame = pd.DataFrame({ID: [1, 2, 3], TIME: [5, 5, 5]})
    with pytest.raises(ValueError, match="empty partition"):
        temporal_split(frame)


def test_features_exclude_identifiers_time_and_label(fixture_data):
    frame, _ = load_dataset(fixture_data)
    features = make_features(frame)
    assert set(features.columns) == set(CATEGORICAL) | set(NUMERICAL)
    assert not {ID, TIME, TARGET}.intersection(features.columns)
    np.testing.assert_allclose(features["log_amount"], np.log1p(frame[AMOUNT]))


def test_metrics_count_ties_at_threshold_and_simulate_cost():
    metrics = score_partition(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.5, 0.5, 0.3]),
        0.5,
        max_fpr=0.0,
        review_cost=5,
        missed_fraud_cost=250,
    )
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["simulated_cost"]["total"] == 255
    assert metrics["roc_recall_at_max_fpr"]["recall"] == 0
