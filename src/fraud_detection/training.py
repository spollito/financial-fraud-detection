"""Train and evaluate a regularized, class-weighted logistic baseline."""

import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .evaluation import require_both_classes, score_partition, select_threshold, temporal_split
from .features import CATEGORICAL, NUMERICAL, RAW_FEATURES, make_features
from .ingestion import TARGET, TIME


def train_baseline(
    frame: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
    evaluate_test: bool,
    max_fpr: float,
    review_cost: float,
    missed_fraud_cost: float,
) -> tuple[Pipeline, dict]:
    start = time.perf_counter()
    train, validation, test = temporal_split(frame, train_fraction, validation_fraction)
    require_both_classes(train, "Train")
    require_both_classes(validation, "Validation")
    splits = {"train": train, "validation": validation, "test": test}
    report = {
        "partitions": {
            name: {
                "rows": len(partition),
                "min_transaction_dt": int(partition[TIME].min()),
                "max_transaction_dt": int(partition[TIME].max()),
            }
            for name, partition in splits.items()
        },
        "threshold_selection": "maximize validation F1, preferring the higher threshold on ties",
        "primary_metric": "sklearn.metrics.average_precision_score",
        "model": {
            "family": "logistic_regression",
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000,
        },
    }
    model = Pipeline(
        [
            ("features", FunctionTransformer(make_features, validate=False)),
            (
                "encode",
                ColumnTransformer(
                    [
                        ("numeric", StandardScaler(), list(NUMERICAL)),
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
        ]
    )
    raw_features = list(RAW_FEATURES)
    fit_start = time.perf_counter()
    model.fit(train[raw_features], train[TARGET])
    report["fit_seconds"] = time.perf_counter() - fit_start
    validation_labels = validation[TARGET].to_numpy()
    validation_scores = model.predict_proba(validation[raw_features])[:, 1]
    threshold = select_threshold(validation_labels, validation_scores)
    report["threshold"] = threshold
    metric_settings = {
        "max_fpr": max_fpr,
        "review_cost": review_cost,
        "missed_fraud_cost": missed_fraud_cost,
    }
    report["validation"] = score_partition(
        validation_labels, validation_scores, threshold, **metric_settings
    )
    if evaluate_test:
        require_both_classes(test, "Test")
        report["test"] = score_partition(
            test[TARGET].to_numpy(),
            model.predict_proba(test[raw_features])[:, 1],
            threshold,
            **metric_settings,
        )
    report["total_seconds"] = time.perf_counter() - start
    return model, report
