"""Chronological holdouts and probability metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .ingestion import ID, TARGET, TIME


def temporal_split(
    frame: pd.DataFrame, train_fraction: float = 0.6, validation_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("Train and validation fractions must be positive and leave room for test")
    ordered = frame.sort_values([TIME, ID], ignore_index=True)
    timestamps = ordered[TIME].to_numpy()
    if len(timestamps) < 3:
        raise ValueError("At least three transactions are needed for three temporal partitions")
    cuts = [
        int(np.searchsorted(timestamps, timestamps[int(len(ordered) * fraction)], side="left"))
        for fraction in (train_fraction, train_fraction + validation_fraction)
    ]
    if not 0 < cuts[0] < cuts[1] < len(ordered):
        raise ValueError("The timestamp ties leave an empty partition. Adjust the split fractions.")
    return ordered.iloc[: cuts[0]], ordered.iloc[cuts[0] : cuts[1]], ordered.iloc[cuts[1] :]


def require_both_classes(frame: pd.DataFrame, split: str) -> None:
    if frame[TARGET].nunique() != 2:
        raise ValueError(f"{split} must contain legitimate and fraudulent transactions")


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(thresholds),
        where=denominator > 0,
    )
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def score_partition(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    max_fpr: float,
    review_cost: float,
    missed_fraud_cost: float,
) -> dict:
    predicted = scores >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(labels, predicted, labels=[0, 1]).ravel())
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    bins = np.minimum((scores * 10).astype(int), 9)
    calibration = [
        {
            "lower": bucket / 10,
            "upper": (bucket + 1) / 10,
            "count": int(mask.sum()),
            "mean_probability": float(scores[mask].mean()),
            "fraud_rate": float(labels[mask].mean()),
        }
        for bucket in np.unique(bins)
        for mask in [bins == bucket]
    ]
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "false_positive_rate": fp / (tn + fp),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "calibration_bins": calibration,
        "roc_recall_at_max_fpr": {"max_fpr": max_fpr, "recall": float(tpr[fpr <= max_fpr].max())},
        "simulated_cost": {
            "false_positive_review_cost": review_cost,
            "missed_fraud_cost": missed_fraud_cost,
            "total": fp * review_cost + fn * missed_fraud_cost,
        },
    }
