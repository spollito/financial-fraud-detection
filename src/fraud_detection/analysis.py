"""Generate a reproducible EDA report from the oldest partition."""

import math

import pandas as pd
from scipy.stats import mannwhitneyu, norm

from .evaluation import require_both_classes, temporal_split
from .ingestion import AMOUNT, ID, IDENTITY_MISSING, TARGET, TIME


def _rate(labels: pd.Series) -> dict:
    total, fraud = len(labels), int(labels.sum())
    rate = fraud / total
    z = norm.ppf(0.975)
    scale = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / scale
    radius = z * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / scale
    return {
        "rows": total,
        "fraud_rows": fraud,
        "fraud_rate": rate,
        "wilson_95": [max(0.0, center - radius), min(1.0, center + radius)],
    }


def _amount_summary(amount: pd.Series) -> dict:
    quantiles = amount.quantile([0.01, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "mean": float(amount.mean()),
        "std": float(amount.std(ddof=0)),
        "quantiles": {str(key): float(value) for key, value in quantiles.items()},
    }


def describe(frame: pd.DataFrame, train_fraction: float, validation_fraction: float) -> dict:
    train, _, _ = temporal_split(frame, train_fraction, validation_fraction)
    require_both_classes(train, "Train")
    amount = train[AMOUNT]
    positive, negative = amount[train[TARGET].eq(1)], amount[train[TARGET].eq(0)]
    comparison = mannwhitneyu(positive, negative, method="asymptotic")
    q1, q3 = amount.quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    weeks = (train[TIME] - train[TIME].min()) // (7 * 86_400)
    correlation_fields = [AMOUNT, TIME, IDENTITY_MISSING, TARGET]
    correlations = train[correlation_fields].corr(method="spearman")
    categories = {}
    for field in ("ProductCD", "DeviceType"):
        categories[field] = [
            {"category": str(value) if pd.notna(value) else None, **_rate(group[TARGET])}
            for value, group in train.groupby(field, dropna=False)
        ]
    return {
        "scope": "oldest partition only",
        "max_transaction_dt": int(train[TIME].max()),
        "class_distribution": _rate(train[TARGET]),
        "missing_rates": {key: float(value) for key, value in train.isna().mean().items()},
        "transactions_without_identity": int(train[IDENTITY_MISSING].sum()),
        "amount": {
            "all": _amount_summary(amount),
            "fraud": _amount_summary(positive),
            "legitimate": _amount_summary(negative),
            "upper_iqr_fence": float(upper),
            "above_upper_fence": int((amount > upper).sum()),
        },
        "fraud_rates_by_category": categories,
        "relative_weeks": [
            {
                "week": int(week),
                "median_amount": float(group[AMOUNT].median()),
                **_rate(group[TARGET]),
            }
            for week, group in train.groupby(weeks)
        ],
        "spearman_correlation": correlations.astype(object)
        .where(pd.notna(correlations), None)
        .to_dict(),
        "amount_comparison": {
            "mann_whitney_u": float(comparison.statistic),
            "asymptotic_p_value": float(comparison.pvalue),
            "rank_biserial_fraud_vs_legitimate": float(
                2 * comparison.statistic / (len(positive) * len(negative)) - 1
            ),
            "interpretation": "Exploratory independence approximation; transactions may cluster.",
        },
        "leakage_review": {
            "excluded_fields": [ID, TIME, TARGET],
            "timestamp_ties_cross_boundaries": False,
            "identity_availability": "Assumed present at transaction time; verify before serving.",
            "preprocessing_fit": "Training partition only",
        },
    }
