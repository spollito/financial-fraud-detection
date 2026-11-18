"""Point-in-time feature contract for the tabular baseline."""

import numpy as np
import pandas as pd

from .ingestion import AMOUNT, IDENTITY_CATEGORIES, IDENTITY_MISSING, TRANSACTION_CATEGORIES

CATEGORICAL = (*TRANSACTION_CATEGORIES, *IDENTITY_CATEGORIES)
RAW_FEATURES = (AMOUNT, IDENTITY_MISSING, *CATEGORICAL)
NUMERICAL = ("log_amount", IDENTITY_MISSING)
HISTORY_FIELDS = ("history_count_7d", "history_amount_ratio_7d", "history_key_missing")
HISTORY_NUMERICAL = (*HISTORY_FIELDS, "history_ratio_missing")
HISTORY_PROXY = ("card1", "card2")
HISTORY_SECONDS = 7 * 86_400


def make_features(frame: pd.DataFrame, *, use_history: bool = False) -> pd.DataFrame:
    columns = (*RAW_FEATURES, *HISTORY_FIELDS) if use_history else RAW_FEATURES
    features = frame.loc[:, list(columns)].copy()
    features["log_amount"] = np.log1p(features.pop(AMOUNT))
    if use_history:
        ratio = features["history_amount_ratio_7d"]
        features["history_ratio_missing"] = ratio.isna().astype("int8")
        features["history_amount_ratio_7d"] = ratio.fillna(0.0)
    return features
