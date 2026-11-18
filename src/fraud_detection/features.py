"""Point-in-time feature contract for the tabular baseline."""

import numpy as np
import pandas as pd

from .ingestion import AMOUNT, IDENTITY_CATEGORIES, IDENTITY_MISSING, TRANSACTION_CATEGORIES

CATEGORICAL = (*TRANSACTION_CATEGORIES, *IDENTITY_CATEGORIES)
RAW_FEATURES = (AMOUNT, IDENTITY_MISSING, *CATEGORICAL)
NUMERICAL = ("log_amount", IDENTITY_MISSING)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.loc[:, list(RAW_FEATURES)].copy()
    features["log_amount"] = np.log1p(features.pop(AMOUNT))
    return features
