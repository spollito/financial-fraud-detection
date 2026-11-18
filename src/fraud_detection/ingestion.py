"""Validate the IEEE-CIS baseline columns and preserve unmatched transactions."""

import csv
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

ID = "TransactionID"
TIME = "TransactionDT"
TARGET = "isFraud"
AMOUNT = "TransactionAmt"
TRANSACTION_CATEGORIES = ("ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain")
IDENTITY_CATEGORIES = ("DeviceType",)
IDENTITY_MISSING = "identity_missing"
TRANSACTION_COLUMNS = (ID, TIME, TARGET, AMOUNT, *TRANSACTION_CATEGORIES)
IDENTITY_COLUMNS = (ID, *IDENTITY_CATEGORIES)


def _read_selected(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    missing = sorted(set(columns) - set(header))
    if missing or len(set(header)) != len(header):
        raise ValueError(f"{path.name}: invalid header (missing columns: {missing})")
    return pd.read_csv(
        path,
        usecols=list(columns),
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        on_bad_lines="error",
    ).loc[:, list(columns)]


def _integer(frame: pd.DataFrame, column: str, minimum: int) -> None:
    raw = frame[column].str.strip()
    if not raw.str.fullmatch(r"[0-9]+", na=False).all():
        raise ValueError(f"{column}: expected a nonnegative integer in every record")
    value = pd.to_numeric(raw, errors="coerce")
    if (
        value.dtype.kind not in "iu"
        or (value < minimum).any()
        or (value > np.iinfo(np.int64).max).any()
    ):
        raise ValueError(f"{column}: integer outside the supported range")
    frame[column] = value.astype("int64")


def _categories(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        frame[column] = frame[column].str.strip().replace("", np.nan)


def _deduplicate(frame: pd.DataFrame, label: str) -> tuple[pd.DataFrame, int]:
    unique = frame.drop_duplicates()
    conflicts = unique.loc[unique[ID].duplicated(keep=False), ID].nunique()
    if conflicts:
        raise ValueError(f"{label}: {conflicts} conflicting duplicate {ID} value(s)")
    return unique, len(frame) - len(unique)


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_dataset(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    start = time.perf_counter()
    transaction_path = raw_dir / "train_transaction.csv"
    identity_path = raw_dir / "train_identity.csv"
    transaction = _read_selected(transaction_path, TRANSACTION_COLUMNS)
    identity = _read_selected(identity_path, IDENTITY_COLUMNS)
    if transaction.empty:
        raise ValueError("train_transaction.csv has no records")

    _integer(transaction, ID, minimum=1)
    _integer(identity, ID, minimum=1)
    _integer(transaction, TIME, minimum=0)
    _integer(transaction, TARGET, minimum=0)
    if not transaction[TARGET].isin((0, 1)).all():
        raise ValueError(f"{TARGET}: labels must be 0 or 1")
    amount = pd.to_numeric(transaction[AMOUNT], errors="coerce")
    if not np.isfinite(amount).all() or (amount < 0).any():
        raise ValueError(f"{AMOUNT}: expected a finite, nonnegative amount in every record")
    transaction[AMOUNT] = amount.astype("float64")
    _categories(transaction, TRANSACTION_CATEGORIES)
    _categories(identity, IDENTITY_CATEGORIES)
    transaction, transaction_duplicates = _deduplicate(transaction, "transaction")
    identity, identity_duplicates = _deduplicate(identity, "identity")

    joined = transaction.merge(identity, on=ID, how="left", validate="one_to_one", indicator=True)
    joined[IDENTITY_MISSING] = joined.pop("_merge").eq("left_only").astype("int8")
    quality = {
        "rows": len(joined),
        "raw_transaction_rows": len(transaction) + transaction_duplicates,
        "raw_identity_rows": len(identity) + identity_duplicates,
        "transaction_duplicates_removed": transaction_duplicates,
        "identity_duplicates_removed": identity_duplicates,
        "orphan_identity_rows": int((~identity[ID].isin(transaction[ID])).sum()),
        "transactions_without_identity": int(joined[IDENTITY_MISSING].sum()),
        "schema": {key: str(value) for key, value in joined.dtypes.items()},
        "input_sha256": {
            path.name: file_sha256(path) for path in (transaction_path, identity_path)
        },
        "processing_seconds": time.perf_counter() - start,
    }
    return joined, quality
