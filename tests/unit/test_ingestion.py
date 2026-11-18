from pathlib import Path

import pandas as pd
import pytest

from fraud_detection.ingestion import ID, IDENTITY_MISSING, load_dataset


def test_join_retains_transactions_and_marks_missing_identity(fixture_data: Path):
    frame, quality = load_dataset(fixture_data)
    assert len(frame) == 120
    assert frame[ID].is_unique
    assert frame.loc[frame[IDENTITY_MISSING].eq(1), "DeviceType"].isna().all()
    assert frame[IDENTITY_MISSING].sum() == quality["transactions_without_identity"] == 40
    assert quality["orphan_identity_rows"] == 0
    assert len(quality["input_sha256"]["train_transaction.csv"]) == 64


def test_exact_duplicate_is_removed_and_orphan_reported(fixture_data: Path):
    path = fixture_data / "train_identity.csv"
    rows = pd.read_csv(path)
    extra = rows.iloc[[0]].copy()
    extra[ID] = 88
    pd.concat([rows, rows.iloc[[0]], extra]).to_csv(path, index=False)
    frame, quality = load_dataset(fixture_data)
    assert len(frame) == 120
    assert quality["identity_duplicates_removed"] == 1
    assert quality["orphan_identity_rows"] == 1


def test_conflicting_duplicate_is_rejected(fixture_data: Path):
    path = fixture_data / "train_identity.csv"
    rows = pd.read_csv(path)
    conflicting = rows.iloc[[0]].assign(DeviceType="different-device")
    pd.concat([rows, conflicting]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        load_dataset(fixture_data)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("TransactionAmt", "-1", "finite, nonnegative"),
        ("TransactionAmt", "inf", "finite, nonnegative"),
        ("TransactionDT", "1.2", "nonnegative integer"),
        ("TransactionDT", "9223372036854775808", "outside the supported range"),
        ("TransactionID", "", "nonnegative integer"),
        ("isFraud", "2", "labels must be 0 or 1"),
    ],
)
def test_invalid_values_are_rejected(fixture_data: Path, field: str, value: str, message: str):
    path = fixture_data / "train_transaction.csv"
    rows = pd.read_csv(path, dtype=str)
    rows.loc[0, field] = value
    rows.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_dataset(fixture_data)


def test_missing_header_is_rejected(fixture_data: Path):
    path = fixture_data / "train_transaction.csv"
    pd.read_csv(path).drop(columns="TransactionDT").to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns.*TransactionDT"):
        load_dataset(fixture_data)


def test_largest_signed_integer_is_preserved(fixture_data: Path):
    path = fixture_data / "train_transaction.csv"
    rows = pd.read_csv(path, dtype=str)
    rows.loc[0, "TransactionDT"] = str(2**63 - 1)
    rows.to_csv(path, index=False)
    frame, _ = load_dataset(fixture_data)
    assert int(frame["TransactionDT"].max()) == 2**63 - 1


def test_empty_identity_table_is_supported(fixture_data: Path):
    (fixture_data / "train_identity.csv").write_text("TransactionID,DeviceType\n", encoding="utf-8")
    frame, quality = load_dataset(fixture_data)
    assert frame[IDENTITY_MISSING].eq(1).all()
    assert quality["raw_identity_rows"] == 0
