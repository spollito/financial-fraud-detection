import json

import pytest

pytest.importorskip("pyspark")
pytest.importorskip("pyarrow")

from fraud_detection.cli import _load_processed  # noqa: E402
from fraud_detection.ingestion import ID  # noqa: E402
from fraud_detection.spark_pipeline import preprocess, spark_session, with_history  # noqa: E402
from fraud_detection.training import train_baseline  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    session = spark_session("local[2]", 4)
    session.sparkContext.setLogLevel("ERROR")
    try:
        yield session
    finally:
        session.stop()


def test_history_excludes_ties_and_unrelated_or_unknown_cards(spark):
    frame = spark.sql("""
        SELECT * FROM VALUES
            (1, 0L, 10D, 'a', 'x'),
            (2, 0L, 30D, 'a', 'x'),
            (3, 604800L, 40D, 'a', 'x'),
            (4, 604801L, 80D, 'a', 'x'),
            (5, 604801L, 90D, 'b', 'x'),
            (6, 0L, 10D, 'a', NULL),
            (7, 604801L, 90D, 'a', NULL),
            (8, 1L, 0D, 'c', 'x'),
            (9, 2L, 5D, 'c', 'x')
        AS input(TransactionID, TransactionDT, TransactionAmt, card1, card2)
    """)
    rows = {row[ID]: row.asDict() for row in with_history(frame).collect()}
    assert [rows[i]["history_count_7d"] for i in range(1, 10)] == [0, 0, 2, 1, 0, 0, 0, 0, 1]
    assert rows[3]["history_amount_ratio_7d"] == rows[4]["history_amount_ratio_7d"] == 2
    assert rows[7]["history_key_missing"] == 1
    assert rows[9]["history_amount_ratio_7d"] is None


def test_partitioned_parquet_round_trip_and_history_model(spark, fixture_data, tmp_path):
    run = tmp_path / "parquet"
    quality = preprocess(spark, str(fixture_data), str(run), "synthetic", 0.61, 0.20)
    frame, manifest = _load_processed(run, "synthetic")
    assert manifest == quality == json.loads((run / "quality.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == 120
    assert manifest["transactions_without_identity"] == 40
    assert set(frame["split"]) == {"train", "validation", "test"}
    assert "card1" not in frame and "card1" not in manifest["schema"]
    assert frame["history_count_7d"].max() > 0
    assert len(list((run / "transactions").glob("split=*"))) == 3
    model, report = train_baseline(
        frame,
        train_fraction=0.61,
        validation_fraction=0.2,
        evaluate_test=True,
        max_fpr=0.01,
        review_cost=5,
        missed_fraud_cost=250,
        use_history=True,
    )
    assert report["partitions"] == quality["partitions"]
    assert report["model"]["uses_history"]
    assert 0 <= report["test"]["average_precision"] <= 1
    inputs = frame[report["raw_features"]]
    assert model.predict_proba(inputs).shape == (120, 2)


def test_spark_rejects_invalid_rows_before_creating_output(spark, fixture_data, tmp_path):
    path = fixture_data / "train_transaction.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    columns = lines[0].split(",")
    first = lines[1].split(",")
    first[columns.index("TransactionAmt")] = "-1"
    lines[1] = ",".join(first)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run = tmp_path / "bad"
    with pytest.raises(ValueError, match="1 invalid row"):
        preprocess(spark, str(fixture_data), str(run), "synthetic", 0.6, 0.2)
    assert not run.exists()
