"""Distributed IEEE-CIS validation, chronology, and history feature generation."""

import json
import os
import time
from itertools import islice
from pathlib import Path

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from .evaluation import PARTITIONS, SPLIT
from .features import HISTORY_PROXY, HISTORY_SECONDS
from .ingestion import (
    AMOUNT,
    ID,
    IDENTITY_COLUMNS,
    IDENTITY_MISSING,
    TARGET,
    TIME,
    TRANSACTION_COLUMNS,
    file_sha256,
)

TRANSACTION_INPUT = (*TRANSACTION_COLUMNS, *HISTORY_PROXY)


def _read_csv(spark: SparkSession, path: str, columns: tuple[str, ...]) -> DataFrame:
    raw = spark.read.options(header=True, mode="FAILFAST", enforceSchema=False).csv(path)
    if missing := sorted(set(columns) - set(raw.columns)):
        raise ValueError(f"Required CSV columns are missing: {missing}")
    return raw.select(*(F.trim(F.col(name)).alias(name) for name in columns)).replace("", None)


def _numeric_integer(name: str) -> Column:
    raw = F.col(name)
    return F.when(raw.rlike(r"^[0-9]+$"), raw.try_cast("long"))


def _validate(
    raw: DataFrame, *, transaction: bool, caches: list[DataFrame]
) -> tuple[DataFrame, dict]:
    name = "transaction" if transaction else "identity"
    typed = raw.withColumn(ID, _numeric_integer(ID))
    valid = F.col(ID) > 0
    if transaction:
        typed = typed.withColumns(
            {
                TIME: _numeric_integer(TIME),
                TARGET: _numeric_integer(TARGET),
                AMOUNT: F.col(AMOUNT).try_cast("double"),
            }
        )
        amount = F.col(AMOUNT)
        valid = (
            valid
            & (F.col(TIME) >= 0)
            & F.col(TARGET).isin(0, 1)
            & (amount >= 0)
            & (amount < float("inf"))
        )
    # Collapse equivalent values on the declared feature schema, retaining their multiplicity.
    grouped = typed.groupBy(*typed.columns).count().persist()
    caches.append(grouped)
    stats = grouped.agg(
        F.coalesce(F.sum("count"), F.lit(0)).alias("raw_rows"),
        F.coalesce(F.sum("count") - F.count("*"), F.lit(0)).alias("duplicates"),
        F.coalesce(F.sum(F.when(valid, F.lit(0)).otherwise(F.col("count"))), F.lit(0)).alias(
            "invalid"
        ),
    ).first()
    if stats.invalid:
        raise ValueError(f"{name} CSV has {stats.invalid} invalid row(s)")
    unique = grouped.drop("count")
    if unique.groupBy(ID).count().where(F.col("count") > 1).limit(1).count():
        raise ValueError(f"{name} CSV contains conflicting duplicate transaction IDs")
    return unique, {"raw_rows": stats.raw_rows, "duplicates_removed": stats.duplicates}


def with_history(frame: DataFrame) -> DataFrame:
    """Use the inclusive lower bound and exclude the current timestamp and all its peers."""
    missing = F.col(HISTORY_PROXY[0]).isNull() | F.col(HISTORY_PROXY[1]).isNull()
    history = (
        Window.partitionBy(*HISTORY_PROXY, F.when(missing, F.col(ID)))
        .orderBy(TIME)
        .rangeBetween(-HISTORY_SECONDS, -1)
    )
    return (
        frame.withColumns(
            {
                "history_key_missing": missing.cast("int"),
                "history_count_7d": F.count("*").over(history),
                "_prior_mean": F.avg(AMOUNT).over(history),
            }
        )
        .withColumn(
            "history_amount_ratio_7d",
            F.when(F.col("_prior_mean") > 0, F.try_divide(F.col(AMOUNT), F.col("_prior_mean"))),
        )
        .drop("_prior_mean")
    )


def _split(frame: DataFrame, rows: int, train: float, validation: float) -> DataFrame:
    if rows < 3:
        raise ValueError("At least three transactions are needed for three temporal partitions")
    ordered = frame.select(TIME).orderBy(TIME)
    first, second = (
        ordered.offset(int(rows * part)).first()[0] for part in (train, train + validation)
    )
    return frame.withColumn(
        SPLIT,
        F.when(F.col(TIME) < first, "train")
        .when(F.col(TIME) < second, "validation")
        .otherwise("test"),
    )


def _location(base: str, child: str) -> str:
    return (
        f"{base.rstrip('/')}/{child}"
        if base.startswith("s3a://")
        else (Path(base) / child).resolve().as_uri()
    )


def _hadoop_path(spark: SparkSession, uri: str):
    context = spark.sparkContext
    path = context._jvm.org.apache.hadoop.fs.Path(uri)
    return path, path.getFileSystem(context._jsc.hadoopConfiguration())


def _source_versions(spark: SparkSession, source: str, filenames: tuple[str, ...]) -> dict:
    versions = {}
    for filename in filenames:
        if source.startswith("s3a://"):
            path, fs = _hadoop_path(spark, _location(source, filename))
            status = fs.getFileStatus(path)
            versions[filename] = {
                "bytes": status.getLen(),
                "modified_epoch_ms": status.getModificationTime(),
            }
        else:
            path = Path(source) / filename
            versions[filename] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return versions


def preprocess(
    spark: SparkSession, source: str, output: str, dataset: str, train: float, validation: float
) -> dict:
    start = time.perf_counter()
    if not 0 < train < 1 or not 0 < validation < 1 - train:
        raise ValueError("Train and validation fractions must be positive and leave room for test")
    filenames = ("train_transaction.csv", "train_identity.csv")
    versions = _source_versions(spark, source, filenames)
    caches: list[DataFrame] = []
    # Prevent projection pushdown from skipping malformed CSV record detection.
    spark.conf.set("spark.sql.csv.parser.columnPruning.enabled", "false")
    try:
        transaction, transaction_stats = _validate(
            _read_csv(spark, _location(source, filenames[0]), TRANSACTION_INPUT),
            transaction=True,
            caches=caches,
        )
        identity, identity_stats = _validate(
            _read_csv(spark, _location(source, filenames[1]), IDENTITY_COLUMNS),
            transaction=False,
            caches=caches,
        )
        joined = (
            transaction.join(identity.withColumn("_identity_present", F.lit(1)), ID, "left")
            .withColumn(IDENTITY_MISSING, F.col("_identity_present").isNull().cast("int"))
            .drop("_identity_present")
            .persist()
        )
        caches.append(joined)
        rows = transaction_stats["raw_rows"] - transaction_stats["duplicates_removed"]
        partitions = _split(joined, rows, train, validation)
        split_counts = {
            row[SPLIT]: dict(
                rows=row.rows, min_transaction_dt=row.start, max_transaction_dt=row.end
            )
            for row in partitions.groupBy(SPLIT)
            .agg(F.count("*").alias("rows"), F.min(TIME).alias("start"), F.max(TIME).alias("end"))
            .collect()
        }
        if set(split_counts) != set(PARTITIONS):
            raise ValueError("Timestamp ties leave an empty partition. Adjust the split fractions.")
        engineered = with_history(partitions)
        quality = {
            "dataset": dataset,
            "synthetic_only": dataset == "synthetic",
            "spark_version": spark.version,
            "rows": rows,
            "inputs": versions,
            "transaction": transaction_stats,
            "identity": identity_stats,
            "orphan_identity_rows": identity.join(transaction.select(ID), ID, "left_anti").count(),
            "transactions_without_identity": joined.where(F.col(IDENTITY_MISSING) == 1).count(),
            "schema": dict(engineered.dtypes),
            "partitions": split_counts,
            "requested_split_fractions": {"train": train, "validation": validation},
            "history_window_seconds": HISTORY_SECONDS,
            "history_proxy": list(HISTORY_PROXY),
        }
        _write(spark, engineered, output, quality, start)
        return quality
    finally:
        for cached in reversed(caches):
            cached.unpersist()


def _write(spark: SparkSession, frame: DataFrame, output: str, quality: dict, start: float) -> None:
    if output.startswith("s3a://"):
        root, fs = _hadoop_path(spark, output)
        if fs.exists(root):
            raise FileExistsError(f"Output already exists: {output}")
    else:
        Path(output).mkdir(parents=True, exist_ok=False)

    export = frame.drop(*HISTORY_PROXY)
    if os.name == "nt" and not output.startswith("s3a://"):
        _write_windows_parquet(export, Path(output) / "transactions")
    else:
        export.write.mode("errorifexists").partitionBy(SPLIT).parquet(
            _location(output, "transactions")
        )
    quality["schema"] = {
        key: value for key, value in quality["schema"].items() if key not in HISTORY_PROXY
    }
    quality["processing_seconds"] = time.perf_counter() - start
    document = json.dumps(quality, indent=2, allow_nan=False) + "\n"
    if output.startswith("s3a://"):
        report, fs = _hadoop_path(spark, _location(output, "quality.json"))
        stream = fs.create(report, False)
        try:
            stream.write(bytearray(document.encode("utf-8")))
        finally:
            stream.close()
    else:
        (Path(output) / "quality.json").write_text(document, encoding="utf-8")


def _write_windows_parquet(frame: DataFrame, output: Path) -> None:
    """Write local files without Hadoop's Windows native filesystem dependency."""
    import pyarrow as pa
    import pyarrow.dataset as ds
    from pyspark.sql.pandas.types import to_arrow_schema

    schema = to_arrow_schema(frame.schema)
    rows = frame.toLocalIterator()

    def batches():
        while batch := list(islice(rows, 8192)):
            yield pa.RecordBatch.from_pylist([row.asDict() for row in batch], schema=schema)

    ds.write_dataset(
        batches(),
        str(output),
        schema=schema,
        format="parquet",
        partitioning=ds.partitioning(pa.schema([(SPLIT, pa.string())]), flavor="hive"),
    )


def spark_session(master: str, shuffle_partitions: int) -> SparkSession:
    return (
        SparkSession.builder.master(master)
        .appName("fraud-preprocess")
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
