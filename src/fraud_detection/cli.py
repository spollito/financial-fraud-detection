"""Local entry points for dataset inspection and baseline experiments."""

import argparse
import json
import math
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .analysis import describe
from .evaluation import SPLIT
from .features import CATEGORICAL, HISTORY_FIELDS, RAW_FEATURES
from .ingestion import ID, TARGET, TIME, load_dataset
from .synthetic import write_fixture
from .training import train_baseline


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("fixture", help="Generate fixture CSV files in a new directory")
    fixture.add_argument("--output", required=True, type=Path)
    fixture.add_argument("--rows", type=int, default=720)
    fixture.add_argument("--seed", type=int, default=42)
    for command in ("inspect", "train", "preprocess"):
        sub = commands.add_parser(command)
        if command == "train":
            source = sub.add_mutually_exclusive_group(required=True)
            source.add_argument("--processed", type=Path, help="Local Spark run directory")
        else:
            source = sub
        source.add_argument(
            "--data", required=command != "train", type=str if command == "preprocess" else Path
        )
        sub.add_argument("--dataset", required=True, choices=("synthetic", "ieee-cis"))
        sub.add_argument(
            "--output",
            required=True,
            type=str if command == "preprocess" else Path,
            help="New output directory",
        )
        sub.add_argument("--train-fraction", type=float)
        sub.add_argument("--validation-fraction", type=float)
        if command == "preprocess":
            sub.add_argument("--master", default="local[2]")
            sub.add_argument("--shuffle-partitions", type=int, default=8)
        if command == "train":
            sub.add_argument("--evaluate-test", action="store_true")
            sub.add_argument("--max-fpr", type=float, default=0.01)
            sub.add_argument("--review-cost", type=float, default=5.0)
            sub.add_argument("--missed-fraud-cost", type=float, default=250.0)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "fixture":
            write_fixture(args.output, rows=args.rows, seed=args.seed)
        elif args.command == "preprocess":
            _preprocess(args)
        else:
            _run(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"fraud: {error}\n")
    print(f"{args.command}: wrote {args.output}")


def _fractions(args: argparse.Namespace) -> tuple[float, float]:
    return (
        0.6 if args.train_fraction is None else args.train_fraction,
        0.2 if args.validation_fraction is None else args.validation_fraction,
    )


def _preprocess(args: argparse.Namespace) -> None:
    try:
        from .spark_pipeline import preprocess, spark_session
    except ModuleNotFoundError as error:
        if error.name != "pyspark":
            raise
        raise ValueError("Install the project's [spark] extra to run preprocessing") from error
    if args.shuffle_partitions < 1:
        raise ValueError("--shuffle-partitions must be positive")
    spark = spark_session(args.master, args.shuffle_partitions)
    try:
        preprocess(spark, args.data, args.output, args.dataset, *_fractions(args))
    finally:
        spark.stop()


def _load_processed(root: Path, dataset: str) -> tuple[pd.DataFrame, dict]:
    quality = json.loads((root / "quality.json").read_text(encoding="utf-8"))
    if quality.get("dataset") != dataset:
        raise ValueError("--dataset does not match the Spark quality report")
    try:
        frame = pd.read_parquet(
            root / "transactions",
            engine="pyarrow",
            columns=[ID, TIME, TARGET, SPLIT, *RAW_FEATURES, *HISTORY_FIELDS],
        )
    except ImportError as error:
        raise ValueError("Install the project's [spark] extra to read processed Parquet") from error
    if len(frame) != quality.get("rows"):
        raise ValueError("Parquet row count does not match the Spark quality report")
    for column in CATEGORICAL:
        frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
    return frame, quality


def _run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.command == "train":
        if not 0 <= args.max_fpr <= 1:
            raise ValueError("--max-fpr must be in [0, 1]")
        if any(
            not math.isfinite(cost) or cost < 0
            for cost in (args.review_cost, args.missed_fraud_cost)
        ):
            raise ValueError("Both simulation costs must be finite and nonnegative")

    processed = getattr(args, "processed", None)
    train_fraction, validation_fraction = _fractions(args)
    if processed:
        if args.train_fraction is not None or args.validation_fraction is not None:
            raise ValueError("Set temporal split fractions in preprocess when training on Parquet")
        frame, quality = _load_processed(processed, args.dataset)
        split_fractions = quality["requested_split_fractions"]
    else:
        frame, quality = load_dataset(args.data)
        split_fractions = {"train": train_fraction, "validation": validation_fraction}
    metadata = {
        "dataset": args.dataset,
        "synthetic_only": args.dataset == "synthetic",
        "quality": quality,
        "requested_split_fractions": split_fractions,
        "versions": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in (
                    "financial-fraud-detection",
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "scipy",
                    "joblib",
                )
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        if args.command == "inspect":
            eda = describe(frame, train_fraction, validation_fraction)
            _write_json(stage / "eda.json", {**metadata, **eda})
        else:
            model, report = train_baseline(
                frame,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                evaluate_test=args.evaluate_test,
                max_fpr=args.max_fpr,
                review_cost=args.review_cost,
                missed_fraud_cost=args.missed_fraud_cost,
                use_history=bool(processed),
            )
            joblib.dump(model, stage / "baseline.joblib", compress=3)
            _write_json(
                stage / "baseline.json",
                {
                    **metadata,
                    **report,
                    "artifact_bytes": (stage / "baseline.joblib").stat().st_size,
                },
            )
        stage.rename(args.output)


if __name__ == "__main__":
    main()
