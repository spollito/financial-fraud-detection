"""Local entry points for dataset inspection and baseline experiments."""

import argparse
import json
import math
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path

import joblib

from .analysis import describe
from .features import RAW_FEATURES
from .ingestion import load_dataset
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
    for command in ("inspect", "train"):
        sub = commands.add_parser(command)
        sub.add_argument("--data", required=True, type=Path)
        sub.add_argument("--dataset", required=True, choices=("synthetic", "ieee-cis"))
        sub.add_argument("--output", required=True, type=Path, help="New output directory")
        sub.add_argument("--train-fraction", type=float, default=0.6)
        sub.add_argument("--validation-fraction", type=float, default=0.2)
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
        else:
            _run(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"fraud: {error}\n")
    print(f"{args.command}: wrote {args.output}")


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

    frame, quality = load_dataset(args.data)
    metadata = {
        "dataset": args.dataset,
        "synthetic_only": args.dataset == "synthetic",
        "quality": quality,
        "requested_split_fractions": {
            "train": args.train_fraction,
            "validation": args.validation_fraction,
        },
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
            eda = describe(frame, args.train_fraction, args.validation_fraction)
            _write_json(stage / "eda.json", {**metadata, **eda})
        else:
            model, report = train_baseline(
                frame,
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
                evaluate_test=args.evaluate_test,
                max_fpr=args.max_fpr,
                review_cost=args.review_cost,
                missed_fraud_cost=args.missed_fraud_cost,
            )
            joblib.dump(model, stage / "baseline.joblib", compress=3)
            _write_json(
                stage / "baseline.json",
                {
                    **metadata,
                    **report,
                    "raw_features": list(RAW_FEATURES),
                    "artifact_bytes": (stage / "baseline.joblib").stat().st_size,
                },
            )
        stage.rename(args.output)


if __name__ == "__main__":
    main()
