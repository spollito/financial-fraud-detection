# Financial Fraud Detection

An IEEE-CIS fraud detection project with a local logistic-regression baseline and a PySpark
preprocessing pipeline. Both paths validate the selected CSV fields, retain transactions
without identity records, and use chronological validation.

## Run locally in VS Code

Open this repository folder in VS Code and use its terminal. The commands below run in PowerShell.
Python 3.11–3.13 is supported by the dependency pins. The local smoke run used Python 3.13.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -c constraints.txt -e ".[dev]"
.\.venv\Scripts\fraud.exe fixture --output data/fixtures/quickstart
.\.venv\Scripts\fraud.exe inspect --dataset synthetic --data data/fixtures/quickstart --output reports/runs/quickstart-eda
.\.venv\Scripts\fraud.exe train --dataset synthetic --data data/fixtures/quickstart --output reports/runs/quickstart-baseline --evaluate-test
.\.venv\Scripts\python.exe -m pytest -q
```

In **Python: Select Interpreter**, choose the repository's `.venv` for editing and debugging.
On macOS/Linux, use `python3 -m venv .venv` and substitute `.venv/bin/python` and
`.venv/bin/fraud` for the corresponding Windows executables. Use a new output directory
for each run so previous results stay available.

`inspect` writes `eda.json`. `train` writes `baseline.json` and a serialized scikit-learn
pipeline in `baseline.joblib`. Load serialized artifacts only from a trusted producer.
The synthetic generator contains deliberately injected signal and a simple label schedule.
Its scores are installation checks only. See the [local smoke report](reports/synthetic-smoke.md).

## Acquire the IEEE-CIS dataset

1. Sign in at the [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
   and accept its rules if prompted.
2. Download the data and extract `train_transaction.csv` and `train_identity.csv` directly
   into `data/raw/`. You can also use the [official Kaggle CLI](https://github.com/Kaggle/kaggle-cli)
   after authenticating outside this repository. From the repo root the download command is
   `kaggle competitions download -c ieee-fraud-detection -p data/raw`. Extract the two named
   CSV files from the ZIP before running the pipeline.
3. Keep credentials outside the repository. Raw and generated files are ignored by Git.
   The competition test files have no public labels and are unnecessary here.

```powershell
.\.venv\Scripts\fraud.exe inspect --dataset ieee-cis --data data/raw --output reports/runs/ieee-eda
.\.venv\Scripts\fraud.exe train --dataset ieee-cis --data data/raw --output reports/runs/ieee-baseline
```

The newest partition remains unscored by default. Add `--evaluate-test` to a new run only after
committing to the feature set, model, calibration policy, and threshold selection procedure.
Record the input SHA-256 hashes from the report whenever you compare experiments.

## PySpark preprocessing and history features

Install the optional Spark dependencies and set `JAVA_HOME` to your JDK 21 installation.
In the existing Windows setup that is `C:\Program Files\Java\jdk-21`.

```powershell
$env:JAVA_HOME = 'C:\Program Files\Java\jdk-21'
$env:SPARK_LOCAL_IP = '127.0.0.1'
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -c constraints.txt -e ".[dev,spark]"
.\.venv\Scripts\fraud.exe fixture --output data/fixtures/spark-demo
.\.venv\Scripts\fraud.exe preprocess --dataset synthetic --data data/fixtures/spark-demo --output data/processed/spark-demo
.\.venv\Scripts\fraud.exe train --dataset synthetic --processed data/processed/spark-demo --output reports/runs/spark-demo
```

For real data use `--dataset ieee-cis --data data/raw` on `preprocess` and a new output
directory. The Spark schema adds `card1` and `card2` to the Phase 1 input fields; regenerate
any older fixtures. Use `--train-fraction` and `--validation-fraction` at preprocessing time
to change the split. `train --processed` consumes those saved partitions.

Spark validates and splits the records, left joins identity, and calculates a seven-day
transaction count and amount-to-prior-mean ratio using the pair `(card1, card2)` as a proxy.
It counts only offsets in `[t - 604800, t - 1]` and excludes every transaction with the same
timestamp as the current transaction. Earlier validation or test transactions can contribute
their observed amounts to subsequent rows. Labels never enter the history features.
The pair can mix unrelated people or fragment a single card's activity. Missing keys are
never grouped together. The ratio is null for a missing or zero prior mean, and the trained
pipeline encodes that missingness explicitly. Live serving would need the same history
window and input fields from a point-in-time source.

The preprocessing run writes `transactions/split=…/` Parquet partitions and a
`quality.json` manifest with counts, schema, partition boundaries and elapsed time.
Windows streams rows from Spark through PyArrow for local Parquet writes. Other platforms
and S3 use Spark's distributed writer. The manifest is written after the Parquet data.
Use a new output path for retries, and only train on directories with a finished manifest.
See the [Phase 2 smoke run](reports/synthetic-spark.md) for the local verification results.

### Optional AWS S3

The preprocessing command also accepts `s3a://bucket/prefix` for either path. Supply the
Hadoop S3A connector matching the Hadoop 3.4.1 jars bundled with the PySpark pin. For example,
set `PYSPARK_SUBMIT_ARGS` to
`--packages org.apache.hadoop:hadoop-aws:3.4.1 pyspark-shell` before starting the CLI.
Configure the bucket region and use workload credentials or a local short-lived AWS session
that S3A can resolve. Grant read access to the source prefix and the list and object permissions
needed by the writer on a dedicated run prefix. Keep credentials out of command history and Git.
Use immutable source prefixes because S3 input metadata records object size and modification
time. To use the local training CLI, copy the completed run directory, including `quality.json`,
to local disk first. AWS processing has not been run against a live bucket.
See the [Hadoop S3A documentation](https://hadoop.apache.org/docs/r3.4.1/hadoop-aws/tools/hadoop-aws/index.html).

## Evaluation

- Default chronology uses approximately 60% for training, 20% for validation, and 20% for test.
  Every transaction with a tied timestamp goes into the later partition at a split boundary.
- Average precision is scikit-learn's `average_precision_score`, the precision-weighted sum
  of recall increments. This uses the stepwise definition of AP. The report also contains
  ROC-AUC, precision, recall, F1, false-positive rate, confusion counts, Brier score, and
  reliability bins.
- The decision threshold maximizes validation F1 with higher thresholds breaking ties.
  The report's `roc_recall_at_max_fpr` is a retrospective ranking diagnostic on the scored
  partition. The confusion matrix uses the frozen validation threshold.
- The baseline reports raw class-weighted probability estimates. Calibration fitting is
  planned for model comparison. The default costs use 5 units per false positive and 250 per
  false negative as simulation assumptions. Override with `--review-cost` and
  `--missed-fraud-cost`.
- The [specification](SPEC.md) describes the feature and data contracts. [TASKS.md](TASKS.md)
  tracks implemented and planned milestones.
