# Financial Fraud Detection

An IEEE-CIS fraud detection project with a runnable local logistic-regression baseline. The
current pipeline validates the selected CSV fields, left joins identity records, analyzes the
oldest partition, and trains a class-weighted classifier with chronological validation.

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
