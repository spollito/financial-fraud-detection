# Platform specification

## Objective

Build a reproducible platform for classifying IEEE-CIS transactions as fraudulent and measuring
ranking quality, false positives, calibration, cost, and deployment performance under
chronological evaluation.

## Current data and model contracts

1. Ingestion projects the baseline columns from the two labeled Kaggle CSV files. It validates
   required headers and finite, nonnegative numeric values. IDs are positive integers, timestamps
   are nonnegative integer offsets, and labels are binary. Invalid values and conflicting IDs
   fail the run. Duplicates equivalent on the normalized, projected columns collapse and are counted.
2. A one-to-one left join preserves each transaction. Empty categorical fields remain missing.
   Identity-row absence is distinct from a present identity record with a missing device type.
   The ingestion report records projected types, row counts, join quality, input hashes, and elapsed time.
3. Chronological partitions share neither transactions nor timestamps. EDA uses the oldest
   partition for feature decisions. Its nominal Wilson intervals and rank-test p-value use
   independence assumptions and serve as descriptive diagnostics for potentially clustered data.
4. The baseline pipeline accepts `TransactionAmt`, `identity_missing`, `ProductCD`, `card4`,
   `card6`, `P_emaildomain`, `R_emaildomain`, and `DeviceType`. It uses log amount, the join
   indicator, and one-hot encodings. The encoder tolerates new categories at inference.
   All fitted preprocessing and class weights derive from training only. The predictor excludes
   the label, row ID, timestamp, and unreconciled identity fields. Confirm that identity metadata
   is available by decision time before using the model to score live transactions.
5. The training stage selects a threshold on validation and writes a probability model with
   a JSON run report. Train and validation must contain both classes. The test partition requires
   both classes when enabled. Test data never participates in fitting or threshold selection.

## Planned architecture

| Stage | Acceptance criteria |
| --- | --- |
| Spark ingestion and feature pipeline | Run in local PySpark; validate and left join the full declared schema; produce partitioned Parquet and quality/timing reports; optionally read/write S3. Share a feature contract with training. Prove historical features use strictly earlier timestamps and document the limitations of entity proxies. |
| Comparative training | Compare logistic regression, XGBoost with early stopping, and a configurable PyTorch MLP with weighted loss, batching, checkpoints, early stopping, and CPU/GPU support. Tune with Optuna on expanding temporal windows. Freeze candidates, calibration and thresholds before scoring the final holdout. Report AP, fixed-recall false-positive reduction, simulated cost, and temporal uncertainty where practical. |
| Explainability and tracking | Store configuration, dataset hashes, seeds, runtimes, model versions and reports in MLflow. Document validation-gated registration and promotion. Generate SHAP global and individual explanations, FP/FN examples, and a model card explaining limitations. |
| Inference and optimization | Serve validated single/batch predictions and health, readiness and model-info endpoints in FastAPI. Make historical-feature supply explicit and avoid logging transaction values. Compare native PyTorch, ONNX Runtime and compatible quantized ONNX with common warm-up, hardware, inputs and quality tolerances. |
| Monitoring and deployment | Replay traffic and report schema, missingness, feature and score drift, latency, invalid request rates, and delayed-label performance. Provide Docker/Compose and optional ECS/S3 configuration with scoped IAM and cleanup guidance. |
| Benchmarks and CI | Record real processing, AP, false positives, latency, throughput and artifact sizes in machine-readable outputs. Label synthetic runs. CI uses small synthetic fixtures and skips full training on the competition dataset. |

## Initial acceptance criteria

The synthetic CLI run must exercise CSV validation and optional identity, produce strict
chronological splits, report train-only EDA, fit and serialize the baseline, and write
validation metrics. Tests must cover schema failures, ties at split boundaries, test isolation,
unknown categories, metrics, and model reload. A real-data evaluation requires the user to obtain
the Kaggle files under the competition's terms.
