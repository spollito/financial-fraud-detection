# Implementation tasks

## Phase 1 and baseline

- [x] Initialize a standalone repository with a pinned Python project and isolated setup instructions.
- [x] Document Kaggle acquisition and protect raw data, credentials, and run artifacts from Git.
- [x] Implement the deterministic fixture generator and baseline CSV validation, deduplication, and join.
- [x] Implement chronological splits, training-only EDA, and a class-weighted logistic baseline.
- [x] Add tests for schema checks, temporal correctness, held-out data isolation, scoring, and serialization.
- [x] Configure a GitHub Actions workflow for linting, tests and a synthetic CLI run.
- [ ] Acquire the labeled IEEE-CIS CSV files and review actual training-partition EDA.

## Later phases

- [ ] Phase 2. Add PySpark local ingestion, partitioned Parquet, S3 support, and history features.
- [ ] Phase 3. Add temporal Optuna studies, XGBoost, PyTorch, calibration, temporal confidence
  intervals, and comparative held-out metrics after model selection.
- [ ] Phase 4. Add SHAP, model card generation, MLflow tracking, registry and promotion criteria.
- [ ] Phase 5. Add FastAPI, Docker/Compose, ONNX export and quantization, and latency/quality benchmarks.
- [ ] Phase 6. Add drift and delayed-label monitoring, replay and alerts, optional ECS/S3 deployment,
  and the full portfolio benchmark report.
