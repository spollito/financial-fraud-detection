# Phase 2 synthetic smoke run

PySpark 4.0.1 ran in `local[2]` on Windows with JDK 21 and Python 3.13.7. The synthetic
fixture used seed 42. The Windows PyArrow writer produced three Parquet partitions and
the logistic pipeline trained directly from the resulting directory.

| Measurement | Observed |
| --- | ---: |
| Transactions retained | 720 |
| Transactions without an identity record | 240 |
| Chronological train / validation / test rows | 432 / 144 / 144 |
| Spark preprocessing and Parquet write, excluding JVM startup | 5.445 s |
| Total size of partitioned Parquet files | 30,933 bytes |
| Logistic baseline fit | 0.013 s |
| Serialized model size | 2,194 bytes |

The [machine-readable run report](synthetic-spark-baseline.json) includes configuration,
input hashes, partition boundaries, and synthetic evaluation metrics. Timing is a single
local measurement. Full-dataset throughput and model quality remain unmeasured.
