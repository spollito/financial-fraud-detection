# Phase 1 synthetic smoke run

The run used Windows, Python 3.13.7, the pinned project dependencies, fixture seed 42,
and the default 60/20/20 temporal split. The CLI ran once on 720 synthetic transactions.
The [EDA](synthetic-eda.json) and [baseline](synthetic-baseline.json) reports contain the
input hashes, versions, settings, and measurements.

| Check | Observed result |
| --- | ---: |
| Transactions retained after left join | 720 |
| Transactions without an identity record | 240 |
| Train / validation / test rows | 432 / 144 / 144 |
| CSV validation, join and hashing | 0.019 s |
| Baseline fit time | 0.011 s |
| Serialized baseline size | 2,021 bytes |
| Average precision on synthetic validation | 0.063 |
| Average precision on synthetic test | 0.169 |

The fixture's oldest partition contained 23 positive labels (5.3%). The amount comparison
showed a positive rank-biserial effect (0.409), consistent with the generator's larger
amounts for the positive class. There were 33 amounts above the upper IQR fence.
These checks exercise class weighting, log amount, and missing-identity handling. Feature
decisions for the competition dataset will depend on EDA from its oldest partition.

Timings are single-run measurements and will vary. The scores, effect estimate, and
processing time apply only to this fixture. IEEE-CIS performance and Spark processing
time are pending.
