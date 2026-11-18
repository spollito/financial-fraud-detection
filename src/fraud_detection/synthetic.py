"""Generate deterministic data for tests and local installation checks."""

from pathlib import Path

import numpy as np
import pandas as pd

from .ingestion import AMOUNT, ID, TARGET, TIME


def write_fixture(output: Path, *, rows: int = 720, seed: int = 42) -> None:
    if rows < 120:
        raise ValueError("Generate at least 120 rows to exercise the temporal splits")
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    index = np.arange(rows)
    label = (index % 19 == 0).astype(int)
    transaction = pd.DataFrame(
        {
            ID: 1_000_001 + index,
            TIME: (index // 2) * 7200,
            TARGET: label,
            AMOUNT: np.round(rng.lognormal(3.5 + 0.6 * label, 0.9), 2),
            "ProductCD": rng.choice(["W", "C", "H"], rows),
            "card4": rng.choice(["visa", "mastercard"], rows),
            "card6": rng.choice(["debit", "credit"], rows),
            "P_emaildomain": rng.choice(["example.com", "example.net", None], rows),
            "R_emaildomain": rng.choice(["example.org", None], rows),
        }
    )
    identity = pd.DataFrame(
        {ID: transaction[ID], "DeviceType": rng.choice(["desktop", "mobile", None], rows)}
    ).loc[index % 3 != 0]
    transaction.sample(frac=1, random_state=seed).to_csv(
        output / "train_transaction.csv", index=False
    )
    identity.to_csv(output / "train_identity.csv", index=False)
