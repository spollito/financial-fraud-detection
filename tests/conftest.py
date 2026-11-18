from pathlib import Path

import pytest

from fraud_detection.synthetic import write_fixture


@pytest.fixture
def fixture_data(tmp_path: Path) -> Path:
    path = tmp_path / "fixture"
    write_fixture(path, rows=120)
    return path
