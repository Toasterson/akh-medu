import sys
from pathlib import Path

import pytest

LILITH = Path(__file__).resolve().parents[2]
DOJO_TESTS = LILITH / "dojo" / "tests"
for entry in (LILITH, DOJO_TESTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from fixtures import World  # noqa: E402


@pytest.fixture
def world(tmp_path):
    return World(tmp_path / "snapshots")
