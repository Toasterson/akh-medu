import sys
from pathlib import Path

import pytest

LILITH = Path(__file__).resolve().parents[2]
DOJO_TESTS = LILITH / "dojo" / "tests"
for path in (LILITH, DOJO_TESTS, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fixtures import World  # noqa: E402  (the dojo's synthetic snapshot worlds)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path / "snapshots")
