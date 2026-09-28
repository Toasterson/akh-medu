import sys
from pathlib import Path

import pytest

LILITH = Path(__file__).resolve().parents[2]
if str(LILITH) not in sys.path:
    sys.path.insert(0, str(LILITH))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from fixtures import World  # noqa: E402


@pytest.fixture
def world(tmp_path):
    return World(tmp_path / "snapshots")
