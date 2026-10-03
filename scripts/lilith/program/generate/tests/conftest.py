import sys
from pathlib import Path

LILITH = Path(__file__).resolve().parents[3]
if str(LILITH) not in sys.path:
    sys.path.insert(0, str(LILITH))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import pytest  # noqa: E402

from program.generate.checks import CitationResolver, LanguageProfiles  # noqa: E402
from program.generate.records import RunDir  # noqa: E402
from program.generate.tasking import Context  # noqa: E402

REPO_ROOT = LILITH.parents[1]


@pytest.fixture
def ctx(tmp_path):
    """A context whose language profiles are empty (language is not judged) and whose citations never touch the network."""
    resolver = CitationResolver(None, get=lambda url: (404, ""))
    return Context(RunDir(tmp_path / "out"), LanguageProfiles({}), resolver, REPO_ROOT)
