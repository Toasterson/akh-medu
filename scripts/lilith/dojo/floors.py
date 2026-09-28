"""The frozen exam floors (`exam-floors-v1.json`).

The file's sha256 is pinned here. Loading refuses a file that no longer hashes
to the pin: floors are frozen before any candidate is scored (plan slice C4),
so a change means a new `exam-floors-v2.json` with its own pin.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Iterator

from dojo.errors import FrozenFloorsError

FLOORS_V1 = Path(__file__).with_name("exam-floors-v1.json")
FLOORS_V1_SHA256 = "5db3576893c97454b59f8f5d906c78937083aea29053a511c0abd18b5253be89"


@dataclasses.dataclass(frozen=True)
class Floor:
    gate: str
    family: str
    metric: str
    minimum: float | None
    maximum: float | None

    def passes(self, value: float) -> bool:
        if self.minimum is not None and value < self.minimum:
            return False
        return self.maximum is None or value <= self.maximum


def load_floors(path: Path = FLOORS_V1, pinned_sha256: str = FLOORS_V1_SHA256) -> dict:
    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != pinned_sha256:
        raise FrozenFloorsError(f"{path} hashes to {actual}, pinned {pinned_sha256}")
    return json.loads(data)


def floors(document: dict | None = None) -> Iterator[Floor]:
    document = document if document is not None else load_floors()
    for metric, bound in document["hard_constraints"].items():
        yield Floor("hard", "all_families", metric, None, bound["max"])
    for gate in ("gate_b_chief_of_staff", "gate_c_illumos_and_steering", "gate_e_environment"):
        for family, metrics in document[gate].items():
            for metric, bound in metrics.items():
                yield Floor(gate, family, metric, bound.get("min"), bound.get("max"))
