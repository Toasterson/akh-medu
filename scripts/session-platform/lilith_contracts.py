"""Operator-selected, versioned artifact contracts for local Lilith fixtures."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from lilith_verification import verify_clamp, verify_u64_range


@dataclass(frozen=True)
class Contract:
    id: str
    artifact: str
    task: str
    verifier: object

    def grant(self):
        return dict(id=self.id, artifact=self.artifact, specification=self.task,
                    verifier_sha256=hashlib.sha256(Path(__file__).with_name('lilith_verification.py').read_bytes()).hexdigest())


CONTRACTS = {
    'integer-clamp/1': Contract('integer-clamp/1', 'clamp_u8.py',
        'Write a Python function clamp_u8(value: int) -> int that returns 0 for integers below 0, '
        '255 for integers above 255, otherwise value. Use only if comparisons and returns.', verify_clamp),
    'u64-range/1': Contract('u64-range/1', 'range_within_u64.py',
        'Write range_within_u64(start: int, length: int, limit: int) -> bool. Inputs are integers. '
        'Return False if any input is outside unsigned 64-bit range 0..18446744073709551615. '
        'Otherwise return whether the half-open range [start, start+length) fits within [0, limit). '
        'An empty range at start==limit is valid. Reject start>limit even for an empty range. '
        'Use the overflow-safe comparison length <= limit - start after checking start <= limit. '
        'Use only integer/bool literals, comparisons, boolean operators, subtraction, if and return.', verify_u64_range),
}
DEFAULT_CONTRACT = 'integer-clamp/1'
