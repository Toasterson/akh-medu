#!/usr/bin/env python3
"""Apply the 0.9999 cosine gate to vision projections from several devices.

Takes the raw f32 dumps written by `vision_projector_probe --dump` and compares
every pair elementwise. Exits non-zero if any pair misses the gate.

Summary statistics are deliberately not the test: mean, standard deviation and
L2 norm are all invariant under a permutation or a sign flip of the vector, so
two devices can agree on every one of them while computing different things.

What this proves is that the backends compute the same function. It does not
prove that function is the reference implementation's - a misread resize policy
or mrope section split would be wrong identically everywhere and pass.
"""
import itertools
import math
import struct
import sys

GATE = 0.9999


def load(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    if len(raw) % 4:
        raise SystemExit(f"{path}: {len(raw)} bytes is not a whole number of f32")
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def compare(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise SystemExit("a zero projection has no direction to compare")
    return (
        dot / (left_norm * right_norm),
        math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right))) / left_norm,
        max(abs(a - b) for a, b in zip(left, right)),
    )


def main(argv):
    if len(argv) < 3:
        raise SystemExit(f"usage: {argv[0]} NAME=PATH NAME=PATH [NAME=PATH ...]")
    dumps = {}
    for entry in argv[1:]:
        name, _, path = entry.partition("=")
        if not path:
            raise SystemExit(f"expected NAME=PATH, got {entry!r}")
        dumps[name] = load(path)
    sizes = {len(values) for values in dumps.values()}
    if len(sizes) != 1:
        raise SystemExit(f"projections differ in length: {sizes}")

    print(f"{len(next(iter(dumps.values())))} elements per projection\n")
    print(f"{'pair':<20}{'cosine':>14}{'rel L2':>12}{'worst abs':>12}  gate")
    failed = []
    for left, right in itertools.combinations(sorted(dumps), 2):
        cosine, relative, worst = compare(dumps[left], dumps[right])
        ok = cosine >= GATE
        if not ok:
            failed.append((left, right, cosine))
        print(
            f"{left + ' vs ' + right:<20}{cosine:>14.10f}{relative:>12.3e}"
            f"{worst:>12.3e}  {'PASS' if ok else 'FAIL'}"
        )
    print()
    if failed:
        for left, right, cosine in failed:
            print(f"FAIL {left} vs {right}: cosine {cosine:.10f} below {GATE}")
        return 1
    print(f"all pairs meet the {GATE} cosine gate")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
