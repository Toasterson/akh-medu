"""Replayable, assistant-authored reference curriculum; never teacher episodes or sealed eval."""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class Exercise:
    id: str
    family: str
    split: str
    specification: str
    source: str
    mutant: str
    misconception: str
    cases: tuple


def exercises():
    alignment = tuple((n, a) for n in range(65) for a in range(1, 18)) + ((2**64-1, 4096), (0, 4096))
    mask = tuple(itertools.product(range(32), repeat=2)) + ((2**63, 2**63), (2**63, 1), (0, 0))
    endian = tuple(itertools.product(range(256), repeat=2))
    interval = tuple(itertools.product(range(5), repeat=4))
    serials = tuple(itertools.product((0, 1, 2, 2**31-1, 2**31, 2**31+1, 2**32-2, 2**32-1), repeat=2))
    return (
        Exercise('align-up/1', 'alignment', 'development_holdout',
            'Implement align_up(n, alignment) for nonnegative integer n and positive integer alignment. Return the smallest multiple of alignment >= n. Alignment need not be a power of two. Python unbounded integer reference only.',
            'def align_up(n, alignment):\n    return n + (-n % alignment)\n',
            'def align_up(n, alignment):\n    return (n + alignment - 1) & -alignment\n',
            'Bit-mask alignment silently assumes power-of-two alignment.', alignment),
        Exercise('is-power-two/1', 'alignment', 'development_holdout',
            'Implement is_power_two(n) for integer n. Return a bool: true exactly for positive powers of two; zero and negative integers are false.',
            'def is_power_two(n):\n    return n > 0 and (n & (n - 1)) == 0\n',
            'def is_power_two(n):\n    return (n & (n - 1)) == 0\n',
            'The bit trick alone accepts zero.', tuple((n,) for n in range(-8, 1025))),
        Exercise('has-all-flags/1', 'permission_masks', 'train_candidate',
            'Implement has_all_flags(available, required) for nonnegative integer bit masks. Return bool: every required bit must be available. Requiring zero bits always succeeds.',
            'def has_all_flags(available, required):\n    return available & required == required\n',
            'def has_all_flags(available, required):\n    return available & required != 0\n',
            'Any matching flag is weaker than all required flags; empty requirements are valid.', mask),
        Exercise('decode-le16/1', 'byte_order', 'train_candidate',
            'Implement decode_le16(low, high) for integer bytes in 0..255. Return the unsigned 16-bit integer encoded in little-endian byte order. No input validation outside that domain is required.',
            'def decode_le16(low, high):\n    return low + 256 * high\n',
            'def decode_le16(low, high):\n    return 256 * low + high\n',
            'Reversing byte significance decodes big-endian instead.', endian),
        Exercise('encode-le16/1', 'byte_order', 'train_candidate',
            'Implement encode_le16(value) for unsigned 16-bit integers. Return a tuple (low, high) of integer bytes representing little-endian byte order.',
            'def encode_le16(value):\n    return (value % 256, value // 256)\n',
            'def encode_le16(value):\n    return (value // 256, value % 256)\n',
            'The first byte must contain the least-significant bits.', tuple((n,) for n in range(65536))),
        Exercise('negative-errno/1', 'error_conventions', 'train_candidate',
            'Implement negative_errno(code) for integers. Preserve zero and negative values; negate positive values. This is an explicit abstract integer convention, not an illumos syscall ABI claim.',
            'def negative_errno(code):\n    if code > 0:\n        return -code\n    return code\n',
            'def negative_errno(code):\n    return -code\n',
            'Blind negation corrupts already-normalized negative errors.', tuple((n,) for n in range(-4096, 4097))),
        Exercise('interval-overlap/1', 'interval_intersection', 'development_holdout',
            'Implement overlaps(a, b, c, d) for integers defining half-open intervals [a,b) and [c,d). Empty or reversed intervals do not overlap. Return a bool.',
            'def overlaps(a, b, c, d):\n    return a < b and c < d and a < d and c < b\n',
            'def overlaps(a, b, c, d):\n    return a <= d and c <= b\n',
            'Touching endpoints and empty intervals are not overlap.', interval),
        Exercise('serial-newer-u32/1', 'modular_sequence_order', 'development_holdout',
            'Implement serial_newer(a, b) for unsigned 32-bit integer sequence numbers. Return bool: a is newer exactly when the forward modular distance from b to a is in 1..2147483647. Equal and exactly-half-space distances return false.',
            'def serial_newer(a, b):\n    delta = (a - b) & 4294967295\n    return 0 < delta < 2147483648\n',
            'def serial_newer(a, b):\n    return a > b\n',
            'Ordinary unsigned ordering fails across sequence wraparound.', serials),
    )


def oracle(exercise, args):
    # Deliberately different constructions from reference implementations.
    if exercise.id == 'align-up/1':
        n, alignment = args
        q, remainder = divmod(n, alignment)
        return (q + (1 if remainder else 0)) * alignment
    if exercise.id == 'is-power-two/1':
        return args[0] in {2**bit for bit in range(11)}
    if exercise.id == 'has-all-flags/1':
        available, required = args
        available_bits = {bit for bit in range(64) if available // (2**bit) % 2}
        required_bits = {bit for bit in range(64) if required // (2**bit) % 2}
        return required_bits.issubset(available_bits)
    if exercise.id == 'decode-le16/1':
        return int.from_bytes(bytes(args), 'little')
    if exercise.id == 'encode-le16/1':
        return tuple(args[0].to_bytes(2, 'little'))
    if exercise.id == 'negative-errno/1':
        return -abs(args[0])
    if exercise.id == 'interval-overlap/1':
        a, b, c, d = args
        return bool(set(range(a, b)).intersection(range(c, d)))
    if exercise.id == 'serial-newer-u32/1':
        a, b = args
        distance = a - b
        if distance < 0:
            distance += 2**32
        return distance in range(1, 2**31)
    raise ValueError('unknown exercise')


def compile_reference(source):
    # Restricted pure syntax reduces accidental side effects. This is NOT a
    # resource-isolated execution sandbox for adversarial model-generated code.
    allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
               ast.Assign, ast.Name, ast.Load, ast.Store, ast.Constant, ast.BinOp,
               ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Add, ast.Sub, ast.Mult,
               ast.Mod, ast.FloorDiv, ast.Tuple, ast.If, ast.BitAnd, ast.USub, ast.And, ast.Or, ast.Eq, ast.NotEq,
               ast.Lt, ast.LtE, ast.Gt, ast.GtE)
    if len(source.encode()) > 4096:
        raise ValueError('source too large')
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('expected one pure function')
    if tree.body[0].decorator_list:
        raise ValueError('decorators forbidden')
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError('unsupported reference syntax')
    scope = {'__builtins__': {}}
    exec(compile(tree, '<curated-reference>', 'exec'), scope)
    return scope[tree.body[0].name]


def verify(exercise, source):
    fn = compile_reference(source)
    failures = []
    for args in exercise.cases:
        expected, actual = oracle(exercise, args), fn(*args)
        if type(actual) is not type(expected) or actual != expected:
            failures.append(dict(inputs=args, expected=expected, actual=actual))
    return dict(passed=not failures, cases=len(exercise.cases), failed_cases=len(failures),
                first_counterexample=failures[0] if failures else None, source_sha256=sha(source))


def build(output):
    items = exercises()
    train_families = {e.family for e in items if e.split == 'train_candidate'}
    holdout_families = {e.family for e in items if e.split == 'development_holdout'}
    if train_families & holdout_families:
        raise ValueError('semantic family leakage')
    verifier_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipts, candidates, holdouts, corrections = [], [], [], []
    for exercise in items:
        receipt = verify(exercise, exercise.source)
        mutant = verify(exercise, exercise.mutant)
        if not receipt['passed'] or mutant['passed']:
            raise ValueError('reference or negative-control verification failed')
        provenance = dict(origin='assistant_authored_reference', teacher_session=None,
                          teacher_model=None, native_illumos_validated=False,
                          verifier_sha256=verifier_hash, cases_sha256=sha(canonical(exercise.cases)))
        receipts.append(dict(id=exercise.id, family=exercise.family, split=exercise.split,
                             reference=receipt, negative_control=mutant, provenance=provenance))
        row = dict(id=exercise.id, family=exercise.family, split=exercise.split,
                   input=exercise.specification, output=exercise.source,
                   source_sha256=sha(exercise.source), provenance=provenance)
        if exercise.split == 'train_candidate':
            candidates.append(row)
            corrections.append(dict(id=exercise.id, family=exercise.family,
                input=exercise.specification, incorrect_source=exercise.mutant,
                counterexample=mutant['first_counterexample'], diagnosis=exercise.misconception,
                corrected_source=exercise.source, provenance=provenance))
        else:
            holdouts.append({key: row[key] for key in ('id', 'family', 'split', 'input')})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('candidates.jsonl', candidates), ('corrections.jsonl', corrections),
                       ('development-holdouts.jsonl', holdouts)]:
        (output / name).write_text(''.join(canonical(row)+'\n' for row in rows))
    manifest = dict(schema='lilith/reference-curriculum@1', train_families=sorted(train_families),
        development_holdout_families=sorted(holdout_families), references=len(items),
        candidate_examples=len(candidates), correction_examples=len(corrections),
        reference_checks=sum(r['reference']['cases'] for r in receipts), receipts=receipts,
        limitations=['public development holdout, not sealed certification',
            'finite reference checks, not proof of correctness', 'no teacher-model or agent execution provenance',
            'no weight update or coding-quality improvement demonstrated',
            'Python integer semantics; no native illumos, C overflow, ABI or kernel validation'])
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps({key: result[key] for key in ('references', 'candidate_examples', 'correction_examples', 'reference_checks')}))


if __name__ == '__main__':
    main()
