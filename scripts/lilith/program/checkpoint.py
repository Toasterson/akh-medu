#!/usr/bin/env python3
"""Atomic checkpoints and program state for the Lilith five-stage training program (stdlib only).

A checkpoint is a directory ``checkpoints/ckpt-<seq>/`` holding the engine's tensors
(adapter, optimizer, RNG: whatever ``save_training_state`` wrote) plus ``state.json``
(the stage cursor, step counts, elapsed seconds, scheduler position, per-stage results)
and ``COMPLETE.json`` (sha256 + size of every other file, written last). The directory
is built under a ``.tmp-`` name and renamed into place, then ``LATEST`` is replaced
atomically, so a crash at any point leaves either the previous checkpoint or the new
one, never a half-written one. ``newest_valid`` re-hashes before a resume trusts it.

Nothing here imports torch: the engine owns tensor formats, this module owns the
protocol, so the whole resume logic is testable on a laptop.
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Iterator

STATE_SCHEMA = 'lilith/program-state@1'
COMPLETE_SCHEMA = 'lilith/program-checkpoint-complete@1'
KEEP_CHECKPOINTS = 2


class ProgramError(RuntimeError):
    """Program failure with a stable diagnostic code and a hint (miette-style: code + help)."""

    def __init__(self, code: str, message: str, help: str):
        super().__init__(f'[{code}] {message} (help: {help})')
        self.code = code
        self.help = help


# ------------------------------------------------------------------- atomic files

def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes_atomic(path: Path, data: bytes, mode: int = 0o600) -> None:
    """tmp + fsync + rename in the same directory; the file is never observed half-written."""
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    with tmp.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.chmod(mode)
    os.replace(tmp, path)
    fsync_directory(path.parent)


def write_json_atomic(path: Path, value: Any, mode: int = 0o600) -> None:
    write_bytes_atomic(Path(path), (json.dumps(value, indent=2, sort_keys=True) + '\n').encode(), mode)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one line and flush it to disk (the step receipts survive a SIGKILL)."""
    with Path(path).open('a') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def truncate_jsonl(path: Path, keep: Callable[[dict[str, Any]], bool]) -> int:
    """Rewrite a JSONL file keeping only lines for which ``keep`` holds; returns lines dropped.

    Used on resume: receipts of steps taken after the checkpoint are re-executed, so their
    old lines are removed instead of duplicated.
    """
    path = Path(path)
    if not path.is_file():
        return 0
    kept, dropped = [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1          # a torn last line from a kill mid-append
            continue
        if keep(record):
            kept.append(line)
        else:
            dropped += 1
    if dropped:
        write_bytes_atomic(path, ''.join(f'{line}\n' for line in kept).encode())
    return dropped


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------ stage cursor

class Phase(str, enum.Enum):
    """What a unit of the program does; the order here is the order within a stage."""
    SFT = 'sft'
    PREFERENCE = 'preference'
    ONPOLICY_WAIT = 'onpolicy_wait'
    ONPOLICY_SFT = 'onpolicy_sft'
    ONPOLICY_PREFERENCE = 'onpolicy_preference'


@dataclasses.dataclass(frozen=True)
class Unit:
    """One pass over one dataset, or the on-policy handshake."""
    phase: Phase
    pass_index: int = 0


@dataclasses.dataclass(frozen=True)
class Cursor:
    """The *next* batch to run: stage, unit within the stage, batch within the unit.

    ``phase``, ``pass_index`` and ``order_seed`` are derived from the stage's unit list but
    stored too, so a checkpoint is readable without the manifest.
    """
    stage_index: int = 0
    unit_index: int = 0
    batch_index: int = 0
    phase: str = Phase.SFT.value
    pass_index: int = 0
    order_seed: int = 0

    def to_record(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> 'Cursor':
        return cls(**{field.name: record[field.name] for field in dataclasses.fields(cls)})


@dataclasses.dataclass
class ProgramState:
    """Everything beyond tensors that a resume needs."""
    cursor: Cursor = dataclasses.field(default_factory=Cursor)
    global_step: int = 0
    checkpoint_seq: int = 0
    elapsed_seconds: float = 0.0
    windows: int = 0
    # stage name -> {'sft': n, 'preference': n, 'onpolicy_sft': n, 'onpolicy_preference': n}
    stage_steps: dict[str, dict[str, int]] = dataclasses.field(default_factory=dict)
    # stage name -> {'dev': {...}, 'adapter_sha256': ..., 'finished_at': ...}
    stage_results: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    # stage name -> accepted on-policy archive {'archive', 'sha256', 'adapter_sha256', 'records': n}
    onpolicy: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    # position of the (stateless) LR schedule: segment, step within it, total, last lr
    scheduler: dict[str, Any] = dataclasses.field(default_factory=dict)
    rng: dict[str, Any] = dataclasses.field(default_factory=dict)   # python RNG; torch RNG is a tensor file

    def to_record(self) -> dict[str, Any]:
        record = dataclasses.asdict(self)
        record['schema'] = STATE_SCHEMA
        record['cursor'] = self.cursor.to_record()
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> 'ProgramState':
        if record.get('schema') != STATE_SCHEMA:
            raise ProgramError('lilith::program::state_schema', f'state schema {record.get("schema")!r} is not {STATE_SCHEMA}',
                               'resume with the same program code that wrote the checkpoint')
        kwargs = {field.name: record[field.name] for field in dataclasses.fields(cls) if field.name in record}
        kwargs['cursor'] = Cursor.from_record(record['cursor'])
        return cls(**kwargs)

    def steps_in(self, stage: str, kind: str) -> int:
        return self.stage_steps.get(stage, {}).get(kind, 0)

    def count_step(self, stage: str, kind: str) -> None:
        bucket = self.stage_steps.setdefault(stage, {})
        bucket[kind] = bucket.get(kind, 0) + 1
        self.global_step += 1


# -------------------------------------------------------------------- the store

class CheckpointStore:
    """``<run>/checkpoints/``: build, seal, find and prune checkpoint directories."""

    def __init__(self, run_dir: Path, keep: int = KEEP_CHECKPOINTS):
        self.root = Path(run_dir) / 'checkpoints'
        self.keep = keep
        self.root.mkdir(parents=True, exist_ok=True)

    def _name(self, seq: int) -> str:
        return f'ckpt-{seq:06d}'

    def clean_stale(self) -> list[str]:
        """Remove ``.tmp-*`` directories left by a kill in the middle of a checkpoint."""
        removed = []
        for path in sorted(self.root.glob('.tmp-*')):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
        return removed

    def begin(self, seq: int) -> Path:
        tmp = self.root / f'.tmp-{self._name(seq)}'
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(mode=0o700)
        return tmp

    def commit(self, tmp: Path, state: ProgramState) -> Path:
        """Write ``state.json``, seal with ``COMPLETE.json`` and rename into place."""
        write_json_atomic(tmp / 'state.json', state.to_record())
        files = {path.name: {'sha256': sha256_file(path), 'bytes': path.stat().st_size}
                 for path in sorted(tmp.iterdir()) if path.is_file() and path.name != 'COMPLETE.json'}
        write_json_atomic(tmp / 'COMPLETE.json', {'schema': COMPLETE_SCHEMA, 'seq': state.checkpoint_seq,
                                                  'global_step': state.global_step, 'files': files})
        fsync_directory(tmp)
        final = self.root / self._name(state.checkpoint_seq)
        if final.exists():
            raise ProgramError('lilith::program::checkpoint_exists', f'{final} already exists',
                               'checkpoint sequence numbers are write-once; the run directory was edited by hand')
        os.rename(tmp, final)
        fsync_directory(self.root)
        write_bytes_atomic(self.root / 'LATEST', (final.name + '\n').encode())
        self.prune()
        return final

    def _seqs(self) -> list[int]:
        found = []
        for path in self.root.iterdir():
            if path.is_dir() and path.name.startswith('ckpt-'):
                try:
                    found.append(int(path.name.removeprefix('ckpt-')))
                except ValueError:
                    continue
        return sorted(found)

    def prune(self) -> list[str]:
        removed = []
        for seq in self._seqs()[:-self.keep]:
            shutil.rmtree(self.root / self._name(seq), ignore_errors=True)
            removed.append(self._name(seq))
        return removed

    def verify(self, directory: Path) -> ProgramState:
        """Check every file against ``COMPLETE.json`` and return the state; raises on any drift."""
        complete_path = directory / 'COMPLETE.json'
        if not complete_path.is_file():
            raise ProgramError('lilith::program::checkpoint_incomplete', f'{directory.name} has no COMPLETE.json',
                               'the checkpoint was cut short; resume uses the previous one')
        complete = read_json(complete_path)
        for name, expected in complete['files'].items():
            path = directory / name
            if not path.is_file() or path.stat().st_size != expected['bytes'] or sha256_file(path) != expected['sha256']:
                raise ProgramError('lilith::program::checkpoint_corrupt', f'{directory.name}/{name} differs from COMPLETE.json',
                                   'restore the volume copy or resume from the previous checkpoint')
        return ProgramState.from_record(read_json(directory / 'state.json'))

    def newest_valid(self) -> tuple[Path, ProgramState] | None:
        """Newest checkpoint whose files verify; if checkpoints exist and none verifies, raise."""
        problems: list[str] = []
        for seq in reversed(self._seqs()):
            directory = self.root / self._name(seq)
            try:
                return directory, self.verify(directory)
            except ProgramError as error:
                problems.append(str(error))
        if problems:
            raise ProgramError('lilith::program::no_valid_checkpoint',
                               f'{len(problems)} checkpoint(s) exist and none verifies: {problems[0]}',
                               'inspect checkpoints/ by hand; never edit state.json')
        return None

    def directories(self) -> Iterator[Path]:
        for seq in self._seqs():
            yield self.root / self._name(seq)
