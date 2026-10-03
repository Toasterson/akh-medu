"""`lilith/program-record@1` records, the on-disk layout of a generation run, and resumable progress.

Layout under `out_dir` (private; never inside the repo):

    <kind>.jsonl             admitted records (weight 1.0)
    <kind>.excluded.jsonl    records that failed a structural check (weight 0), with the failed checks
    illumos-pack.jsonl       `{work_item_id, subsystem, source: "panel"}` for the dojo's illumos-guidance family
    progress/<kind>.jsonl    one line per finished seed: done | excluded | error (resume skips done and excluded)
    transcripts/<kind>/<key>.json   prompt + teacher reply per attempt (private)
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import canon
from .acp_worker import TEACHER_ID, Teacher, TeacherReply
from .checks import Check, failed

KINDS = ("variation", "sparring", "guidance", "lineage")
PRIVATE_ROOT_NAME = "program-20261003"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seed_id(seed: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of a seed record (the drill corpus carries no ids)."""
    return sha256_text(canonical_json(seed))


@dataclasses.dataclass
class Draft:
    """What a generator knows about a record before it is checked and sealed."""

    source: str  # variation | sparring | guidance | lineage
    generator: str
    stage: str
    families: list[str]
    voice: str
    language: str
    messages: list[dict]
    component_id: str
    seed_ref: dict  # {"seed_record_id": ...} or {"work_item_id": ...}
    prompt_sha256: str
    transcript_sha256: str
    kata: str | None = None
    extra_ref: dict = dataclasses.field(default_factory=dict)
    provenance_extra: dict = dataclasses.field(default_factory=dict)


def seal(draft: Draft, checks: Sequence[Check]) -> dict:
    """The final record. `weight` is 1.0 when every check passed and 0.0 otherwise (the writer routes weight 0
    to the excluded file)."""
    bad = failed(checks)
    record: dict[str, Any] = {
        "schema": canon.SCHEMA,
        "source": draft.source,
        "source_ref": {"generator": draft.generator, "teacher": TEACHER_ID, "prompt_sha256": draft.prompt_sha256,
                       **draft.seed_ref, **draft.extra_ref},
        "stage": draft.stage,
        "families": list(draft.families),
        "voice": draft.voice,
        "language": draft.language,
        "kind": "sft",
        "weight": 0.0 if bad else 1.0,
        "messages": draft.messages,
        "component_id": draft.component_id,
        "split": "train",
        "provenance": {"teacher_transcript_sha256": draft.transcript_sha256,
                       "checks": [c.to_dict() for c in checks], **draft.provenance_extra},
        "verification": {"checks_passed": [c.name for c in checks if c.passed],
                         "checks_failed": [c.name for c in bad]},
    }
    if draft.kata is not None:
        record["kata"] = draft.kata
    record["record_id"] = sha256_text(canonical_json(record))
    return record


class RunDir:
    """The output directory of one run: thread-safe appends, progress, transcripts."""

    def __init__(self, out_dir: Path):
        self.out = Path(out_dir)
        for sub in ("progress", "transcripts"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # -- jsonl --------------------------------------------------------------------------------------------

    def _append(self, path: Path, value: Mapping[str, Any]) -> None:
        line = json.dumps(value, ensure_ascii=False, sort_keys=True)
        with self._lock, path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def read(self, name: str) -> list[dict]:
        path = self.out / name
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def write_record(self, kind: str, record: Mapping[str, Any]) -> None:
        name = f"{kind}.jsonl" if record.get("weight") == 1.0 else f"{kind}.excluded.jsonl"
        self._append(self.out / name, record)

    # -- progress -----------------------------------------------------------------------------------------

    def progress(self, kind: str) -> dict[str, dict]:
        """seed key -> its latest progress line."""
        latest: dict[str, dict] = {}
        for line in self.read(f"progress/{kind}.jsonl"):
            latest[line["key"]] = line
        return latest

    def finished(self, kind: str, retry_excluded: bool = False) -> set[str]:
        keep = {"done"} if retry_excluded else {"done", "excluded"}
        return {key for key, line in self.progress(kind).items() if line.get("status") in keep}

    def mark(self, kind: str, key: str, status: str, **fields: Any) -> None:
        self._append(self.out / "progress" / f"{kind}.jsonl", {"key": key, "status": status, **fields})

    # -- transcripts --------------------------------------------------------------------------------------

    def save_transcript(self, kind: str, key: str, attempts: Sequence[Mapping[str, Any]]) -> str:
        """Write the per-seed transcript and return its sha256."""
        directory = self.out / "transcripts" / kind
        directory.mkdir(parents=True, exist_ok=True)
        body = json.dumps({"key": key, "teacher": TEACHER_ID, "attempts": list(attempts)}, ensure_ascii=False, indent=1)
        (directory / f"{sha256_text(key)[:24]}.json").write_text(body, encoding="utf-8")
        return sha256_text(body)


def ask_with_transcript(teacher: Teacher, prompt: str) -> tuple[TeacherReply, dict]:
    reply = teacher.ask(prompt)
    return reply, {"prompt": prompt, "prompt_sha256": sha256_text(prompt), "reply": reply.text,
                   "stop_reason": reply.stop_reason, "elapsed_s": round(reply.elapsed_s, 2)}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
