"""The shared generation loop: ask, assess structurally, retry once with the failed check names, seal, write.

A `Task` is one seed. `run_task` is resumable by construction: the caller skips keys the progress file already
lists as done or excluded, and a teacher failure leaves the key unrecorded so the next run retries it.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable, Sequence

from .acp_worker import Teacher, TeacherError
from .checks import Check, CitationResolver, LanguageProfiles, failed
from .records import Draft, RunDir, ask_with_transcript, seal

MAX_ATTEMPTS = 2


@dataclasses.dataclass
class Context:
    run: RunDir
    languages: LanguageProfiles
    resolver: CitationResolver
    repo_root: Path
    snapshot_root: Path | None = None
    snapshot_cutoff: str = "20261002T000000Z"
    max_attempts: int = MAX_ATTEMPTS
    retry_excluded: bool = False


@dataclasses.dataclass
class Assessment:
    checks: list[Check]
    # Builds the draft once the transcript hash is known; None when the reply was unusable (no JSON header).
    make_draft: Callable[[str], Draft] | None = None
    extra: dict = dataclasses.field(default_factory=dict)
    # Paired records sealed only when the main record passed: (kind, draft, checks) built from the sealed main record.
    make_extras: Callable[[str, dict], list[tuple[str, Draft, list[Check]]]] | None = None


@dataclasses.dataclass
class Task:
    kind: str
    key: str
    label: str
    run: Callable[[Teacher], "Outcome"]


@dataclasses.dataclass
class Outcome:
    kind: str
    key: str
    status: str  # done | excluded | error
    record_id: str | None = None
    failed_checks: tuple[str, ...] = ()
    detail: str = ""
    extra: dict = dataclasses.field(default_factory=dict)


def feedback_suffix(bad: Sequence[Check]) -> str:
    lines = [f"- {c.name}: {c.detail}" if c.detail else f"- {c.name}" for c in bad]
    return ("\n\nYour previous reply failed these structural checks, so it cannot be used:\n" + "\n".join(lines)
            + "\nReturn a corrected full reply, ending with the fenced JSON block.")


def run_task(ctx: Context, teacher: Teacher, kind: str, key: str, build_prompt: Callable[[], str],
             assess: Callable[[str, str], Assessment]) -> Outcome:
    base = build_prompt()
    attempts: list[dict] = []
    prompt = base
    assessment: Assessment | None = None
    try:
        for attempt in range(ctx.max_attempts):
            reply, entry = ask_with_transcript(teacher, prompt)
            assessment = assess(reply.text, entry["prompt_sha256"])
            entry["checks"] = [c.to_dict() for c in assessment.checks]
            attempts.append(entry)
            bad = failed(assessment.checks)
            if not bad:
                break
            prompt = base + feedback_suffix(bad)
    except TeacherError as error:
        ctx.run.mark(kind, key, "error", detail=str(error)[:200], error_kind=error.kind)
        return Outcome(kind, key, "error", detail=str(error)[:200], extra={"error_kind": error.kind})
    assert assessment is not None
    transcript_sha = ctx.run.save_transcript(kind, key, attempts)
    bad = failed(assessment.checks)
    if assessment.make_draft is None:
        ctx.run.mark(kind, key, "excluded", failed_checks=[c.name for c in bad], transcript_sha256=transcript_sha)
        return Outcome(kind, key, "excluded", failed_checks=tuple(c.name for c in bad), extra=assessment.extra)
    record = seal(assessment.make_draft(transcript_sha), assessment.checks)
    ctx.run.write_record(kind, record)
    status = "excluded" if bad else "done"
    if not bad and assessment.make_extras is not None:
        for extra_kind, extra_draft, extra_checks in assessment.make_extras(transcript_sha, record):
            ctx.run.write_record(extra_kind, seal(extra_draft, extra_checks))
    ctx.run.mark(kind, key, status, record_id=record["record_id"], failed_checks=[c.name for c in bad],
                 transcript_sha256=transcript_sha, **{k: v for k, v in assessment.extra.items() if k == "pack"})
    return Outcome(kind, key, status, record["record_id"], tuple(c.name for c in bad), extra=assessment.extra)
