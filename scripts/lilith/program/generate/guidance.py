"""Black guidance: `lilith/cos-guidance@1` findings on real illumos / OpenIndiana work items, written by the sealed
teacher akh acting as Lilith and verified against native evidence.

Work items come from the newest recorded snapshot at or before the cutoff (read-only), projects illumos-gate and
oi-userland. Every citation must resolve: man pages against a local illumos-gate checkout or illumos.org, source
files against the checkout or the raw illumos-gate `master` file. This is man-page/source resolution, NOT a native
check inside an OpenIndiana VM; the record says so (`provenance.native_check`).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

from . import canon
from .acp_worker import Teacher
from .checks import Check, extract_json, jaccard, non_empty
from .records import Draft, sha256_text
from .tasking import Assessment, Context, Outcome, Task, run_task

KIND = "guidance"
GENERATOR = "program.generate.guidance@1"
QUOTA = 120
PROJECTS = ("019dda27-b1c5-79b0-929b-f9debcbd0b0b", "019de483-cdd9-7f52-9f49-7db6177102b3")
MIN_DESCRIPTION_CHARS = 150
MAX_DESCRIPTION_CHARS = 3000
CANDIDATE_FACTOR = 2  # candidates drawn per quota slot; the driver stops at the quota
LANGUAGE_ROTATION = ("en", "en", "de", "en", "fr", "en")
AUDIENCE_ROTATION = ("implementer", "implementer", "product-manager", "implementer", "product-owner")
_BCP47 = re.compile(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$")
ALLOWED_HEADER_KEYS = {"contract", "language", "audience", "subsystem", "severity", "finding", "recommendation", "citations", "voice"}


@dataclasses.dataclass(frozen=True)
class Candidate:
    work_item_id: str
    code: str
    title: str
    description: str
    project_id: str
    snapshot_id: str
    observed_at: int
    detail_sha256: str
    language: str
    audience: str

    @property
    def key(self) -> str:
        return self.work_item_id


def select_candidates(root: Path, cutoff: str, quota: int = QUOTA, factor: int = CANDIDATE_FACTOR) -> list[Candidate]:
    """Deterministic, hash-ordered, alternating between the two projects. Read-only over the snapshot root."""
    import sys
    lilith = Path(__file__).resolve().parents[2]
    if str(lilith) not in sys.path:
        sys.path.insert(0, str(lilith))
    from dojo.snapshot import SnapshotStore  # noqa: PLC0415

    store = SnapshotStore.open(root, cutoff)
    snapshot = store.get(store.ids[-1])
    pools: list[list[Candidate]] = []
    for project_id in PROJECTS:
        collector = snapshot.project_collectors.get(project_id)
        pool: list[tuple[str, Candidate]] = []
        if collector is not None:
            for wid, state in snapshot.work_items_of(None if collector == "anima" else collector).items():
                if state.detail_ref is None:
                    continue
                found = snapshot.detail(wid)
                if found is None:
                    continue
                detail, ref = found
                item = detail.get("work_item") or {}
                description = str(item.get("description") or "").strip()
                if len(description) < MIN_DESCRIPTION_CHARS or not state.title:
                    continue
                order = hashlib.sha256(f"20261003:{wid}".encode()).hexdigest()
                pool.append((order, Candidate(wid, state.code, state.title, description[:MAX_DESCRIPTION_CHARS], project_id,
                                              snapshot.id, snapshot.unix_time, ref.sha256, "en", "implementer")))
        pools.append([c for _, c in sorted(pool, key=lambda t: t[0])])
    merged: list[Candidate] = []
    for position in range(max((len(p) for p in pools), default=0)):
        for pool in pools:
            if position < len(pool):
                merged.append(pool[position])
    limit = quota * factor
    return [dataclasses.replace(c, language=LANGUAGE_ROTATION[i % len(LANGUAGE_ROTATION)],
                                audience=AUDIENCE_ROTATION[i % len(AUDIENCE_ROTATION)]) for i, c in enumerate(merged[:limit])]


def user_message(c: Candidate) -> str:
    return f"work_item: {c.code} ({c.work_item_id})\ntitle: {c.title}\ndescription:\n{c.description}"


def build_prompt(c: Candidate) -> str:
    shape = ('{"contract": "lilith/cos-guidance@1", "language": "%s", "audience": "%s", "subsystem": "...", "severity": "...", '
             '"finding": "...", "recommendation": "...", "citations": [{"kind": "man-page", "reference": "svcadm(8)"}, '
             '{"kind": "source", "reference": "usr/src/..."}], "voice": "..."}') % (c.language, c.audience)
    voices = ", ".join(v.id for v in canon.VOICES)
    return (
        "You are Lilith, the operator's Chief of Staff. Part of your work is making sure implementers, the product manager and the "
        "product owner do the right work on illumos and OpenIndiana. You write the finding directly; do not use any tools.\n\n"
        "Here is a work item as the tracker returned it:\n"
        f"{user_message(c)}\n\n"
        f"Write one `lilith/cos-guidance@1` finding for the audience `{c.audience}` (implementer: review the brief or change against "
        "illumos reality; product-manager: scope, acceptance criteria, dependency order; product-owner: supported releases, packaging, "
        "upgrade path, scope cuts or additions).\n"
        "- `subsystem` is one of: smf, ips, zones, zfs, rbac, libc-abi, dtrace, portability-trap, other (kernel, drivers and anything else).\n"
        "- `severity` is note, concern or blocker.\n"
        "- `finding` states what is wrong, risky or missing and why, grounded in what the work item says; "
        "`recommendation` is the concrete next step for the audience. Do not invent facts about the code: where you are unsure, say what to check.\n"
        "- `citations`: at least one, only of kind `man-page` (written `name(section)` as in the illumos manual, for example "
        "`smf_method(7)`) or `source` (a repository-relative illumos-gate path such as `usr/src/uts/common/os/cpu.c`). "
        "Cite only pages and files you are sure exist in illumos-gate; every citation is checked and a single wrong one discards the whole finding. "
        "You cannot run an OpenIndiana VM here, so do not claim a VM check.\n"
        f"- Write `finding` and `recommendation` in {canon.LANGUAGE_NAMES[c.language]} ({c.language}); the citations stay as identifiers.\n"
        f"- `voice` is the one of your crew ({voices}) that suits this finding.\n\n"
        f"Finish with exactly one fenced ```json block of this shape:\n{shape}\n"
    )


def finding_object(c: Candidate, header: dict) -> dict:
    """The full cos-guidance@1 object, in the schema's field order, with this episode's receipt."""
    return {
        "contract": canon.GUIDANCE_CONTRACT,
        "language": header["language"],
        "audience": header["audience"],
        "subject": {"kind": "work-item", "id": c.work_item_id},
        "subsystem": header["subsystem"],
        "severity": header["severity"],
        "finding": header["finding"],
        "recommendation": header["recommendation"],
        "citations": [{"kind": x["kind"], "reference": x["reference"]} for x in header["citations"]],
        "receipts": [{"kind": "work-item", "reference": c.work_item_id, "observed_at": c.observed_at, "digest": c.detail_sha256}],
    }


def schema_checks(c: Candidate, header: dict) -> list[Check]:
    checks = [
        Check("contract_const", header.get("contract") == canon.GUIDANCE_CONTRACT, f"{header.get('contract')!r}"),
        Check("language_echo", header.get("language") == c.language and bool(_BCP47.match(str(header.get("language")))),
              f"{header.get('language')!r} vs {c.language!r}"),
        Check("audience_requested", header.get("audience") == c.audience and c.audience in canon.GUIDANCE_AUDIENCES,
              f"{header.get('audience')!r} vs {c.audience!r}"),
        Check("subsystem_enum", header.get("subsystem") in canon.GUIDANCE_SUBSYSTEMS, f"{header.get('subsystem')!r}"),
        Check("severity_enum", header.get("severity") in canon.GUIDANCE_SEVERITIES, f"{header.get('severity')!r}"),
        Check("finding_non_empty", non_empty(header.get("finding")), ""),
        Check("recommendation_non_empty", non_empty(header.get("recommendation")), ""),
        Check("voice_canonical", canon.canonical_voice(str(header.get("voice"))) is not None, f"{header.get('voice')!r}"),
        Check("no_unknown_keys", set(header) <= ALLOWED_HEADER_KEYS, f"extra: {sorted(set(header) - ALLOWED_HEADER_KEYS)}"),
    ]
    citations = header.get("citations")
    shaped = isinstance(citations, list) and citations and all(
        isinstance(x, dict) and set(x) == {"kind", "reference"} and non_empty(x.get("kind")) and non_empty(x.get("reference"))
        for x in citations)
    checks.append(Check("citations_shape", bool(shaped), "citations need {kind, reference} entries, at least one"))
    if non_empty(header.get("finding")) and non_empty(header.get("recommendation")):
        j = jaccard(str(header["finding"]), str(header["recommendation"]))
        checks.append(Check("finding_differs_from_recommendation", j < 0.8, f"jaccard {j:.2f}"))
    return checks


def assess(c: Candidate, ctx: Context, reply: str, prompt_sha: str) -> Assessment:
    header = extract_json(reply)
    checks: list[Check] = [Check("json_header", header is not None, "" if header else "no parseable fenced JSON object")]
    if header is None:
        return Assessment(checks)
    checks += schema_checks(c, header)
    resolutions: list = []
    shaped = next((x for x in checks if x.name == "citations_shape"), None)
    if shaped is not None and shaped.passed:
        citation_check, resolutions = ctx.resolver.check_all(header["citations"])
        checks.append(citation_check)
    if non_empty(header.get("finding")):
        checks.append(ctx.languages.matches(str(header["finding"]), c.language))
    usable = all(x.passed for x in checks if x.name in {"contract_const", "audience_requested", "subsystem_enum", "severity_enum",
                                                        "finding_non_empty", "recommendation_non_empty", "citations_shape",
                                                        "voice_canonical"})

    def make(transcript_sha: str) -> Draft:
        obj = finding_object(c, header)
        voice = canon.canonical_voice(str(header.get("voice"))) or "truthkeeper"
        messages = [{"role": "system", "content": canon.system_prompt(voice)},
                    {"role": "user", "content": user_message(c)},
                    {"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)}]
        return Draft(source="guidance", generator=GENERATOR, stage=canon.STAGE_BLACK,
                     families=["illumos-guidance", f"subsystem:{header.get('subsystem')}", f"audience:{c.audience}"],
                     voice=voice, language=c.language, messages=messages, component_id=c.work_item_id,
                     seed_ref={"work_item_id": c.work_item_id}, prompt_sha256=prompt_sha, transcript_sha256=transcript_sha,
                     provenance_extra={"subsystem": header.get("subsystem"), "audience": c.audience, "project_id": c.project_id,
                                       "snapshot_id": c.snapshot_id, "detail_sha256": c.detail_sha256,
                                       "native_check": "man-page and source resolution (local illumos-gate checkout or HTTPS); "
                                                       "not run inside an OpenIndiana VM",
                                       "citation_resolutions": [r.to_dict() for r in resolutions]})

    return Assessment(checks, make if usable else None)


def plan(ctx: Context, candidates: Sequence[Candidate]) -> list[Task]:
    done = ctx.run.finished(KIND, ctx.retry_excluded)
    tasks = []
    for candidate in candidates:
        if candidate.key in done:
            continue

        def run(teacher: Teacher, candidate: Candidate = candidate) -> Outcome:
            return run_task(ctx, teacher, KIND, candidate.key, lambda: build_prompt(candidate),
                            lambda r, p: assess(candidate, ctx, r, p))

        tasks.append(Task(KIND, candidate.key, f"guidance {candidate.code} {candidate.audience} {candidate.language}", run))
    return tasks


def pack_entries(records: Sequence[dict]) -> list[dict]:
    """illumos-pack.jsonl rows for the admitted guidance records (the dojo's `load_illumos_pack` format)."""
    seen: dict[str, dict] = {}
    for record in records:
        wid = record["source_ref"]["work_item_id"]
        seen.setdefault(wid, {"work_item_id": wid, "subsystem": record["provenance"]["subsystem"], "source": "panel"})
    return list(seen.values())
