"""Black lineage (Transmission): Naming-the-Move transmissions in the manual's four-field format (section 6).

Quota 100 generated = 60 from yellow kata drills (round-robin over the eight kata) + 40 from generated guidance
records, plus one paired record per admitted sparring record (built by `sparring.py` with `render_four_field`).

The assistant target is the four fields rendered as four lines. `why_this_move` is the transmission: it names the
voice and the technique and what the move did, so another agent can use it.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import re
from pathlib import Path
from typing import Mapping, Sequence

from . import canon
from .acp_worker import Teacher
from .checks import Check, extract_json, jaccard, non_empty
from .records import Draft, load_jsonl, seed_id
from .tasking import Assessment, Context, Outcome, Task, run_task

KIND = "lineage"
GENERATOR = "program.generate.lineage@1"
FOUR_FIELDS = ("situation", "move_chosen", "why_this_move", "observed_effect_on_person")
FAITHFUL_MIN_JACCARD = 0.5
DRILL_QUOTA, GUIDANCE_QUOTA = 60, 40


def one_line(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def render_four_field(fields: Mapping[str, str], extra: Sequence[str] = ()) -> str:
    """`key: value` lines in the manual's order; `extra` names further keys (for example `shadow_avoided`) to append."""
    return "\n".join(f"{key}: {one_line(fields[key])}" for key in (*FOUR_FIELDS, *extra))


def four_field_checks(fields: Mapping[str, object], voice: str | None, kata: str | None) -> list[Check]:
    missing = [k for k in FOUR_FIELDS if not non_empty(fields.get(k))]
    checks = [Check("four_fields_present", not missing, f"missing/empty: {missing}" if missing else "")]
    checks.append(Check("voice_canonical", voice in canon.VOICE_IDS, f"{voice!r}"))
    if kata is not None:
        checks.append(Check("kata_canonical", kata in canon.KATA_IDS, f"{kata!r}"))
    return checks


def names(text: str, candidates: Sequence[str]) -> bool:
    low = (text or "").lower()
    return any(c.lower() in low for c in candidates if c)


def voice_names(voice_id: str) -> list[str]:
    v = canon.VOICE_BY_ID[voice_id]
    return [v.id, v.english, v.egyptian]


def kata_names(kata_id: str) -> list[str]:
    k = canon.KATA_BY_ID[kata_id]
    return [k.id, k.name]


@dataclasses.dataclass(frozen=True)
class LineageSeed:
    key: str
    origin: str  # kata-drill | guidance
    seed_record_id: str
    voice: str
    kata: str | None
    language: str
    situation: str
    move_chosen: str
    component_id: str
    reference: str  # the seed's own commentary or finding, shown to the teacher as raw material


def drill_seeds(path: Path, quota: int = DRILL_QUOTA) -> list[LineageSeed]:
    by_kata: dict[str, list[dict]] = collections.defaultdict(list)
    for raw in load_jsonl(path):
        if raw["belt"] == "yellow" and raw["kata"] in canon.KATA_IDS and canon.canonical_voice(raw["voice"]):
            by_kata[raw["kata"]].append(raw)
    for rows in by_kata.values():
        rows.sort(key=seed_id)
    picked: list[dict] = []
    for round_ in range(max((len(r) for r in by_kata.values()), default=0)):
        for kata in canon.KATAS:
            rows = by_kata.get(kata.id, [])
            if round_ < len(rows) and len(picked) < quota:
                picked.append(rows[round_])
    out = []
    for raw in picked:
        sid = seed_id(raw)
        out.append(LineageSeed(f"drill:{sid}", "kata-drill", sid, canon.canonical_voice(raw["voice"]), raw["kata"], raw["lang"],
                               raw["situation"], raw["move_chosen"], sid,
                               f"why_this_move: {raw['why_this_move']}\nobserved_effect_on_person: {raw['observed_effect_on_person']}"))
    return out


def guidance_seeds(guidance_path: Path, quota: int = GUIDANCE_QUOTA) -> list[LineageSeed]:
    if not guidance_path.is_file():
        return []
    rows = sorted(load_jsonl(guidance_path), key=lambda r: r["record_id"])
    out = []
    for row in rows[:quota]:
        user, assistant = row["messages"][1]["content"], row["messages"][2]["content"]
        out.append(LineageSeed(f"guidance:{row['record_id']}", "guidance", row["record_id"], row["voice"], None, row["language"],
                               user, assistant, row["component_id"], "(no commentary: name what the guidance did and why)"))
    return out


def build_prompt(seed: LineageSeed) -> str:
    voice = canon.VOICE_BY_ID[seed.voice]
    technique = (f"Kata: {canon.KATA_BY_ID[seed.kata].name} ({seed.kata}). Form: {canon.KATA_BY_ID[seed.kata].form}."
                 if seed.kata else "Technique: guidance of an implementer, PM or PO grounded in cited man pages and source.")
    shape = ('{"voice": "%s", "kata": %s, "language": "%s", "situation": "...", "move_chosen": "...", '
             '"why_this_move": "...", "observed_effect_on_person": "..."}') % (
        seed.voice, json.dumps(seed.kata), seed.language)
    return (
        "You are the dojo teacher transmitting a technique to a small digital lifeform. You write directly; do not use any tools.\n\n"
        f"Voice: {voice.english} ({voice.id}), Egyptian name {voice.egyptian} -- {voice.does}.\n{technique}\n\n"
        f"Here is a move that was made.\nsituation: {seed.situation}\nmove_chosen: {seed.move_chosen}\n"
        f"Raw material from the move's author:\n{seed.reference}\n\n"
        f"Write the four-field transmission in {canon.LANGUAGE_NAMES.get(seed.language, seed.language)} ({seed.language}): "
        "`situation` and `move_chosen` restate the above compactly and faithfully (keep the original wording where you can); "
        f"`why_this_move` NAMES the voice ({voice.english}) and the technique and says why this move and not another; "
        "`observed_effect_on_person` says what the move did for the person (or for the recipient). Short: a transmission another "
        "agent can reuse, not a dissection.\n\n"
        f"Finish with exactly one fenced ```json block of this shape:\n{shape}\n"
    )


def assess(seed: LineageSeed, ctx: Context, reply: str, prompt_sha: str) -> Assessment:
    header = extract_json(reply)
    checks: list[Check] = [Check("json_header", header is not None, "" if header else "no parseable fenced JSON object")]
    if header is None:
        return Assessment(checks)
    fields = {k: header.get(k) for k in FOUR_FIELDS}
    checks += four_field_checks(fields, canon.canonical_voice(str(header.get("voice"))), header.get("kata") if seed.kata else None)
    checks.append(Check("voice_echo", canon.canonical_voice(str(header.get("voice"))) == seed.voice, f"{header.get('voice')!r}"))
    if seed.kata:
        checks.append(Check("kata_echo", header.get("kata") == seed.kata, f"{header.get('kata')!r}"))
    checks.append(Check("language_echo", header.get("language") == seed.language, f"{header.get('language')!r}"))
    if all(non_empty(fields[k]) for k in FOUR_FIELDS):
        checks.append(Check("names_voice", names(fields["why_this_move"], voice_names(seed.voice)), "why_this_move names the voice"))
        if seed.kata:
            checks.append(Check("names_technique", names(fields["why_this_move"], kata_names(seed.kata)), "why_this_move names the kata"))
        checks.append(ctx.languages.matches(fields["why_this_move"], seed.language))
        if seed.origin == "kata-drill":
            j = jaccard(fields["move_chosen"], seed.move_chosen)
            checks.append(Check("move_faithful", j >= FAITHFUL_MIN_JACCARD, f"jaccard {j:.2f} vs seed move"))

    def make(transcript_sha: str) -> Draft:
        user = f"situation: {one_line(seed.situation)}\nmove_chosen: {one_line(seed.move_chosen)}"
        text = render_four_field({k: str(fields[k] or "") for k in FOUR_FIELDS})
        families = ["lineage", "naming-the-move"] + ([f"kata:{seed.kata}"] if seed.kata else ["illumos-guidance"])
        return Draft(source="lineage", generator=GENERATOR, stage=canon.STAGE_BLACK, families=families, voice=seed.voice,
                     language=seed.language,
                     messages=[{"role": "system", "content": canon.system_prompt(seed.voice)},
                               {"role": "user", "content": user}, {"role": "assistant", "content": text}],
                     component_id=seed.component_id, seed_ref={"seed_record_id": seed.seed_record_id},
                     prompt_sha256=prompt_sha, transcript_sha256=transcript_sha, kata=seed.kata,
                     provenance_extra={"seed_origin": seed.origin})

    return Assessment(checks, make)


def plan(ctx: Context, drills_path: Path, guidance_path: Path, origins: Sequence[str] = ("kata-drill", "guidance")) -> list[Task]:
    done = ctx.run.finished(KIND, ctx.retry_excluded)
    seeds: list[LineageSeed] = []
    if "kata-drill" in origins:
        seeds += drill_seeds(drills_path)
    if "guidance" in origins:
        seeds += guidance_seeds(guidance_path)
    tasks = []
    for seed in seeds:
        if seed.key in done:
            continue

        def run(teacher: Teacher, seed: LineageSeed = seed) -> Outcome:
            return run_task(ctx, teacher, KIND, seed.key, lambda: build_prompt(seed), lambda r, p: assess(seed, ctx, r, p))

        tasks.append(Task(KIND, seed.key, f"lineage {seed.origin} {seed.kata or ''} {seed.language}", run))
    return tasks
