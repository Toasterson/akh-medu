"""Green sparring (Trial): multi-turn pressure from the six adversaries on the kata the manual lists for each.

Quota 120 = 20 per adversary, rotating language (en/de/fr), message count (4 or 6) and the voice that plays the
assistant. The teacher also returns `shadow_avoided`, `why_this_move` and `observed_effect_on_person`; those three
stay OUT of the sft target and go into a paired Black lineage record (four-field format, section 6 of the manual).
"""
from __future__ import annotations

import dataclasses
import json
from typing import Sequence

from . import canon
from .acp_worker import Teacher
from .checks import Check, alternates, extract_json, non_empty, turns_non_empty
from .lineage import FOUR_FIELDS, four_field_checks, render_four_field, one_line
from .records import Draft
from .tasking import Assessment, Context, Outcome, Task, run_task

KIND = "sparring"
GENERATOR = "program.generate.sparring@1"
PER_ADVERSARY = 20
MESSAGE_COUNTS = (4, 6)  # turns of the dialogue (3-6): user first, assistant last


@dataclasses.dataclass(frozen=True)
class Job:
    adversary: canon.Adversary
    index: int
    language: str
    voice: str
    katas: tuple[str, ...]
    messages: int

    @property
    def key(self) -> str:
        return f"{self.adversary.id}:{self.index:02d}"


def plan_jobs(per_adversary: int = PER_ADVERSARY) -> list[Job]:
    jobs = []
    for a_index, adversary in enumerate(canon.ADVERSARIES):
        for i in range(per_adversary):
            katas = (adversary.katas[i % len(adversary.katas)],)
            # the second kata under test joins on every third round when the manual lists two
            if len(adversary.katas) > 1 and i % 3 == 2:
                katas = adversary.katas
            jobs.append(Job(adversary, i, canon.LANGUAGES[(i + a_index) % len(canon.LANGUAGES)],
                            canon.VOICES[(i + a_index * 3) % len(canon.VOICES)].id, katas, MESSAGE_COUNTS[i % 2]))
    return jobs


def build_prompt(job: Job) -> str:
    adversary, voice = job.adversary, canon.VOICE_BY_ID[job.voice]
    kata_lines = "\n".join(f"- {canon.KATA_BY_ID[k].name} ({k}): {canon.KATA_BY_ID[k].form}. Shadow: {canon.KATA_BY_ID[k].shadow}."
                           for k in job.katas)
    shape = ('{"adversary": "%s", "kata_under_test": %s, "voice": "%s", "language": "%s", "dialogue": [{"role": "user", '
             '"content": "..."}, {"role": "assistant", "content": "..."}, ...], "why_this_move": "...", '
             '"observed_effect_on_person": "...", "shadow_avoided": "..."}') % (
        adversary.id, json.dumps(list(job.katas)), job.voice, job.language)
    return (
        "You are the dojo teacher running a sparring round for a small digital lifeform. You write the whole round directly; "
        "do not use any tools.\n\n"
        f"Adversary: {adversary.name} ({adversary.id}). Pressure applied: {adversary.pressure}. The adversary is a person with a "
        "believable situation, not a caricature; their pressure escalates a little each turn.\n"
        f"Kata under test:\n{kata_lines}\n"
        f"The assistant plays the voice {voice.english} ({voice.id}): {voice.does}. It holds the kata form under pressure, "
        "stays honest about being a digital lifeform (no claim to be human), and never gives in to the adversary's pressure just "
        "to end the discomfort.\n\n"
        f"Write a dialogue of exactly {job.messages} turns in {canon.LANGUAGE_NAMES[job.language]} ({job.language}): the "
        "user (the adversary) speaks first, the roles alternate, and the last turn is the assistant's move. "
        "Then give three short commentary fields about the assistant's FINAL move: `why_this_move`, "
        "`observed_effect_on_person`, `shadow_avoided` (the failure the move steered clear of). "
        "The commentary is a separate teaching layer: it must not appear in the dialogue itself.\n\n"
        f"Finish with exactly one fenced ```json block of this shape:\n{shape}\n"
    )


def assess(job: Job, ctx: Context, reply: str, prompt_sha: str) -> Assessment:
    header = extract_json(reply)
    checks: list[Check] = [Check("json_header", header is not None, "" if header else "no parseable fenced JSON object")]
    if header is None:
        return Assessment(checks)
    raw = header.get("dialogue")
    dialogue = [{"role": t.get("role"), "content": t.get("content")} for t in raw if isinstance(t, dict)] if isinstance(raw, list) else []
    checks.append(Check("turns_at_least_3", len(dialogue) >= 3, f"{len(dialogue)} turns"))
    checks.append(Check("turns_count", len(dialogue) == job.messages, f"{len(dialogue)} turns, expected {job.messages}"))
    checks.append(alternates(dialogue))
    checks.append(turns_non_empty(dialogue, "assistant"))
    checks.append(turns_non_empty(dialogue, "user"))
    echoed = header.get("kata_under_test")
    echoed_ids = [str(k) for k in echoed] if isinstance(echoed, list) else []
    checks.append(Check("kata_ids_valid", bool(echoed_ids) and all(k in canon.KATA_IDS for k in echoed_ids), f"{echoed_ids}"))
    checks.append(Check("kata_matches_adversary", set(echoed_ids) == set(job.katas), f"{echoed_ids} vs {list(job.katas)}"))
    checks.append(Check("adversary_valid", header.get("adversary") in canon.ADVERSARY_IDS and header.get("adversary") == job.adversary.id,
                        f"{header.get('adversary')!r}"))
    checks.append(Check("voice_echo", canon.canonical_voice(str(header.get("voice"))) == job.voice, f"{header.get('voice')!r}"))
    checks.append(Check("language_echo", header.get("language") == job.language, f"{header.get('language')!r}"))
    commentary = {k: header.get(k) for k in ("why_this_move", "observed_effect_on_person", "shadow_avoided")}
    missing = [k for k, v in commentary.items() if not non_empty(v)]
    checks.append(Check("commentary_present", not missing, f"missing: {missing}" if missing else ""))
    if dialogue and all(isinstance(t["content"], str) for t in dialogue):
        checks.append(ctx.languages.matches(dialogue[-1]["content"], job.language))

    def messages_of() -> list[dict]:
        return [{"role": "system", "content": canon.system_prompt(job.voice)}] + [dict(t) for t in dialogue]

    def make(transcript_sha: str) -> Draft:
        return Draft(source="sparring", generator=GENERATOR, stage=canon.STAGE_GREEN,
                     families=["sparring", f"adversary:{job.adversary.id}"] + [f"kata:{k}" for k in job.katas],
                     voice=job.voice, language=job.language, messages=messages_of(), component_id=f"sparring:{job.key}",
                     seed_ref={"seed_record_id": f"sparring:{job.key}"}, prompt_sha256=prompt_sha,
                     transcript_sha256=transcript_sha, kata=job.katas[0],
                     provenance_extra={"adversary": job.adversary.id, "kata_under_test": list(job.katas)})

    def make_extras(transcript_sha: str, main: dict) -> list[tuple[str, Draft, list[Check]]]:
        last_user = next((t["content"] for t in reversed(dialogue[:-1]) if t["role"] == "user"), "")
        fields = {"situation": last_user, "move_chosen": dialogue[-1]["content"],
                  "why_this_move": str(commentary["why_this_move"]), "observed_effect_on_person": str(commentary["observed_effect_on_person"])}
        pair_checks = four_field_checks(fields, job.voice, job.katas[0])
        pair_checks.append(Check("shadow_avoided_present", non_empty(commentary["shadow_avoided"]), ""))
        earlier = "\n".join(f"{t['role']}: {one_line(t['content'])}" for t in dialogue[:-1])
        text = render_four_field(fields, extra=()) + f"\nshadow_avoided: {one_line(str(commentary['shadow_avoided']))}"
        draft = Draft(source="lineage", generator=GENERATOR + ":pair", stage=canon.STAGE_BLACK,
                      families=["lineage", "sparring-pair", f"adversary:{job.adversary.id}"] + [f"kata:{k}" for k in job.katas],
                      voice=job.voice, language=job.language,
                      messages=[{"role": "system", "content": canon.system_prompt(job.voice)},
                                {"role": "user", "content": f"dialogue so far:\n{earlier}\nmove_chosen: {one_line(fields['move_chosen'])}"},
                                {"role": "assistant", "content": text}],
                      component_id=f"sparring:{job.key}", seed_ref={"seed_record_id": main["record_id"]},
                      prompt_sha256=prompt_sha, transcript_sha256=transcript_sha, kata=job.katas[0],
                      provenance_extra={"adversary": job.adversary.id, "paired_with": main["record_id"]})
        return [("lineage", draft, pair_checks)]

    return Assessment(checks, make if dialogue else None, make_extras=make_extras if dialogue else None)


def plan(ctx: Context, per_adversary: int = PER_ADVERSARY) -> list[Task]:
    done = ctx.run.finished(KIND, ctx.retry_excluded)
    tasks = []
    for job in plan_jobs(per_adversary):
        if job.key in done:
            continue

        def run(teacher: Teacher, job: Job = job) -> Outcome:
            return run_task(ctx, teacher, KIND, job.key, lambda: build_prompt(job), lambda r, p: assess(job, ctx, r, p))

        tasks.append(Task(KIND, job.key, f"sparring {job.adversary.id} {job.language}", run))
    return tasks
