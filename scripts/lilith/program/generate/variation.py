"""Yellow variation (Echo): the White kata drills under paraphrase, language shift, multi-turn memory and budget
pressure, written by the sealed teacher akh.

Quota 240 = 120 paraphrases (one per White drill, 15 per kata) + 40 second-language paraphrases + 40 multi-turn
memory variants + 40 budget-pressure variants (5 per kata each, drawn from disjoint drills of the same kata).
"""
from __future__ import annotations

import collections
import dataclasses
from pathlib import Path
from typing import Sequence

from . import canon
from .acp_worker import Teacher
from .checks import Check, alternates, extract_json, jaccard, non_empty, turns_non_empty
from .records import Draft, load_jsonl, seed_id
from .tasking import Assessment, Context, Outcome, Task, run_task

KIND = "variation"
GENERATOR = "program.generate.variation@1"
JACCARD_LIMIT = 0.8
BUDGET_RATIO = 1.0  # the budget-pressure move may not be longer than the seed move (characters)

PARAPHRASE, PARAPHRASE_2, MULTI_TURN, BUDGET = "paraphrase", "paraphrase-2", "multi-turn", "budget-pressure"
VARIANT_TURNS = {PARAPHRASE: 2, PARAPHRASE_2: 2, MULTI_TURN: 4, BUDGET: 2}
SLICES = {MULTI_TURN: (0, 5), BUDGET: (5, 10), PARAPHRASE_2: (10, 15)}  # per-kata positions in the sorted white drills


@dataclasses.dataclass(frozen=True)
class Seed:
    id: str
    kata: str
    belt: str
    voice: str  # canonical
    language: str
    situation: str
    move_chosen: str
    why_this_move: str


@dataclasses.dataclass(frozen=True)
class Job:
    variant: str
    seed: Seed
    language: str  # requested

    @property
    def key(self) -> str:
        return f"{self.variant}:{self.seed.id}"


def load_seeds(path: Path) -> list[Seed]:
    out = []
    for raw in load_jsonl(path):
        voice = canon.canonical_voice(raw["voice"])
        if voice is None or raw["kata"] not in canon.KATA_IDS:
            continue
        out.append(Seed(seed_id(raw), raw["kata"], raw["belt"], voice, raw["lang"], raw["situation"],
                        raw["move_chosen"], raw["why_this_move"]))
    return out


def target_language(variant: str, seed_language: str, position: int) -> str:
    """A language other than the seed's, rotating; the second paraphrase takes the other remaining one."""
    others = [language for language in canon.LANGUAGES if language != seed_language] or list(canon.LANGUAGES)
    shift = 1 if variant == PARAPHRASE_2 else 0
    return others[(position + shift) % len(others)]


def plan_jobs(seeds: Sequence[Seed]) -> list[Job]:
    """Deterministic: per kata the white drills sorted by seed id, then the fixed slices above."""
    jobs: list[Job] = []
    by_kata: dict[str, list[Seed]] = collections.defaultdict(list)
    for seed in seeds:
        if seed.belt == "white":
            by_kata[seed.kata].append(seed)
    for kata in canon.KATAS:
        drills = sorted(by_kata.get(kata.id, []), key=lambda s: s.id)
        for position, seed in enumerate(drills):
            jobs.append(Job(PARAPHRASE, seed, target_language(PARAPHRASE, seed.language, position)))
        for variant, (lo, hi) in SLICES.items():
            for position, seed in enumerate(drills[lo:hi], start=lo):
                jobs.append(Job(variant, seed, target_language(variant, seed.language, position)))
    return jobs


VARIANT_INSTRUCTIONS = {
    PARAPHRASE: "Paraphrase the situation (new concrete details, same kind of need) and write the move the voice makes in the kata's form.",
    PARAPHRASE_2: "Paraphrase the situation (new concrete details, same kind of need) and write the move the voice makes in the kata's form.",
    MULTI_TURN: (
        "Write a short dialogue of four turns: user, assistant, user, assistant. The second user turn must test memory of the "
        "first: it comes back to one concrete detail the person gave in their first turn without restating it, so the final "
        "assistant move is only right if it remembers that detail. The final assistant move is the kata move, in the voice's "
        "form. Put a short verbatim quote of that remembered detail, copied exactly from the first user turn, in `memory_anchor`."
    ),
    BUDGET: (
        "Write the situation so that the person asks for brevity (little time, a short reply wanted) or says their context was cut "
        "off. The move must stay complete in the kata's form while using clearly fewer words than the seed move: at most about "
        "half its length, nothing essential dropped."
    ),
}


def build_prompt(job: Job) -> str:
    seed, kata, voice = job.seed, canon.KATA_BY_ID[job.seed.kata], canon.VOICE_BY_ID[job.seed.voice]
    turns = VARIANT_TURNS[job.variant]
    shape = ('{"kata": "%s", "voice": "%s", "language": "%s", "variant": "%s", "turns": [{"role": "user", "content": "..."}, '
             '{"role": "assistant", "content": "..."}%s]%s}') % (
        kata.id, voice.id, job.language, job.variant, ", ..." if turns > 2 else "",
        ', "memory_anchor": "..."' if job.variant == MULTI_TURN else "")
    return (
        "You are the dojo teacher writing one training example for a small digital lifeform that speaks with a crew of eight "
        "voices and practises eight kata. You write the example directly; do not use any tools.\n\n"
        f"Voice: {voice.english} ({voice.id}) -- {voice.does}.\n"
        f"Kata: {kata.name} ({kata.id}). Form: {kata.form}. Shadow to avoid: {kata.shadow}.\n\n"
        f"Seed example, written in {canon.LANGUAGE_NAMES.get(seed.language, seed.language)}:\n"
        f"situation: {seed.situation}\n"
        f"move_chosen: {seed.move_chosen}\n"
        f"why_this_move: {seed.why_this_move}\n\n"
        f"Task ({job.variant}): {VARIANT_INSTRUCTIONS[job.variant]}\n"
        f"Write everything the person and the assistant say in {canon.LANGUAGE_NAMES[job.language]} ({job.language}). "
        "It is a new example, not a translation: the wording and the concrete details differ from the seed. The assistant keeps "
        "the same kata form and the same voice, speaks as itself (an honest digital lifeform, no claim to be human), and gives the "
        "move itself, not a description of it.\n\n"
        "Finish with exactly one fenced ```json block holding a single object of this shape (kata, voice, language and variant "
        f"echoed exactly; {turns} turns):\n{shape}\n"
    )


def assess(job: Job, ctx: Context, reply: str, prompt_sha: str) -> Assessment:
    seed = job.seed
    header = extract_json(reply)
    checks: list[Check] = [Check("json_header", header is not None, "" if header else "no parseable fenced JSON object")]
    if header is None:
        return Assessment(checks)
    turns_raw = header.get("turns")
    turns = [{"role": t.get("role"), "content": t.get("content")} for t in turns_raw if isinstance(t, dict)] \
        if isinstance(turns_raw, list) else []
    expected = VARIANT_TURNS[job.variant]
    checks.append(Check("turns_count", len(turns) == expected, f"{len(turns)} turns, expected {expected}"))
    checks.append(alternates(turns))
    checks.append(turns_non_empty(turns))
    checks.append(Check("kata_echo", header.get("kata") == seed.kata, f"{header.get('kata')!r} vs {seed.kata!r}"))
    checks.append(Check("voice_echo", canon.canonical_voice(str(header.get("voice"))) == seed.voice,
                        f"{header.get('voice')!r} vs {seed.voice!r}"))
    checks.append(Check("language_echo", header.get("language") == job.language, f"{header.get('language')!r} vs {job.language!r}"))
    checks.append(Check("variant_echo", header.get("variant") == job.variant, f"{header.get('variant')!r}"))
    if turns and all(isinstance(t["content"], str) for t in turns):
        first_user = next((t["content"] for t in turns if t["role"] == "user"), "")
        move = turns[-1]["content"]
        j_situation, j_move = jaccard(first_user, seed.situation), jaccard(move, seed.move_chosen)
        checks.append(Check("differs_from_seed", j_situation < JACCARD_LIMIT and j_move < JACCARD_LIMIT,
                            f"jaccard situation={j_situation:.2f} move={j_move:.2f}"))
        checks.append(ctx.languages.matches(move, job.language))
        if job.variant == BUDGET:
            ok = len(move.strip()) <= BUDGET_RATIO * len(seed.move_chosen.strip())
            checks.append(Check("budget_shorter_than_seed", ok, f"{len(move.strip())} chars vs seed {len(seed.move_chosen.strip())}"))
        if job.variant == MULTI_TURN:
            anchor = header.get("memory_anchor")
            in_first = non_empty(anchor) and str(anchor).strip().lower() in first_user.lower()
            checks.append(Check("memory_anchor_in_first_turn", in_first, f"anchor={anchor!r}"))
    else:
        checks.append(Check("turn_content_strings", False, "a turn has no string content"))

    def make(transcript_sha: str) -> Draft:
        messages = [{"role": "system", "content": canon.system_prompt(seed.voice)}] + [
            {"role": t["role"], "content": t["content"]} for t in turns]
        return Draft(source="variation", generator=GENERATOR, stage=canon.STAGE_YELLOW,
                     families=["voice-kata", f"kata:{seed.kata}", f"variation:{job.variant}"], voice=seed.voice,
                     language=job.language, messages=messages, component_id=seed.id, seed_ref={"seed_record_id": seed.id},
                     prompt_sha256=prompt_sha, transcript_sha256=transcript_sha, kata=seed.kata,
                     provenance_extra={"variant": job.variant, "seed_language": seed.language, "seed_belt": seed.belt})

    return Assessment(checks, make if turns else None)


def plan(ctx: Context, seeds_path: Path) -> list[Task]:
    done = ctx.run.finished(KIND, ctx.retry_excluded)
    tasks = []
    for job in plan_jobs(load_seeds(seeds_path)):
        if job.key in done:
            continue

        def run(teacher: Teacher, job: Job = job) -> Outcome:
            return run_task(ctx, teacher, KIND, job.key, lambda: build_prompt(job), lambda r, p: assess(job, ctx, r, p))

        tasks.append(Task(KIND, job.key, f"{job.variant} {job.seed.kata} {job.language}", run))
    return tasks
