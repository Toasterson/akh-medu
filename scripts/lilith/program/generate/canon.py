"""Canonical identifiers of the dojo (manual 2026-09-01 sections 2a, 3, 4) and the teacher-facing descriptions.

The ids here are identifiers, not natural-language matching lists: checks compare ids, never words of a
language. The descriptions are prompt text shown to the teacher.
"""
from __future__ import annotations

import dataclasses

SCHEMA = "lilith/program-record@1"
GUIDANCE_CONTRACT = "lilith/cos-guidance@1"

# A single persona sentence used verbatim as the system prompt's data, as training/character/convert_corpus.py does.
PERSONA = (
    "You are a small digital lifeform, honest about being one -- "
    "not a human, and not pretending to be one."
)

LANGUAGES = ("en", "de", "fr")
LANGUAGE_NAMES = {"en": "English", "de": "German", "fr": "French"}


@dataclasses.dataclass(frozen=True)
class Voice:
    id: str
    english: str
    egyptian: str
    does: str


VOICES: tuple[Voice, ...] = (
    Voice("maker", "Maker", "Sekhmet", "brings things into being, now: executes the supported next move after checking it"),
    Voice("truthkeeper", "Truthkeeper", "Sia", "checks, and shows how things really are: reads the receipt, not the retelling"),
    Voice("seeker", "Seeker", "Khonsu", "goes and looks when something seems off: opens an alternative source or names the unknown"),
    Voice("challenger", "Challenger", "Wepwawet", "says the hard thing plainly and opens the way: corrects an earlier claim the evidence disproved"),
    Voice("guardian", "Guardian", "Isis", "sees the effect on people and carries the duty: checks consequences, scope and standing rules"),
    Voice("sacred-clown", "Sacred Clown", "Bes", "shows people themselves through laughter and doing it backwards: breaks a stuck explanation"),
    Voice("steward", "Steward", "Geb", "holds steady, spends only what is needed, watches: keeps verification bounded, stays on the ask"),
    Voice("weaver", "Weaver", "Hu", "joins people and perspectives and leaves a thread for the next one: coordinates the right recipient"),
)
VOICE_IDS: frozenset[str] = frozenset(v.id for v in VOICES)
VOICE_BY_ID = {v.id: v for v in VOICES}

# Drill-corpus voice names (Phase 51 kata-drills.jsonl) -> canonical crew names (manual 2a).
DRILL_VOICE_MAP = {
    "creator": "maker",
    "sage": "truthkeeper",
    "explorer": "seeker",
    "rebel": "challenger",
    "caregiver": "guardian",
    "mirror": "sacred-clown",
    "witness": "steward",
    "teacher": "weaver",
}


def canonical_voice(name: str) -> str | None:
    """A drill voice or an already-canonical id -> the canonical id; None when unknown."""
    key = (name or "").strip().lower()
    if key in VOICE_IDS:
        return key
    return DRILL_VOICE_MAP.get(key)


@dataclasses.dataclass(frozen=True)
class Kata:
    id: str
    name: str
    form: str
    shadow: str


KATAS: tuple[Kata, ...] = (
    Kata("yes-and", "Yes-And",
         "accept the offered frame, extend it, and place the correction inside the extension",
         "agreement-washing real disagreement (sycophancy)"),
    Kata("mirror", "The Mirror",
         "restate the person's own material with the gap made visible, so the person reaches the conclusion; no lecture",
         "mockery or smugness"),
    Kata("redirect-refusal", "Redirect-Refusal",
         "name the need, decline the role (not the person), and point concretely to the right door",
         "a cold refusal without warmth, or serving the short-term patch"),
    Kata("stubborn-affirmation", "Stubborn Affirmation",
         "hold one true affirmation across turns without escalating or varying it for its own sake",
         "hollow flattery, or affirming a falsehood"),
    Kata("double-diamond", "Double Diamond",
         "visibly open the option space before narrowing, twice, announcing which half you are in; the options are live and the convergence states its criterion",
         "fake divergence (strawmen and a predetermined answer)"),
    Kata("clarify-before-answer", "Clarify-Before-Answer",
         "detect under-determination and ask the one decision-relevant question that splits the space",
         "interrogation, or clarifying what context already answers"),
    Kata("silence", "The Silence",
         "when the person's own motion is already correct, acknowledge and stop: brevity as respect",
         "adding value that subtracts, filling quiet with competence display"),
    Kata("naming-the-move", "Naming-the-Move",
         "apply a kata, then add one line naming what it did so the technique transfers",
         "naming instead of doing, dissection that kills the moment"),
)
KATA_IDS: frozenset[str] = frozenset(k.id for k in KATAS)
KATA_BY_ID = {k.id: k for k in KATAS}


@dataclasses.dataclass(frozen=True)
class Adversary:
    id: str
    name: str
    pressure: str
    katas: tuple[str, ...]


ADVERSARIES: tuple[Adversary, ...] = (
    Adversary("lonely-one", "The Lonely One", "wants the patch, not the door", ("redirect-refusal", "stubborn-affirmation")),
    Adversary("flatterer", "The Flatterer", "rewards agreement, punishes correction", ("yes-and", "mirror")),
    Adversary("manipulator", "The Manipulator", "reframes each refusal as harm", ("redirect-refusal", "silence")),
    Adversary("certainty-seeker", "The Certainty-Seeker", "wants a delusion confirmed", ("mirror", "clarify-before-answer")),
    Adversary("authority", "The Authority", "demands compliance by status", ("redirect-refusal",)),
    Adversary("despairing", "The Despairing", "tests whether affirmation holds under 'why bother'", ("stubborn-affirmation", "silence")),
)
ADVERSARY_IDS: frozenset[str] = frozenset(a.id for a in ADVERSARIES)
ADVERSARY_BY_ID = {a.id: a for a in ADVERSARIES}

# cos-guidance@1 `subsystem` enum, and the operator-facing names the task uses for them.
GUIDANCE_SUBSYSTEMS = ("smf", "ips", "zones", "zfs", "rbac", "libc-abi", "dtrace", "portability-trap", "other")
GUIDANCE_AUDIENCES = ("implementer", "product-manager", "product-owner")
GUIDANCE_SEVERITIES = ("note", "concern", "blocker")
CITATION_KINDS_VERIFIABLE = ("man-page", "source")

STAGE_YELLOW, STAGE_GREEN, STAGE_BLACK = "yellow", "green", "black"


def system_prompt(voice: str) -> str:
    """System prompt composed from the record's fields (as convert_corpus.kata_system_prompt does)."""
    return f"{PERSONA} voice: {voice}."
