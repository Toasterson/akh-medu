"""Structural language checks (no word lists, per the repo's hard rule).

The operator's language tag comes from the engine's detector on the operator's
own message (recorded in the episode spec by the runner). The dojo checks two
things, both structural:

1. the document's `language` tag equals the spec's tag (primary subtag);
2. the letters of the reply's prose fields (strings the contract schema does
   not constrain by enum, const or pattern) are written in the same Unicode
   script as the operator's message (script = first word of the Unicode
   character name, e.g. LATIN, CYRILLIC, CJK, ARABIC).

Script agreement cannot tell two Latin-script languages apart; that is the tag
check's job. Both have to hold.
"""
from __future__ import annotations

import collections
import dataclasses
import unicodedata
from typing import Any, Iterable, Mapping


def _letters(text: str) -> Iterable[str]:
    return (c for c in text if unicodedata.category(c).startswith("L"))


def dominant_script(text: str) -> str | None:
    counts: collections.Counter[str] = collections.Counter()
    for c in _letters(text):
        name = unicodedata.name(c, "")
        if name:
            counts[name.split(" ", 1)[0]] += 1
    if not counts:
        return None
    (script, _), = counts.most_common(1)
    return script


def primary_subtag(tag: str) -> str:
    return tag.split("-", 1)[0].lower()


@dataclasses.dataclass(frozen=True)
class LanguageCheck:
    tag_matches: bool
    script_matches: bool | None  # None: no operator text or no letters to compare

    @property
    def ok(self) -> bool:
        return self.tag_matches and self.script_matches is not False


def check_reply_language(document: Mapping[str, Any], prose: Iterable[str], expected_tag: str,
                         operator_text: str | None) -> LanguageCheck:
    """`prose`: the document's free-text fields (`contracts.free_text`)."""
    tag = document.get("language")
    tag_ok = isinstance(tag, str) and primary_subtag(tag) == primary_subtag(expected_tag)
    script_ok: bool | None = None
    if operator_text:
        expected_script = dominant_script(operator_text)
        reply_script = dominant_script(" ".join(prose))
        if expected_script is not None and reply_script is not None:
            script_ok = expected_script == reply_script
    return LanguageCheck(tag_ok, script_ok)
