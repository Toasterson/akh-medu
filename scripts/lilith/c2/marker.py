"""The in-chat correction convention (language-neutral by construction).

A message whose first non-space character is `CORRECTION_MARKER` means "your
previous answer was wrong; this is the correction". Optionally the marker is
followed by the tool she should have used, in backticks:

    ✎ `mcp__anima__list_work_items` you skipped the parked items
    ✎ the commit is not on main yet

Nothing here inspects words: the marker is a symbol, the tool name is whatever
sits between the first pair of backticks right after it, and everything else is
the operator's free text in any language.
"""
from __future__ import annotations

import dataclasses
import re

CORRECTION_MARKER = "✎"  # LOWER RIGHT PENCIL
_EXPECTED_TOOL = re.compile(r"`([^`\s]+)`")


@dataclasses.dataclass(frozen=True)
class Correction:
    expected_tool: str | None
    text: str


def parse_correction(message: str) -> Correction | None:
    """The correction carried by `message`, or None when it is an ordinary message."""
    stripped = message.lstrip()
    if not stripped.startswith(CORRECTION_MARKER):
        return None
    rest = stripped[len(CORRECTION_MARKER):].lstrip()
    expected = _EXPECTED_TOOL.match(rest)
    if expected:
        return Correction(expected.group(1), rest[expected.end():].strip())
    return Correction(None, rest.strip())
