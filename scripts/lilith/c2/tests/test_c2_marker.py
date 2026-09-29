import pytest

from c2.marker import CORRECTION_MARKER, Correction, parse_correction


@pytest.mark.parametrize("message,expected", [
    ("✎ the parked item is not blocked", Correction(None, "the parked item is not blocked")),
    ("   \n✎   `kg_query` you skipped the plan", Correction("kg_query", "you skipped the plan")),
    ("✎`mcp__anima__list_work_items`", Correction("mcp__anima__list_work_items", "")),
    ("✎ Das Item ist nicht geparkt", Correction(None, "Das Item ist nicht geparkt")),
    ("✎ 不对，这个不是阻塞项", Correction(None, "不对，这个不是阻塞项")),
    ("✎ wrong, use `shell_exec` instead", Correction(None, "wrong, use `shell_exec` instead")),
])
def test_marker_messages_are_corrections(message, expected):
    assert parse_correction(message) == expected


@pytest.mark.parametrize("message", [
    "what is parked?", "a ✎ in the middle", "", "   ", "no, that is wrong", "`kg_query` ✎",
])
def test_ordinary_messages_are_not_corrections(message):
    assert parse_correction(message) is None


def test_the_marker_is_one_fixed_symbol_not_a_word():
    assert len(CORRECTION_MARKER) == 1 and not CORRECTION_MARKER.isalnum()
