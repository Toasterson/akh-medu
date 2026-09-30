"""Client-injected preamble: the T3 app appends tag blocks to every operator message."""
import json

import pytest

from c2 import harvest_sessions as hs
from support import row

RUNTIME_INFO = "<runtime_info>In case you're asked: you are running in T3 Code through the Grok harness.</runtime_info>"
PR_LINKING = "<pull_request_linking>\nWhen the t3-code MCP server exposes link_pull_request, use it.\n</pull_request_linking>"


def harvest_one(tmp_path, rows):
    cfg = hs.HarvestConfig(tmp_path / "out", "127.0.0.1:18444", "x" * 30)
    hs.harvest(cfg, lambda *a: iter(rows))
    return json.loads(next((tmp_path / "out" / "sessions").glob("*.json")).read_text())


@pytest.mark.parametrize("prompt,expected", [
    (f"Hello lilith {RUNTIME_INFO}", "Hello lilith"),
    (f"Yes handle and upark 4 items {RUNTIME_INFO}\n\n{PR_LINKING}", "Yes handle and upark 4 items"),
    ("No preamble at all", "No preamble at all"),
    # any lowercase snake_case tag, not just the two T3 names
    ("Fix it <some_new_block>anything</some_new_block>", "Fix it"),
    # the operator's own tags in the middle of the message stay
    ("Compare <runtime_info>x</runtime_info> with the old output, then stop",
     "Compare <runtime_info>x</runtime_info> with the old output, then stop"),
    # not well formed, not snake_case, or glued to the text: untouched
    ("Fix it <runtime_info>never closed", "Fix it <runtime_info>never closed"),
    ("Fix it <RuntimeInfo>x</RuntimeInfo>", "Fix it <RuntimeInfo>x</RuntimeInfo>"),
    ("a<runtime_info>x</runtime_info>", "a<runtime_info>x</runtime_info>"),
    # a message that is only preamble has no operator text
    (RUNTIME_INFO, ""),
    ("", ""),
])
def test_operator_text_drops_only_the_trailing_tag_blocks(prompt, expected):
    assert hs.operator_text(prompt) == expected


def test_sessions_keep_the_raw_prompt_and_add_operator_text(tmp_path):
    prompt = f"What is parked? {RUNTIME_INFO}\n\n{PR_LINKING}"
    session = harvest_one(tmp_path, [row(1, 0, prompt, "nothing")])
    assert session["schema"] == "lilith/c2-session@1"
    turn = session["turns"][0]
    assert turn["user_prompt"] == prompt and turn["operator_text"] == "What is parked?"


def test_the_correction_marker_reads_operator_text_not_the_preamble(tmp_path):
    rows = [row(1, 0, "q", "a"),
            row(2, 1, f"✎ `mcp__anima__list_projects` you skipped the project {RUNTIME_INFO}\n\n{PR_LINKING}", "ok")]
    session = harvest_one(tmp_path, rows)
    assert session["turns"][1]["correction"] == {"expected_tool": "mcp__anima__list_projects",
                                                 "text": "you skipped the project", "corrects_turn": 0}


def test_a_marker_inside_the_client_block_is_not_a_correction(tmp_path):
    session = harvest_one(tmp_path, [row(1, 0, "q", "a"), row(2, 1, "a normal question <runtime_info>✎ x</runtime_info>", "ok")])
    assert session["turns"][1]["correction"] is None
