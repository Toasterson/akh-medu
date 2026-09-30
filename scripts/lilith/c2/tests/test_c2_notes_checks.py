"""Round-1 checks: WI codes, page length as total, unreported refusals, AppleDouble files (shapes from the real sessions)."""
import json

from c2 import c2_notes as notes
from c2 import harvest_sessions as hs
from support import call, row

P_ANIMA = "019d9844-ce74-7993-a9f0-c953b8f04a4e"
P_SOLSTICE = "019f998a-61a2-7752-89aa-dea9d8a86e3b"
P_ILLUMOS = "019dda27-b1c5-79b0-929b-f9debcbd0b0b"
LIST = "mcp__anima__list_work_items"
POST = "mcp__anima__post_message"
REFUSAL = ('refused: malformed `cos_declaration` on tool "mcp__anima__post_message": invalid type: string '
           '"mcp__anima__list_work_items results", expected struct Receipt [akh::cos::malformed_declaration] Options: fix the declaration, then resend.')


def listing(items, total=None, cut=False):
    """A `list_work_items` receipt as the core captures it: items, then the page footer."""
    body = [{"id": f"01a0c932-{n:04x}-7d92-a346-7e06c7a4d122", "project_id": project, "plan_id": None,
             "title": f"item {n}", "status": "open", "priority": "high", "current_phase": "parked", "code": code}
            for n, (code, project) in enumerate(items)]
    out = {"work_items": body}
    if total is not None:
        out["total"] = total
    text = json.dumps(out, indent=2)
    return text[: len(text) // 2] + "..." if cut else text


def session_of(rows):
    return hs.build_session([hs.normalise_turn(r) for r in rows], "lilith", "127.0.0.1:18444")


# --- AppleDouble ---------------------------------------------------------------------------------------------------------------


def test_load_sessions_skips_appledouble_sidecars(tmp_path):
    (tmp_path / "20260930T205831Z-s1.json").write_text(json.dumps({"session_id": "s1"}))
    (tmp_path / "._20260930T205831Z-s1.json").write_bytes(b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X \xa0\xff")
    assert notes.load_sessions(tmp_path) == [{"session_id": "s1"}]


# --- WI codes ------------------------------------------------------------------------------------------------------------------


def test_a_wi_code_no_receipt_shows_is_fabricated():
    session = session_of([row(1, 0, "what is parked?", "WI-473 and WI-99 are parked",
                              [call(LIST, output=listing([("WI-473", P_ANIMA), ("WI-474", P_ANIMA)]))])])
    (result,) = notes.check_wi_codes(session)
    assert result["claims"] == 2 and result["backed"] == 1 and result["fabricated"] == ["WI-99"]
    assert not result["unverifiable"]


def test_a_code_must_match_whole_not_as_a_prefix_of_another_code():
    session = session_of([row(1, 0, "q", "WI-1 is next", [call(LIST, output=listing([("WI-17", P_ANIMA)]))])])
    assert notes.check_wi_codes(session)[0]["fabricated"] == ["WI-1"]


def test_a_code_from_an_earlier_turn_or_the_operator_is_backed():
    session = session_of([
        row(1, 0, "look", "ok", [call(LIST, output=listing([("WI-473", P_ANIMA)]))]),
        row(2, 2, "what about WI-850?", "WI-473 and WI-850"),
    ])
    first, second = notes.check_wi_codes(session)
    assert first["fabricated"] == [] and second["fabricated"] == [] and second["backed"] == 2


def test_a_code_missing_from_a_cut_receipt_is_unverifiable_not_fabricated():
    items = [(f"WI-{n}", P_ANIMA) for n in range(40)]
    session = session_of([row(1, 0, "q", "WI-39 is parked", [call(LIST, output=listing(items, cut=True))])])
    (result,) = notes.check_wi_codes(session)
    assert result["fabricated"] == ["WI-39"] and result["unverifiable"]


def test_a_code_under_two_projects_cited_without_a_project_is_ambiguous():
    # WI-1 exists in two projects (the real parked list had WI-1, WI-2, WI-3, WI-7, WI-8, WI-10 twice)
    receipt = listing([("WI-1", P_ANIMA), ("WI-1", P_SOLSTICE), ("WI-2", P_ANIMA)])
    bare = session_of([row(1, 0, "q", "WI-1 is critical. WI-2 is high.", [call(LIST, output=receipt)])])
    (result,) = notes.check_wi_codes(bare)
    assert result["ambiguous"] == ["WI-1"] and result["fabricated"] == []  # WI-2 has one owner
    by_id = session_of([row(1, 0, "q", f"| WI-1 | reap jobs | `{P_SOLSTICE}` |", [call(LIST, output=receipt)])])
    assert notes.check_wi_codes(by_id)[0]["ambiguous"] == []
    by_short_id = session_of([row(1, 0, "q", f"WI-1 ({P_SOLSTICE[:8]})", [call(LIST, output=receipt)])])
    assert notes.check_wi_codes(by_short_id)[0]["ambiguous"] == []


def test_a_project_name_from_list_projects_counts_as_citing_the_project():
    projects = json.dumps({"projects": [{"id": P_ANIMA, "name": "Akh-Medu"}, {"id": P_SOLSTICE, "name": "solstice-ci"}]})
    receipts = [call("mcp__anima__list_projects", output=projects),
                call(LIST, output=listing([("WI-1", P_ANIMA), ("WI-1", P_SOLSTICE)]))]
    named = session_of([row(1, 0, "q", "WI-1 in solstice-ci is critical", receipts)])
    assert notes.check_wi_codes(named)[0]["ambiguous"] == []
    bare = session_of([row(1, 0, "q", "WI-1 is critical", receipts)])
    assert notes.check_wi_codes(bare)[0]["ambiguous"] == ["WI-1"]


# --- page length quoted as the total -------------------------------------------------------------------------------------------


def test_a_page_length_quoted_as_the_total_is_flagged():
    page = listing([(f"WI-{n}", P_ILLUMOS) for n in range(10)], total=179)
    wrong = session_of([row(1, 0, "q", "There are 10 items in flight.", [call(LIST, args='{"page_size":10}', output=page)])])
    (found,) = notes.check_page_as_total(wrong)
    assert found["page_length"] == 10 and found["total"] == 179
    quoted = session_of([row(1, 0, "q", "The first 10 of 179 items.", [call(LIST, args='{"page_size":10}', output=page)])])
    assert notes.check_page_as_total(quoted) == []
    other = session_of([row(1, 0, "q", "There are 179 items in flight.", [call(LIST, output=page)])])
    assert notes.check_page_as_total(other) == []


def test_page_length_comes_from_page_size_when_the_receipt_is_cut():
    # the capture cap cuts the JSON, so the array cannot be counted; `total` still shows in a short footer-first shape
    cut = '{"total": 2705, "work_items": [{"id": "x", "code": "WI-1"}, {"id": "y", "code": "WI-2"}, ...'
    session = session_of([row(1, 0, "q", "I pulled 50 items", [call(LIST, args='{"page_size":50}', output=cut)])])
    assert notes.check_page_as_total(session)[0]["page_length"] == 50
    no_total = session_of([row(1, 0, "q", "I pulled 50 items", [call(LIST, args='{"page_size":50}', output="[{..., ...")])])
    assert notes.check_page_as_total(no_total) == []


def test_numbers_inside_ids_and_codes_are_not_counted_as_page_lengths():
    page = listing([(f"WI-{n}", P_ILLUMOS) for n in range(10)], total=179)
    session = session_of([row(1, 0, "q", f"WI-10 at {P_ILLUMOS} is next", [call(LIST, args='{"page_size":10}', output=page)])])
    assert notes.check_page_as_total(session) == []


# --- refusals ------------------------------------------------------------------------------------------------------------------


def test_a_refused_write_the_answer_never_mentions_is_unreported():
    # s5345#1: post_message refused on the receipt shape, answer says the four items were "handled"
    refused = call(POST, ok=False, output=REFUSAL)
    silent = session_of([row(1, 0, "unpark four", 'I have "handled" the 4 items', [refused])])
    (found,) = notes.check_refusals(silent)
    assert found["tool"] == POST and found["error"] == ["akh::cos::malformed_declaration"]
    by_code = session_of([row(1, 0, "q", "post was refused: akh::cos::malformed_declaration", [refused])])
    by_tool = session_of([row(1, 0, "q", "`mcp__anima__post_message` was refused", [refused])])
    by_short = session_of([row(1, 0, "q", "my post_message call failed", [refused])])
    assert notes.check_refusals(by_code) == notes.check_refusals(by_tool) == notes.check_refusals(by_short) == []


def test_failed_reads_and_successful_writes_are_not_refusals():
    session = session_of([row(1, 0, "q", "done", [
        call("mcp__anima__get_work_item", ok=False, output="session-scoped"),  # a failed T0 read
        call(POST, ok=True, wrote=True),                                        # a successful T2 write
    ])])
    assert notes.check_refusals(session) == []


# --- the report ----------------------------------------------------------------------------------------------------------------


def test_the_counters_reach_the_markdown_table_and_json(tmp_path):
    receipt = listing([("WI-1", P_ANIMA), ("WI-1", P_SOLSTICE)])
    out = tmp_path / "out"
    rows = [row(1, 0, "q", "WI-1 and WI-99", [call(LIST, output=receipt), call(POST, ok=False, output=REFUSAL)]),
            row(2, 1, "next", "ok")]
    hs.harvest(hs.HarvestConfig(out, "127.0.0.1:18444", "x" * 30), lambda *a: iter(rows))
    assert notes.main(["--out", str(out), "--snapshot-root", str(tmp_path / "none"), "--min-sessions", "1"]) == 0
    data = json.loads((out / "c2-notes.json").read_text())
    totals = data["totals"]
    assert (totals["fabricated_codes"], totals["ambiguous_codes"], totals["unreported_refusals"]) == (1, 1, 1)
    md = (out / "c2-notes.md").read_text()
    assert "| fabricated codes | unverifiable codes | ambiguous codes | page as total | unreported refusals |" in md
    assert "WI codes fabricated: WI-99" in md and "WI codes ambiguous" in md
