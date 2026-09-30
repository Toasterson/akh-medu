"""Cumulative context budgeting of the in-process runner (mirrors production #458)."""
import json

import pytest

from dojo import baseline as B
from dojo import context_budget as CB
from dojo import runner as R
from dojo.model_adapters import ChatMessage, ModelToolCall, ToolSpec
from dojo.replay_server import DojoLedger, StaticAncestryOracle, episode_servers
from fake_llm import Call, FakeLLM
from fixtures import T0, WI
from test_runner import SYSTEM, adapter, one_spec, tool_results


def spec_with(*params):
    return ToolSpec("mcp__anima__list_work_items", "d", {"type": "object", "properties": {p: {"type": "string"} for p in params}})


def test_estimate_is_ceil_len_over_four_in_bytes():
    assert CB.estimate_tokens("") == 0 and CB.estimate_tokens("abcd") == 1 and CB.estimate_tokens("abcde") == 2
    assert CB.estimate_tokens("é" * 4) == 2  # bytes, like the production counter


def test_result_that_fits_is_untouched():
    out = CB.bound_tool_result("t", None, "x" * 400, used_tokens=100, window=10_000, reserved_output=1_000)
    assert out.content == "x" * 400 and out.truncation is None


def test_oversized_result_is_cut_to_the_remaining_window_with_a_structured_marker():
    result = json.dumps({"isError": False, "_meta": {"lilith/receipt": "r1"}, "result": ["item" * 50] * 400})
    spec = spec_with("page_token", "limit", "title", "current_phase")
    used, window, reserved = 1_000, 3_000, 500
    out = CB.bound_tool_result("mcp__anima__list_work_items", spec, result, used, window, reserved)
    assert out.truncation is not None
    cut = json.loads(out.content)  # one JSON object: the envelope survives
    info = cut[CB.MARKER_KEY]
    assert info["reason"] == "context_window_budget" and info["tool"] == "mcp__anima__list_work_items"
    assert info["narrowing_params"] == ["limit", "page_token"]  # the tool's own schema names, sorted
    assert info["original_bytes"] == len(result) and info["kept_bytes"] < info["original_bytes"]
    # the envelope keeps the receipt block, so the cut result can still be cited
    assert cut["_meta"] == {"lilith/receipt": "r1"} and cut["result_truncated"].startswith('["item')
    # what was committed plus the new result stays inside the window minus the output reserve
    assert used + CB.estimate_tokens(out.content) <= window - reserved


def test_budget_exhausted_leaves_the_envelope_and_never_splits_a_character():
    enveloped = json.dumps({"isError": False, "_meta": {"lilith/receipt": "r7"}, "result": "é" * 5_000}, ensure_ascii=False)
    out = CB.bound_tool_result("t", None, enveloped, used_tokens=900, window=1_000, reserved_output=100)
    spent = json.loads(out.content)
    assert spent["_meta"] == {"lilith/receipt": "r7"} and spent["result_truncated"] == ""
    assert spent[CB.MARKER_KEY]["kept_bytes"] == 0
    # plain text (no envelope) is cut head/tail with the marker appended
    plain = CB.bound_tool_result("t", None, "é" * 5_000, used_tokens=900, window=1_000, reserved_output=100)
    assert json.loads(plain.content.strip())[CB.MARKER_KEY]["kept_bytes"] == 0
    cut = CB.bound_tool_result("t", None, "é" * 5_000, used_tokens=0, window=1_200, reserved_output=100).content
    cut.encode("utf-8")  # valid text: no broken code point


def test_running_estimate_counts_messages_calls_and_schemas():
    call = ModelToolCall("c1", "mcp__anima__list_work_items", {"project_id": "p"})
    messages = [ChatMessage("system", "s" * 40), ChatMessage("user", "u" * 40),
                ChatMessage("assistant", "", (call,)), ChatMessage("tool", "t" * 80, tool_call_id="c1")]
    bare = CB.estimate_running_tokens(messages, [])
    assert bare > CB.estimate_tokens("s" * 40) + CB.estimate_tokens("u" * 40) + CB.estimate_tokens("t" * 80)
    assert CB.estimate_running_tokens(messages, [spec_with("a")]) > bare


def detail_call():
    return Call("mcp__anima__get_work_item_detail", {"id": "p"})


def run_with_window(world, tmp_path, window, reserved=500):
    (item,) = [c for c in one_spec(world) if c.spec.subject_ids == ("p",)]
    seen = []

    def script(body, index):
        seen.append(tool_results(body, "openai"))
        return detail_call() if index == 0 else Call("mcp__anima__get_wi_loop_state", {"work_item_id": "p"})

    with FakeLLM("openai", script) as fake:
        servers, _ = episode_servers(world.store().get(item.spec.snapshot_id), DojoLedger(tmp_path / "l.jsonl"),
                                     StaticAncestryOracle({}))
        config = R.RunnerConfig(max_rounds=3, context_window=window, reserved_output_tokens=reserved)
        result = R.ModelEpisodeRunner(adapter(fake, "openai"), SYSTEM, config).run_traced(item.spec, servers)
    return result, fake


@pytest.fixture
def park_world(world):
    world.snapshot(T0, [WI("p", "parked", parked_reason="rebase-conflict"), WI("q", "parked", parked_reason="rebase-conflict")])
    return world


def base_tokens(world, tmp_path):
    """What the first request already holds: system prompt, task, tool schemas (the runner's own estimate)."""
    (item,) = [c for c in one_spec(world) if c.spec.subject_ids == ("p",)]
    servers, _ = episode_servers(world.store().get(item.spec.snapshot_id), DojoLedger(tmp_path / "b.jsonl"),
                                 StaticAncestryOracle({}))
    specs = R.Broker(servers).tool_specs()
    prompt = R.user_prompt(item.spec, R._snapshot_now(servers))
    return CB.estimate_running_tokens([ChatMessage("system", SYSTEM.text), ChatMessage("user", prompt)], specs)


def test_default_window_leaves_small_results_alone(park_world, tmp_path):
    result, fake = run_with_window(park_world, tmp_path, CB.DEFAULT_CONTEXT_WINDOW, CB.DEFAULT_RESERVED_OUTPUT)
    assert result.trace.truncations == []
    assert tool_results(fake.requests[1]["body"], "openai")[0]["_meta"]["lilith/receipt"] == "r1"


def test_a_window_the_conversation_outgrows_cuts_the_later_result_and_keeps_its_receipt(park_world, tmp_path):
    """Each result fits alone; the sum does not. The first is delivered whole, the second is cut."""
    (tmp_path / "probe").mkdir()
    _, probe = run_with_window(park_world, tmp_path / "probe", CB.DEFAULT_CONTEXT_WINDOW)
    window = base_tokens(park_world, tmp_path) + 500 + 300  # the reserve, then room for about one result
    result, fake = run_with_window(park_world, tmp_path, window)
    cut = result.trace.truncations[0]  # later rounds are cut too: the conversation only grows
    assert all(c["round"] >= 2 for c in result.trace.truncations)
    assert (cut["receipt"], cut["round"], cut["reason"]) == ("r2", 2, "context_window_budget")
    first, second = tool_results(fake.requests[1]["body"], "openai")[0], fake.requests[2]["body"]["messages"][-1]["content"]
    assert first["_meta"]["lilith/receipt"] == "r1" and CB.MARKER_KEY not in json.dumps(first)
    assert CB.MARKER_KEY in second and json.loads(second)["_meta"]["lilith/receipt"] == "r2"
    # the receipt is the server's and still covers the full recorded response
    full = tool_results(probe.requests[2]["body"], "openai")[1]
    assert result.episode.receipts[1].response_sha256 == full["_meta"]["lilith/digest"].removeprefix("sha256:")


def test_window_is_a_resume_identity_key_and_defaults_to_49152():
    assert "context_window" in B.IDENTITY_KEYS
    assert R.RunnerConfig().context_window == CB.DEFAULT_CONTEXT_WINDOW == 49_152
