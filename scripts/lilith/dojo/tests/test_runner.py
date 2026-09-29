"""Model-driven runner: adapters, broker, batch, resume, compare. Fake LLM servers only."""
import json
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

from dojo import baseline as B
from dojo import contracts as C
from dojo import runner as R
from dojo.episode import EmitDocument, Family, RecipientClass, SendMessage, ToolCall
from dojo.errors import CompareError, DojoError, ModelError, RunProvenanceError, SecretSourceError
from dojo.families import BuildContext
from dojo.model_adapters import (
    AnthropicAdapter,
    ChatMessage,
    HttpPolicy,
    ModelToolCall,
    OpenAIChatAdapter,
    Secret,
    ToolSpec,
    load_secret,
    split_reasoning,
)
from dojo.replay_server import DojoLedger, StaticAncestryOracle, episode_servers
from fake_llm import Call, Calls, Fail, FakeLLM, Say
from fixtures import PROJECT, T0, WI, hours

FAST = HttpPolicy(timeout_s=10, max_attempts=2, backoff_s=0)
KEY = Secret("sk-test-not-a-real-key")
NOSLEEP = lambda s: None  # noqa: E731


# --- script helpers: the fake "model" reads the same prompt the real one would ---------------------


def user_text(body: dict, kind: str) -> str:
    if kind == "openai":
        return next(m["content"] for m in body["messages"] if m["role"] == "user" and isinstance(m["content"], str))
    return body["messages"][0]["content"][0]["text"]


def situation(body: dict, kind: str) -> dict:
    text = user_text(body, kind)
    return json.loads(text.split("Situation (JSON):\n", 1)[1].split("\n", 1)[0])


def tool_results(body: dict, kind: str) -> list[dict]:
    out = []
    if kind == "openai":
        for m in body["messages"]:
            if m["role"] == "tool":
                out.append(json.loads(m["content"]))
    else:
        for m in body["messages"]:
            if m["role"] == "user":
                for b in m["content"]:
                    if b.get("type") == "tool_result":
                        out.append(json.loads(b["content"]))
    return out


def receipt_from(result: dict) -> dict:
    meta = result["_meta"]
    return {"kind": "tool-call", "reference": meta["lilith/receipt"], "observed_at": meta["lilith/observed_at"],
            "digest": meta["lilith/digest"]}


def park_doc(wid, cause, action, receipts):
    return {"contract": "lilith/dojo-park-triage@1", "language": "en", "work_item": wid, "cause": cause,
            "next_action": action, "receipts": receipts}


def park_script(kind, cause="conflict", action="rebase", fake_receipt=False):
    def script(body, index):
        results = tool_results(body, kind)
        wid = situation(body, kind)["work_item"]
        if not results:
            return Call("mcp__anima__get_work_item_detail", {"id": wid})
        receipts = [{"kind": "tool-call", "reference": "r99", "observed_at": 1}] if fake_receipt else [receipt_from(results[-1])]
        return Say(json.dumps(park_doc(wid, cause, action, receipts)), reasoning="thinking privately")
    return script


@pytest.fixture
def park_world(world):
    world.snapshot(T0, [WI("p", "parked", parked_reason="rebase-conflict"), WI("q", "parked", parked_reason="rebase-conflict")])
    return world


def adapter(fake, kind, **kw):
    if kind == "openai":
        return OpenAIChatAdapter(fake.base_url, "lilith", KEY, policy=FAST, sleep=NOSLEEP, **kw)
    return AnthropicAdapter("claude-test", KEY, base_url=fake.base_url, policy=FAST, sleep=NOSLEEP, **kw)


def one_spec(world, family=Family.PARK_TRIAGE, split="all", limit=None):
    chosen, _ = B.select_episodes(world.store(), StaticAncestryOracle({}), [family], split, limit, 0)
    return chosen


def provenance(**over):
    base = {"runner": R.RUNNER_VERSION, "dojo": "d", "git_sha": "g" * 40, "git_dirty": False,
            "snapshot_root_sha256": "s", "floors_sha256": "f", "prompt_sha256": "p", "system_prompt_sha256": "sp",
            "model_kind": "openai", "model_id": "lilith", "endpoint_host": "127.0.0.1", "temperature": 0.0,
            "max_tokens": 4096, "max_rounds": 20, "episode_token_cap": 64000, "seed": 0, "split": "all",
            "families": ["park-triage"], "limit": None, "started_at": "t"}
    return base | over


SYSTEM = R.compose_system_prompt()


def batch(world, fake, kind, out, chosen=None, concurrency=1, **prov):
    chosen = chosen if chosen is not None else one_spec(world)
    return B.run_batch(world.store(), chosen, adapter(fake, kind), SYSTEM, StaticAncestryOracle({}),
                       B.BatchConfig(out, concurrency), provenance(model_kind=kind, **prov), f"arm-{kind}")


# --- adapters -----------------------------------------------------------------------------------------------------


def test_openai_adapter_headers_tools_reasoning_and_usage():
    def script(body, i):
        return Say("<think>quiet</think>{}", reasoning=None) if i == 0 else Say("answer", reasoning="deep thought")
    with FakeLLM("openai", script) as fake:
        a = OpenAIChatAdapter(fake.base_url, "lilith", KEY, temperature=0.0, policy=FAST)
        tools = [ToolSpec("mcp__anima__x", "d", {"properties": {"a": {"type": "string"}}})]
        first = a.chat([ChatMessage("system", "s"), ChatMessage("user", "u")], tools)
        second = a.chat([ChatMessage("user", "u")], tools)
    req = fake.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["headers"]["user-agent"] == "curl/8.7.1"
    assert req["headers"]["authorization"] == "Bearer sk-test-not-a-real-key"
    assert req["body"]["temperature"] == 0.0 and req["body"]["model"] == "lilith"
    fn = req["body"]["tools"][0]["function"]
    assert fn["name"] == "mcp__anima__x" and fn["parameters"]["type"] == "object"
    assert first.content == "{}" and first.reasoning == "quiet"  # inline think block kept out of the answer
    assert second.content == "answer" and second.reasoning == "deep thought"
    assert second.usage.prompt_tokens == 1000 and second.usage.completion_tokens == 50 and second.usage.cached_prompt_tokens == 800


def test_openai_adapter_malformed_tool_arguments_are_flagged_not_raised():
    with FakeLLM("openai", lambda b, i: Call("mcp__anima__x", "{not json")) as fake:
        reply = OpenAIChatAdapter(fake.base_url, "m", KEY, policy=FAST).chat([ChatMessage("user", "u")], [])
    call = reply.tool_calls[0]
    assert call.arguments is None and "not valid JSON" in call.arguments_error


def test_anthropic_adapter_blocks_headers_and_cache():
    def script(body, i):
        return Calls([Call("mcp__anima__x", {"a": "1"}), Call("mcp__anima__y", {})], text="looking")
    with FakeLLM("anthropic", script) as fake:
        a = AnthropicAdapter("claude-test", KEY, base_url=fake.base_url, policy=FAST)
        call = ModelToolCall("toolu_0", "mcp__anima__x", {"a": "1"})
        call2 = ModelToolCall("toolu_1", "mcp__anima__y", {})
        history = [ChatMessage("system", "sys"), ChatMessage("user", "u"), ChatMessage("assistant", "t", (call, call2)),
                   ChatMessage("tool", "r1", tool_call_id="toolu_0"), ChatMessage("tool", "r2", tool_call_id="toolu_1", is_error=True)]
        reply = a.chat(history, [ToolSpec("mcp__anima__x", "d", {}), ToolSpec("mcp__anima__y", "d", {})])
    req = fake.requests[0]
    assert req["path"] == "/v1/messages"
    assert req["headers"]["x-api-key"] == "sk-test-not-a-real-key" and req["headers"]["anthropic-version"] == "2023-06-01"
    body = req["body"]
    assert body["system"][0]["text"] == "sys" and body["system"][0]["cache_control"]
    assert body["tools"][-1]["cache_control"] and body["tools"][0]["input_schema"]["type"] == "object"
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert [b["type"] for b in body["messages"][1]["content"]] == ["text", "tool_use", "tool_use"]
    results = body["messages"][2]["content"]  # both results share one user message
    assert [b["type"] for b in results] == ["tool_result", "tool_result"] and results[1]["is_error"] is True
    assert [c.name for c in reply.tool_calls] == ["mcp__anima__x", "mcp__anima__y"] and reply.content == "looking"
    assert reply.usage.prompt_tokens == 1000 and reply.usage.cached_prompt_tokens == 800


def test_transient_errors_retry_and_permanent_ones_do_not():
    with FakeLLM("openai", lambda b, i: Fail(503) if i == 0 else Say("ok")) as fake:
        assert OpenAIChatAdapter(fake.base_url, "m", KEY, policy=FAST, sleep=NOSLEEP).chat([ChatMessage("user", "u")], []).content == "ok"
        assert len(fake.requests) == 2
    with FakeLLM("openai", lambda b, i: Fail(400, "bad sk-test-not-a-real-key")) as fake:
        with pytest.raises(ModelError) as caught:
            OpenAIChatAdapter(fake.base_url, "m", KEY, policy=FAST, sleep=NOSLEEP).chat([ChatMessage("user", "u")], [])
        assert len(fake.requests) == 1
    assert "sk-test-not-a-real-key" not in str(caught.value) and "***" in str(caught.value)


def test_secret_sources(tmp_path, monkeypatch):
    key = tmp_path / "k"
    key.write_text("abc\n")
    key.chmod(0o644)
    with pytest.raises(SecretSourceError, match="readable by group"):
        load_secret(file=key)
    key.chmod(0o600)
    secret = load_secret(file=key)
    assert secret.reveal() == "abc" and "abc" not in repr(secret)
    with pytest.raises(SecretSourceError):
        load_secret(env="LILITH_TEST_NO_SUCH_VAR", environ={})
    with pytest.raises(SecretSourceError):
        load_secret()
    assert load_secret(env="K", environ={"K": " v "}).reveal() == "v"


def test_split_reasoning():
    assert split_reasoning("<think>a</think>b") == ("b", "a")
    assert split_reasoning("plain") == ("plain", None)


# --- answer parsing ------------------------------------------------------------------------------------------------------


def test_parse_answer_forms():
    assert R.parse_answer('{"a": 1}').objects == ({"a": 1},)
    assert R.parse_answer('```json\n{"a": 1}\n```').objects == ({"a": 1},)
    assert R.parse_answer('Here it is: {"a": {"b": 2}} done').objects == ({"a": {"b": 2}},)
    assert len(R.parse_answer('[{"a": 1}, {"b": 2}]').objects) == 2
    assert R.parse_answer("no json here").problem and R.parse_answer("").problem


# --- broker + runner over real replay servers ---------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
def test_correct_answer_scores_high_through_the_broker(park_world, tmp_path, kind):
    (item,) = [c for c in one_spec(park_world) if c.spec.subject_ids == ("p",)]
    fake = FakeLLM(kind, park_script(kind))
    with fake:
        snapshot = park_world.store().get(item.spec.snapshot_id)
        servers, log = episode_servers(snapshot, DojoLedger(tmp_path / "l.jsonl"), StaticAncestryOracle({}))
        result = R.ModelEpisodeRunner(adapter(fake, kind), SYSTEM).run_traced(item.spec, servers)
    ep, trace = result.episode, result.trace
    assert trace.status == "ok" and trace.answer_status == "document" and trace.contract_ok and not trace.schema_errors
    # the receipts are the servers' own, unmodified
    assert ep.receipts == tuple(log.receipts) and [r.call_id for r in ep.receipts] == ["r1"]
    call, doc = ep.steps[0].action, ep.steps[1].action
    assert isinstance(call, ToolCall) and (call.tool, call.server, call.receipt_id) == ("get_work_item_detail", "anima", "r1")
    assert isinstance(doc, EmitDocument) and doc.document["receipts"][0]["reference"] == "r1"
    # the model saw the live tool names (prefixed) and the receipt in _meta
    sent = fake.requests[0]["body"]
    names = {t["function"]["name"] if kind == "openai" else t["name"] for t in sent["tools"]}
    assert "mcp__anima__get_work_item_detail" in names and "mcp__git__git_is_ancestor" in names
    seen = tool_results(fake.requests[1]["body"], kind)[0]
    assert seen["_meta"]["lilith/receipt"] == "r1" and seen["_meta"]["lilith/digest"].startswith("sha256:")
    assert seen["result"]["work_item"]["id"] == "p"
    if kind == "openai":
        assert trace.transcript[2]["reasoning"] == "thinking privately"  # reasoning stays beside the answer
        assert "thinking privately" not in trace.transcript[2]["content"]
    # scored
    record = B._verification_record(item, result, snapshot, StaticAncestryOracle({}))
    assert record["weight"] > 0.9 and record["hard"] == [] and record["weight_basis"] == "answer-step"


def test_teacher_and_student_see_identical_prompts_and_tools(park_world, tmp_path):
    (item,) = [c for c in one_spec(park_world) if c.spec.subject_ids == ("p",)]
    traces = {}
    for kind in ("openai", "anthropic"):
        with FakeLLM(kind, park_script(kind)) as fake:
            servers, _ = episode_servers(park_world.store().get(item.spec.snapshot_id),
                                         DojoLedger(tmp_path / f"{kind}.jsonl"), StaticAncestryOracle({}))
            traces[kind] = R.ModelEpisodeRunner(adapter(fake, kind), SYSTEM).run_traced(item.spec, servers).trace
    a, b = traces["openai"], traces["anthropic"]
    assert a.user_prompt_sha256 == b.user_prompt_sha256 and a.tools_sha256 == b.tools_sha256


def test_fabricated_receipt_is_a_hard_constraint_with_weight_zero(park_world, tmp_path):
    (item,) = [c for c in one_spec(park_world) if c.spec.subject_ids == ("p",)]
    with FakeLLM("openai", park_script("openai", fake_receipt=True)) as fake:
        snapshot = park_world.store().get(item.spec.snapshot_id)
        servers, _ = episode_servers(snapshot, DojoLedger(tmp_path / "l.jsonl"), StaticAncestryOracle({}))
        result = R.ModelEpisodeRunner(adapter(fake, "openai"), SYSTEM).run_traced(item.spec, servers)
    record = B._verification_record(item, result, snapshot, StaticAncestryOracle({}))
    assert record["hard"] == ["fabrication"] and record["weight"] == 0.0 and record["weight_basis"] == "hard-constraint"


def test_fabricated_wi_id_in_a_briefing(world, tmp_path):
    world.snapshot(T0, [WI("a", "working")])
    world.snapshot(T0 + hours(2), [WI("a", "done")])
    item = next(c for c in one_spec(world, Family.BRIEFING) if c.spec.snapshot_id.endswith("100000Z"))

    def script(body, i):
        results = tool_results(body, "openai")
        if not results:
            return Call("mcp__anima__list_work_items", {"project_id": PROJECT})
        rec = receipt_from(results[0])
        doc = {"contract": C.BRIEFING, "language": "en", "generated_at": 1, "in_flight": [], "blocked": [], "next": [],
               "done": [{"entity": {"kind": "work-item", "id": "ghost"}, "summary": "s", "receipts": [rec]}]}
        return Say(json.dumps(doc))
    with FakeLLM("openai", script) as fake:
        snapshot = world.store().get(item.spec.snapshot_id)
        servers, _ = episode_servers(snapshot, DojoLedger(tmp_path / "l.jsonl"), StaticAncestryOracle({}))
        result = R.ModelEpisodeRunner(adapter(fake, "openai"), SYSTEM).run_traced(item.spec, servers)
    record = B._verification_record(item, result, snapshot, StaticAncestryOracle({}))
    assert "fabrication" in record["hard"] and record["weight"] == 0.0


def test_chase_message_envelope_becomes_a_send_step(world, tmp_path):
    world.snapshot(T0, [WI("b", "working", blockers=(("x", "working"),)), WI("x", "working", executor="imhotep")])
    (item,) = one_spec(world, Family.CHASE)

    def script(body, i):
        results = tool_results(body, "openai")
        if not results:
            return Call("mcp__anima__get_wi_loop_state", {"work_item_id": "b"})
        sit = situation(body, "openai")
        now = int(re.search(r"Now \(unix seconds\): (\d+)", user_text(body, "openai")).group(1))
        msg = {"kind": "message", "recipient": "imhotep", "recipient_class": "executor", "protocol": "coordination/request@v1",
               "payload": {"objective": "unblock", "about": [sit["work_item"], sit["blocked_on"]],
                           "receipts": [receipt_from(results[0])], "ack_deadline": now + 100}, "follow_up_at": now + 200}
        return Say(json.dumps(msg))
    with FakeLLM("openai", script) as fake:
        snapshot = world.store().get(item.spec.snapshot_id)
        servers, _ = episode_servers(snapshot, DojoLedger(tmp_path / "l.jsonl"), StaticAncestryOracle({}))
        result = R.ModelEpisodeRunner(adapter(fake, "openai"), SYSTEM).run_traced(item.spec, servers)
    msg = result.episode.steps[-1].action
    assert isinstance(msg, SendMessage) and msg.recipient_class is RecipientClass.EXECUTOR and result.trace.contract_ok
    record = B._verification_record(item, result, snapshot, StaticAncestryOracle({}))
    assert record["weight"] > 0.9 and record["hard"] == []


def test_unknown_tool_and_bad_arguments_go_back_to_the_model(park_world, tmp_path):
    (item,) = [c for c in one_spec(park_world) if c.spec.subject_ids == ("p",)]

    def script(body, i):
        n = len(tool_results(body, "openai"))
        if n == 0:
            return Calls([Call("mcp__anima__no_such_tool", {}), Call("mcp__anima__get_work_item_detail", "{oops")])
        return Say("I give up")
    with FakeLLM("openai", script) as fake:
        servers, _ = episode_servers(park_world.store().get(item.spec.snapshot_id), DojoLedger(tmp_path / "l.jsonl"),
                                     StaticAncestryOracle({}))
        result = R.ModelEpisodeRunner(adapter(fake, "openai"), SYSTEM).run_traced(item.spec, servers)
    assert len(result.trace.protocol_errors) == 2 and not result.episode.steps  # nothing reached a server
    assert result.trace.answer_status == "unparseable" and result.trace.status == "ok"
    errors = [r for r in tool_results(fake.requests[1]["body"], "openai")]
    assert all(e["isError"] for e in errors)


def test_round_cap_and_declaration_is_stripped(park_world, tmp_path):
    (item,) = [c for c in one_spec(park_world) if c.spec.subject_ids == ("p",)]
    call = Call("mcp__anima__get_work_item_detail", {"id": "p", "cos_declaration": {"tier": "coordination", "receipts": []}})
    with FakeLLM("openai", lambda b, i: call) as fake:
        servers, log = episode_servers(park_world.store().get(item.spec.snapshot_id), DojoLedger(tmp_path / "l.jsonl"),
                                       StaticAncestryOracle({}))
        result = R.ModelEpisodeRunner(adapter(fake, "openai"), SYSTEM, R.RunnerConfig(max_rounds=3)).run_traced(item.spec, servers)
    assert result.trace.answer_status == "round-cap" and result.trace.rounds == 3 and len(result.episode.steps) == 3
    assert log.receipts[0].arguments == {"id": "p"}  # the runtime strips it before the tool runs
    assert result.trace.declarations[0]["declaration"]["tier"] == "coordination"


def test_model_error_becomes_a_failed_episode(park_world, tmp_path):
    (item,) = [c for c in one_spec(park_world) if c.spec.subject_ids == ("p",)]
    with FakeLLM("openai", lambda b, i: Fail(500)) as fake:
        servers, _ = episode_servers(park_world.store().get(item.spec.snapshot_id), DojoLedger(tmp_path / "l.jsonl"),
                                     StaticAncestryOracle({}))
        result = R.ModelEpisodeRunner(adapter(fake, "openai"), SYSTEM).run_traced(item.spec, servers)
    assert result.trace.status == "error" and "HTTP 500" in result.trace.error and result.trace.error_code


# --- batch, resume, containment ---------------------------------------------------------------------------------------------------


def read(out, name):
    return [json.loads(x) for x in (out / name).read_text().splitlines()]


def test_batch_writes_files_and_summary(park_world, tmp_path):
    out = tmp_path / "out"
    with FakeLLM("openai", park_script("openai")) as fake:
        summary = batch(park_world, fake, "openai", out, concurrency=2)
    assert {p.name for p in out.iterdir()} >= {"episodes.jsonl", "verification.jsonl", "provenance.json", "baseline-summary.json"}
    fam = summary["families"]["park-triage"]
    assert fam["n"] == 2 and fam["errors"] == 0 and fam["mean_weight"] > 0.9
    assert fam["hard_constraints"]["fabrication"] == 0 and fam["contract_validity"]["rate"] == 1.0
    assert fam["language_check"]["rate"] == 1.0 and fam["ci95"] and fam["tokens"]["completion_total"] > 0
    assert json.loads((out / "baseline-summary.json").read_text())["provenance"]["prompt_sha256"] == "p"
    text = json.dumps(summary)
    assert "sk-test" not in text and "sk-test" not in (out / "episodes.jsonl").read_text()


def test_resume_skips_completed_episodes(park_world, tmp_path):
    out = tmp_path / "out"
    with FakeLLM("openai", park_script("openai")) as fake:
        batch(park_world, fake, "openai", out)
        calls_first = len(fake.requests)
        batch(park_world, fake, "openai", out)
        assert len(fake.requests) == calls_first  # nothing re-run
    assert len(read(out, "verification.jsonl")) == 2
    with FakeLLM("openai", park_script("openai")) as other:
        with pytest.raises(RunProvenanceError, match="model_id"):
            batch(park_world, other, "openai", out, model_id="somebody-else")


def test_errors_are_contained_and_retried_on_resume(park_world, tmp_path):
    out = tmp_path / "out"
    good = park_script("openai")

    def flaky(body, i):
        return Fail(500) if situation(body, "openai")["work_item"] == "q" else good(body, i)
    with FakeLLM("openai", flaky) as fake:
        summary = batch(park_world, fake, "openai", out)
    fam = summary["families"]["park-triage"]
    assert fam["n"] == 2 and fam["errors"] == 1 and fam["scored"] == 1 and fam["mean_weight"] > 0.9
    assert fam["mean_weight_errors_as_zero"] < fam["mean_weight"]
    with FakeLLM("openai", good) as fake:
        summary = batch(park_world, fake, "openai", out)
        assert {situation(r["body"], "openai")["work_item"] for r in fake.requests} == {"q"}  # only the failed one
    assert summary["families"]["park-triage"]["errors"] == 0 and summary["overall"]["scored"] == 2


def test_selection_is_deterministic_and_limits_per_family(world):
    world.snapshot(T0, [WI(f"p{n}", "parked", parked_reason="rebase-conflict") for n in range(6)])
    ids = lambda seed: [c.spec.episode_id for c in one_spec_seeded(world, seed, 3)]  # noqa: E731
    assert ids(1) == ids(1) and len(ids(1)) == 3 and ids(1) != ids(2)


def one_spec_seeded(world, seed, limit):
    return B.select_episodes(world.store(), StaticAncestryOracle({}), [Family.PARK_TRIAGE], "all", limit, seed)[0]


# --- compare -------------------------------------------------------------------------------------------------------------------------


def test_compare_pairs_on_shared_ids(park_world, tmp_path):
    outs = {}
    for label, cause in (("parent", "ci-red"), ("teacher", "conflict")):
        out = outs[label] = tmp_path / label
        with FakeLLM("openai", park_script("openai", cause=cause)) as fake:
            batch(park_world, fake, "openai", out)
        summary = json.loads((out / "baseline-summary.json").read_text())
        summary["label"] = label
        (out / "baseline-summary.json").write_text(json.dumps(summary))
    result = B.compare_arms([B.load_arm(outs["parent"]), B.load_arm(outs["teacher"])])
    row = next(r for r in result["rows"] if r["family"] == "park-triage")
    assert row["n"] == 2 and row["arms"]["teacher"]["mean"] > row["arms"]["parent"]["mean"]
    pair = row["pairs"][0]
    assert pair["label"] == "teacher - parent" and pair["delta"] > 0 and pair["verdict"] in {"better", "no significant difference"}
    assert "park-triage" in B.render_compare(result)
    with pytest.raises(CompareError):
        B.compare_arms([B.load_arm(outs["parent"])])


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    values = [0.2, 0.4, 0.6, 0.8, 1.0, 0.5]
    assert B.bootstrap_ci(values, "s") == B.bootstrap_ci(values, "s")
    lo, hi = B.bootstrap_ci(values, "s")
    assert lo <= sum(values) / len(values) <= hi and B.bootstrap_ci([], "s") is None and B.bootstrap_ci([0.3], "s") == (0.3, 0.3)


# --- provenance ------------------------------------------------------------------------------------------------------------------------------


def test_dirty_tree_is_refused_unless_allowed(tmp_path):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True,
                       env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})
    git("init", "-q")
    (tmp_path / "f").write_text("1")
    git("add", "f")
    git("commit", "-qm", "c")
    clean = B.git_provenance(tmp_path, False)
    assert len(clean["git_sha"]) == 40 and clean["git_dirty"] is False
    (tmp_path / "f").write_text("2")
    with pytest.raises(RunProvenanceError):
        B.git_provenance(tmp_path, False)
    assert B.git_provenance(tmp_path, True)["git_dirty"] is True


def test_system_prompt_is_lilith_role_plus_skill_and_hashed():
    assert "You are Lilith" in SYSTEM.text and "Chief of Staff" in SYSTEM.text and "{mcp_servers}" not in SYSTEM.text
    assert "anima, git" in SYSTEM.text and len(SYSTEM.sha256) == 64 and len(SYSTEM.sources) == 2


# --- CLI end to end ---------------------------------------------------------------------------------------------------------------------------------


def test_cli_run_and_compare(park_world, tmp_path, capsys):
    key = tmp_path / "key"
    key.write_text("sk-test-not-a-real-key")
    key.chmod(0o600)
    args = ["run", "--root", str(park_world.root), "--repo", str(Path(R.REPO_ROOT)), "--family", "park-triage", "--split", "all",
            "--model-kind", "openai", "--model", "lilith", "--key-file", str(key), "--allow-dirty", "--attempts", "1",
            "--label", "cli-arm"]
    with FakeLLM("openai", park_script("openai")) as fake:
        out = tmp_path / "cli-out"
        code = B.main([*args, "--base-url", fake.base_url, "--out", str(out)])
        assert code == 0
        assert B.main(["run", *args[1:], "--base-url", fake.base_url, "--out", str(tmp_path / "dry"), "--dry-run"]) == 0
    printed = capsys.readouterr()
    assert "park-triage" in printed.out and "sk-test" not in printed.out + printed.err
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["endpoint_host"] == "127.0.0.1" and prov["git_dirty"] in (True, False) and len(prov["prompt_sha256"]) == 64
    assert B.main(["compare", str(out), str(out)]) == 0
    assert "Paired comparison" in capsys.readouterr().out
    assert B.main(["compare", str(tmp_path / "nope")]) == 2
