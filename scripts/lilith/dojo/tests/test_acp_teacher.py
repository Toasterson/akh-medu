"""The teacher arm through a sealed akh over ACP (acp_teacher.py + mcp_gateway.py). Fake ACP peer only."""
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from dojo import acp_teacher as T
from dojo import baseline as B
from dojo import runner as R
from dojo.episode import EmitDocument, ToolCall
from dojo.errors import DojoError
from dojo.mcp_gateway import ReplayMcpGateway
from dojo.model_adapters import Secret
from dojo.replay_server import DojoLedger, StaticAncestryOracle, episode_servers
from dojo.snapshot import sha256_hex
from fake_acp import TOKEN, FakeAkhomed, PeerState
from fake_llm import FakeLLM
from fixtures import T0, WI
from test_runner import adapter, one_spec, park_doc, park_script, provenance, tool_results

SYSTEM = R.compose_system_prompt()
SECRET = Secret(TOKEN)


@pytest.fixture
def park_world(world):
    world.snapshot(T0, [WI("p", "parked", parked_reason="rebase-conflict"), WI("q", "parked", parked_reason="rebase-conflict")])
    return world


def receipt_of(text: str) -> dict:
    meta = json.loads(text)["_meta"]
    return {"kind": "tool-call", "reference": meta["lilith/receipt"], "observed_at": meta["lilith/observed_at"],
            "digest": meta["lilith/digest"]}


def park_akh(ctx):
    """The fake akh's model: read the parked item through the replay tools, answer with receipts."""
    wid = ctx.situation["work_item"]
    seen = ctx.call("mcp__anima__get_work_item_detail", {"id": wid,
                    "cos_declaration": {"tier": "read", "receipts": []}}, text="Reading the item first.")
    ctx.say(json.dumps(park_doc(wid, "conflict", "rebase", [receipt_of(seen)])))


def teacher(peer, gateway, **config):
    return T.AcpEpisodeRunner(T.tcp_connector(peer.endpoint), SECRET, SYSTEM, gateway,
                              T.AcpTeacherConfig(akh="dojo-teacher", rpc_timeout_s=10, turn_timeout_s=20, **config))


def the_p_item(world):
    (item,) = [c for c in one_spec(world) if c.spec.subject_ids == ("p",)]
    return item


def servers_for(world, item, tmp_path, name="l.jsonl"):
    return episode_servers(world.store().get(item.spec.snapshot_id), DojoLedger(tmp_path / name),
                           StaticAncestryOracle({}))


# --- routing and receipts -------------------------------------------------------------------------------------


def test_episode_through_the_akh_carries_the_servers_receipts(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, log = servers_for(park_world, item, tmp_path)
    with FakeAkhomed(park_akh) as peer, ReplayMcpGateway() as gateway:
        result = teacher(peer, gateway).run_traced(item.spec, servers)
    ep, trace = result.episode, result.trace
    assert trace.status == "ok" and trace.answer_status == "document" and trace.contract_ok, trace.to_dict()
    # the receipts are the servers' own, unmodified, and the steps cite them
    assert ep.receipts == tuple(log.receipts) and [r.call_id for r in ep.receipts] == ["r1"]
    call, doc = ep.steps[0].action, ep.steps[1].action
    assert isinstance(call, ToolCall) and (call.tool, call.server, call.receipt_id) == ("get_work_item_detail", "anima", "r1")
    assert isinstance(doc, EmitDocument) and doc.document["receipts"][0]["reference"] == "r1"
    assert ep.steps[0].why_this_move == "Reading the item first."
    # cos_declaration stripped before the server, kept in the trace (as in the in-process broker)
    assert log.receipts[0].arguments == {"id": "p"}
    assert trace.declarations[0]["declaration"]["tier"] == "read"
    # the akh received the episode's replay servers, under the live names, behind a per-episode bearer
    (session,) = peer.sessions
    assert [s["name"] for s in session["mcpServers"]] == ["anima", "git"]
    assert all(s["url"].startswith("http://127.0.0.1:") for s in session["mcpServers"])
    assert session["_meta"] == {"akh/workspace": "dojo-teacher"}
    # the prompt carries the same system prompt and user prompt bytes as the in-process arm
    text = peer.contexts[0].prompt_text
    assert SYSTEM.text in text and "Situation (JSON):" in text
    # usage is the akh's own counter difference over the turn
    assert trace.usage.completion_tokens == 250 and trace.usage.prompt_tokens == 1000
    record = B._verification_record(item, result, park_world.store().get(item.spec.snapshot_id), StaticAncestryOracle({}))
    assert record["weight"] > 0.9 and record["hard"] == [] and record["weight_basis"] == "answer-step"


def test_the_akh_reads_the_same_bytes_as_the_in_process_arm(park_world, tmp_path):
    """Same broker: same tool names and schemas, same result text, same prompt hashes."""
    item = the_p_item(park_world)
    servers, _ = servers_for(park_world, item, tmp_path, "a.jsonl")
    listed = {}

    def script(ctx):
        listed["anima"] = ctx.list_tools("anima")
        listed["git"] = ctx.list_tools("git")
        park_akh(ctx)

    with FakeAkhomed(script) as peer, ReplayMcpGateway() as gateway:
        acp = teacher(peer, gateway).run_traced(item.spec, servers)
        akh_saw = peer.contexts[0].seen[0]
    with FakeLLM("openai", park_script("openai")) as fake:
        servers2, _ = servers_for(park_world, item, tmp_path, "b.jsonl")
        direct = R.ModelEpisodeRunner(adapter(fake, "openai"), SYSTEM).run_traced(item.spec, servers2)
        model_saw = tool_results(fake.requests[1]["body"], "openai")[0]
    assert acp.trace.tools_sha256 == direct.trace.tools_sha256
    assert acp.trace.user_prompt_sha256 == direct.trace.user_prompt_sha256
    assert json.loads(akh_saw) == model_saw  # the tool result the model reads, byte for byte in content
    broker = R.Broker(servers)
    names = {f"mcp__{server}__{t['name']}" for server, tools in listed.items() for t in tools}
    assert names == {s.name for s in broker.tool_specs()}
    assert {t["name"]: t["inputSchema"] for t in listed["anima"]} == {
        t["name"]: t["inputSchema"] for t in broker.server_tools("anima")}


def test_unknown_replay_tool_is_a_protocol_error_at_the_gateway(park_world, tmp_path):
    """A name the replay servers do not list reaches no server and issues no receipt; the gateway
    answers with the broker's protocol-error text (what the in-process arm returns to the model)."""
    item = the_p_item(park_world)
    servers, log = servers_for(park_world, item, tmp_path)
    broker = R.Broker(servers)
    with ReplayMcpGateway() as gateway:
        episode = gateway.open(broker)
        answer = episode.handle("anima", {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                                          "params": {"name": "no_such_tool", "arguments": {}}})
    seen = json.loads(answer["result"]["content"][0]["text"])
    assert answer["result"]["isError"] is True and seen["isError"] is True and "unknown tool" in seen["error"]
    assert log.receipts == [] and episode.rejected and episode.calls[0].dispatched.receipt_id is None


def test_a_tool_the_replay_servers_do_not_list_is_foreign(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, _ = servers_for(park_world, item, tmp_path)
    with FakeAkhomed(lambda ctx: ctx.call("mcp__anima__no_such_tool", {})) as peer, ReplayMcpGateway() as gateway:
        with pytest.raises(T.IsolationRefusal, match="no_such_tool"):
            teacher(peer, gateway).run_traced(item.spec, servers)
    assert peer.cancelled == ["s1"]


# --- isolation --------------------------------------------------------------------------------------------------------


def test_refuses_an_akh_with_tools_of_its_own(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, log = servers_for(park_world, item, tmp_path)
    state = PeerState(own_tools=["http_fetch", "kg_query", "mcp__anima__get_thread"])
    with FakeAkhomed(park_akh, state) as peer, ReplayMcpGateway() as gateway:
        with pytest.raises(T.IsolationRefusal, match="3 tool\\(s\\) of its own.*http_fetch"):
            teacher(peer, gateway).run_traced(item.spec, servers)
    assert peer.sessions == [] and peer.contexts == [] and log.receipts == []  # no prompt was ever sent


def test_refuses_when_the_tool_surface_cannot_be_verified(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, _ = servers_for(park_world, item, tmp_path)
    with FakeAkhomed(park_akh, PeerState(stale=True)) as peer, ReplayMcpGateway() as gateway:
        still_busy = teacher(peer, gateway)
        still_busy.IDLE_WAIT_S, still_busy.IDLE_POLL_S = 0.2, 0.05
        with pytest.raises(T.IsolationRefusal, match="stale"):
            still_busy.run_traced(item.spec, servers)
    with FakeAkhomed(park_akh, PeerState(mcp_http=False)) as peer, ReplayMcpGateway() as gateway:
        with pytest.raises(T.IsolationRefusal, match="mcpCapabilities.http"):
            teacher(peer, gateway).run_traced(item.spec, servers)


def test_waits_for_the_previous_turn_to_settle_before_verifying(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, _ = servers_for(park_world, item, tmp_path)
    with FakeAkhomed(park_akh, PeerState(stale_for=2)) as peer, ReplayMcpGateway() as gateway:
        settling = teacher(peer, gateway)
        settling.IDLE_POLL_S = 0.01
        assert settling.run_traced(item.spec, servers).trace.status == "ok"


def test_a_foreign_tool_mid_turn_cancels_and_refuses(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, _ = servers_for(park_world, item, tmp_path)

    def script(ctx):
        ctx.start_foreign_tool("http_fetch")
        ctx.say("{}")

    with FakeAkhomed(script) as peer, ReplayMcpGateway() as gateway:
        with pytest.raises(T.IsolationRefusal, match="http_fetch"):
            teacher(peer, gateway).run_traced(item.spec, servers)
    assert peer.cancelled == ["s1"]


def test_permission_is_granted_for_replay_tools_only(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, _ = servers_for(park_world, item, tmp_path)
    answers = {}

    def ok(ctx):
        answers["replay"] = ctx.ask_permission("mcp__anima__get_work_item_detail")
        park_akh(ctx)

    with FakeAkhomed(ok) as peer, ReplayMcpGateway() as gateway:
        assert teacher(peer, gateway).run_traced(item.spec, servers).trace.status == "ok"
    assert answers["replay"]["result"]["outcome"] == {"outcome": "selected", "optionId": "allow"}

    def foreign(ctx):
        answers["foreign"] = ctx.ask_permission("shell_exec")
        ctx.say("{}")

    servers2, _ = servers_for(park_world, item, tmp_path, "b.jsonl")
    with FakeAkhomed(foreign) as peer, ReplayMcpGateway() as gateway:
        with pytest.raises(T.IsolationRefusal, match="shell_exec"):
            teacher(peer, gateway).run_traced(item.spec, servers2)
    assert answers["foreign"]["result"]["outcome"] == {"outcome": "cancelled"}


def test_memory_written_during_an_episode_refuses(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, _ = servers_for(park_world, item, tmp_path)

    def remembers(ctx):
        park_akh(ctx)
        ctx.peer.state.triples += 3  # the akh stored something it read

    with FakeAkhomed(remembers) as peer, ReplayMcpGateway() as gateway:
        with pytest.raises(T.IsolationRefusal, match="wrote to its memory"):
            teacher(peer, gateway).run_traced(item.spec, servers)


def test_memory_changed_between_episodes_refuses(park_world, tmp_path):
    item = the_p_item(park_world)
    with FakeAkhomed(park_akh) as peer, ReplayMcpGateway() as gateway:
        runner = teacher(peer, gateway)
        servers, _ = servers_for(park_world, item, tmp_path)
        assert runner.run_traced(item.spec, servers).trace.status == "ok"
        peer.state.symbols += 1  # something outside the episode wrote to the teacher
        servers2, _ = servers_for(park_world, item, tmp_path, "b.jsonl")
        with pytest.raises(T.IsolationRefusal, match="between episodes"):
            runner.run_traced(item.spec, servers2)


def test_gateway_answers_only_the_open_episode_with_its_bearer(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, log = servers_for(park_world, item, tmp_path)
    with ReplayMcpGateway() as gateway:
        episode = gateway.open(R.Broker(servers))
        (anima, _) = episode.acp_mcp_servers()
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": "get_work_item_detail", "arguments": {"id": "p"}}}).encode()

        def post(url, bearer):
            request = urllib.request.Request(url, data=body, method="POST", headers={
                "Content-Type": "application/json", **({"Authorization": f"Bearer {bearer}"} if bearer else {})})
            try:
                with urllib.request.urlopen(request, timeout=5) as response:
                    return response.status
            except urllib.error.HTTPError as error:
                return error.code

        assert post(anima["url"], "wrong") == 404 and post(anima["url"], None) == 404
        assert post(anima["url"].rsplit("/", 1)[0] + "/shell", episode.token) == 404  # no such server
        assert post(anima["url"], episode.token) == 200 and len(log.receipts) == 1
        episode.close()
        assert post(anima["url"], episode.token) == 404  # closed with the episode
    assert len(log.receipts) == 1


def test_provider_error_is_an_unscored_episode(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, _ = servers_for(park_world, item, tmp_path)
    with FakeAkhomed(lambda ctx: ctx.say("error[ollama]: HTTP 429 quota")) as peer, ReplayMcpGateway() as gateway:
        result = teacher(peer, gateway).run_traced(item.spec, servers)
    assert result.trace.status == "error" and "429" in result.trace.error
    assert result.trace.error_code == "lilith::dojo::acp_transport"
    with FakeAkhomed(lambda ctx: "max_turn_requests") as peer, ReplayMcpGateway() as gateway:
        servers2, _ = servers_for(park_world, item, tmp_path, "b.jsonl")
        capped = teacher(peer, gateway).run_traced(item.spec, servers2)
    assert capped.trace.status == "ok" and capped.trace.answer_status == "round-cap"


def test_round_cap_cancels_the_turn(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, log = servers_for(park_world, item, tmp_path)

    def loops(ctx):
        for _ in range(4):
            ctx.call("mcp__anima__get_work_item_detail", {"id": "p"})
        return "cancelled"

    with FakeAkhomed(loops) as peer, ReplayMcpGateway() as gateway:
        result = teacher(peer, gateway, runner=R.RunnerConfig(max_rounds=2)).run_traced(item.spec, servers)
    assert peer.cancelled == ["s1"]
    assert result.trace.status == "ok" and result.trace.answer_status == "round-cap"
    assert len(result.episode.steps) == len(log.receipts)  # every call the servers saw is a step


def test_wrong_token_is_a_transport_error_and_never_leaks(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, _ = servers_for(park_world, item, tmp_path)
    with FakeAkhomed(park_akh) as peer, ReplayMcpGateway() as gateway:
        runner = T.AcpEpisodeRunner(T.tcp_connector(peer.endpoint), Secret("not-the-token-" + "x" * 20), SYSTEM, gateway,
                                    T.AcpTeacherConfig(akh="dojo-teacher", rpc_timeout_s=5))
        result = runner.run_traced(item.spec, servers)
    assert result.trace.status == "error" and "unauthenticated" in result.trace.error
    assert "not-the-token" not in json.dumps(result.trace.to_dict())


def test_endpoint_must_be_loopback_tcp():
    with pytest.raises(DojoError, match="not loopback"):
        T.tcp_connector("tcp://10.0.0.5:18444")
    with pytest.raises(DojoError, match="tcp://host:port"):
        T.tcp_connector("wss://anima.example/akh/acp")


def test_teacher_config_must_be_sealed(tmp_path):
    good = tmp_path / "dojo-teacher.toml"
    good.write_text('name = "dojo-teacher"\nmodel = "glm-5.3"\nbackend = "ollama"\ncapture_transcripts = false\n'
                    'tool_surface = "session-only"\n[mcp]\nservers = []\n')
    declared = T.read_teacher_config(good, "dojo-teacher")
    assert declared["model"] == "glm-5.3" and len(declared["sha256"]) == 64
    bad = tmp_path / "janus.toml"
    bad.write_text('name = "janus"\ncapture_transcripts = true\n')
    with pytest.raises(T.IsolationRefusal, match="tool_surface.*capture_transcripts|name"):
        T.read_teacher_config(bad, "dojo-teacher")


# --- prompt placement parity ---------------------------------------------------------------------------------------


def sealed_toml(tmp_path, prompt, name="dojo-teacher.toml"):
    path = tmp_path / name
    path.write_text('name = "dojo-teacher"\nmodel = "glm-5.3"\nbackend = "ollama"\ncapture_transcripts = false\n'
                    f'tool_surface = "session-only"\nsystem_prompt = {json.dumps(prompt, ensure_ascii=False)}\n'
                    '[mcp]\nservers = []\n')
    return path


def test_akh_system_placement_sends_the_task_alone(park_world, tmp_path):
    """The dojo prompt is the akh's system prompt; the prompt turn is exactly the parent's user message."""
    item = the_p_item(park_world)
    servers, _ = servers_for(park_world, item, tmp_path)
    with FakeAkhomed(park_akh) as peer, ReplayMcpGateway() as gateway:
        runner = teacher(peer, gateway, placement=T.PromptPlacement.AKH_SYSTEM)
        result = runner.run_traced(item.spec, servers)
        assert runner.probe()["prompt_delivery"] == "akh-system-prompt"
    assert result.trace.status == "ok" and result.trace.contract_ok
    (blocks,) = [c.prompt_blocks for c in peer.contexts]
    assert len(blocks) == 1 and SYSTEM.text not in blocks[0]["text"] and T.PROMPT_PREAMBLE not in blocks[0]["text"]
    assert sha256_hex(blocks[0]["text"].encode("utf-8")) == result.trace.user_prompt_sha256


def test_session_turn_placement_is_the_recorded_legacy_layout(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, _ = servers_for(park_world, item, tmp_path)
    with FakeAkhomed(park_akh) as peer, ReplayMcpGateway() as gateway:
        runner = teacher(peer, gateway)
        runner.run_traced(item.spec, servers)
        assert runner.probe()["prompt_delivery"] == "acp-session-prompt"
    assert SYSTEM.text in peer.contexts[0].prompt_text


def test_prompt_parity_is_verified_against_the_rendered_akh_config(tmp_path):
    ok = T.read_teacher_config(sealed_toml(tmp_path, SYSTEM.text), "dojo-teacher")
    assert ok["system_prompt_sha256"] == SYSTEM.sha256
    T.verify_prompt_parity(ok, SYSTEM)
    other = T.read_teacher_config(sealed_toml(tmp_path, "You are the teacher.", "o.toml"), "dojo-teacher")
    with pytest.raises(T.PromptParityRefusal, match="system_prompt is sha256"):
        T.verify_prompt_parity(other, SYSTEM)
    unset = tmp_path / "u.toml"
    unset.write_text('name = "dojo-teacher"\ncapture_transcripts = false\ntool_surface = "session-only"\n')
    with pytest.raises(T.PromptParityRefusal, match="not set"):
        T.verify_prompt_parity(T.read_teacher_config(unset, "dojo-teacher"), SYSTEM)
    with pytest.raises(T.PromptParityRefusal, match="--akh-config"):
        T.verify_prompt_parity(None, SYSTEM)


def test_cli_defaults_to_akh_placement_and_refuses_an_unverified_prompt(park_world, tmp_path, capsys):
    key = tmp_path / "dojo-teacher.key"
    key.write_text(TOKEN + "\n")
    key.chmod(0o600)
    common = ["run", "--root", str(park_world.root), "--repo", str(Path(R.REPO_ROOT)), "--family", "park-triage",
              "--split", "all", "--model-kind", "acp", "--key-file", str(key), "--allow-dirty", "--label", "teacher"]
    with FakeAkhomed(park_akh) as peer:
        endpoint = ["--acp-endpoint", peer.endpoint]
        assert B.main([*common, *endpoint, "--out", str(tmp_path / "none")]) == 2  # no --akh-config
        wrong = sealed_toml(tmp_path, "You are the teacher.", "wrong.toml")
        assert B.main([*common, *endpoint, "--akh-config", str(wrong), "--out", str(tmp_path / "wrong")]) == 2
        assert peer.sessions == []  # refused before a session was opened
        good = sealed_toml(tmp_path, SYSTEM.text)
        out = tmp_path / "good"
        assert B.main([*common, *endpoint, "--akh-config", str(good), "--out", str(out)]) == 0
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["teacher"]["prompt_delivery"] == "akh-system-prompt"
    assert prov["teacher"]["declared"]["system_prompt_sha256"] == SYSTEM.sha256 == prov["system_prompt_sha256"]
    assert all(len(c.prompt_blocks) == 1 for c in peer.contexts)
    err = capsys.readouterr().err
    assert "teacher_prompt_parity" in err and "--teacher-prompt session" in err


# --- batch + CLI ---------------------------------------------------------------------------------------------------


def test_batch_through_the_akh_writes_the_same_files(park_world, tmp_path):
    out = tmp_path / "out"
    with FakeAkhomed(park_akh) as peer, ReplayMcpGateway() as gateway:
        runner = teacher(peer, gateway)
        summary = B.run_batch(park_world.store(), one_spec(park_world), None, SYSTEM, StaticAncestryOracle({}),
                              B.BatchConfig(out, 1), provenance(model_kind="acp", model_id=runner.model_id),
                              "teacher", episode_runner=runner)
    fam = summary["families"]["park-triage"]
    assert fam["n"] == 2 and fam["errors"] == 0 and fam["mean_weight"] > 0.9 and fam["hard_constraints"]["fabrication"] == 0
    assert len(peer.sessions) == 2  # one ACP session per episode
    assert TOKEN not in (out / "episodes.jsonl").read_text() + json.dumps(summary)


def test_batch_aborts_on_isolation_refusal(park_world, tmp_path):
    with FakeAkhomed(park_akh, PeerState(own_tools=["web_search"])) as peer, ReplayMcpGateway() as gateway:
        with pytest.raises(T.IsolationRefusal):
            B.run_batch(park_world.store(), one_spec(park_world), None, SYSTEM, StaticAncestryOracle({}),
                        B.BatchConfig(tmp_path / "out", 1), provenance(model_kind="acp"), "teacher",
                        episode_runner=teacher(peer, gateway))
    assert not (tmp_path / "out" / "verification.jsonl").exists()


def test_cli_acp_run_compare_and_refusals(park_world, tmp_path, capsys):
    key = tmp_path / "dojo-teacher.key"
    key.write_text(TOKEN + "\n")
    key.chmod(0o600)
    common = ["run", "--root", str(park_world.root), "--repo", str(Path(R.REPO_ROOT)), "--family", "park-triage",
              "--split", "all", "--model-kind", "acp", "--key-file", str(key), "--allow-dirty", "--label", "teacher",
              "--teacher-prompt", "session"]
    with FakeAkhomed(park_akh) as peer:
        out = tmp_path / "teacher"
        assert B.main([*common, "--acp-endpoint", peer.endpoint, "--out", str(out)]) == 0
        # resume: nothing re-runs
        assert B.main([*common, "--acp-endpoint", peer.endpoint, "--out", str(out)]) == 0
        assert len(peer.sessions) == 2
        assert B.main([*common, "--acp-endpoint", peer.endpoint, "--out", str(tmp_path / "c"), "--concurrency", "2"]) == 2
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["model_kind"] == "acp" and prov["model_id"] == "akh:dojo-teacher"
    assert prov["teacher"]["own_tools"] == 0 and prov["teacher"]["memory"] == {"symbols": 10, "triples": 20}
    assert B.main(["compare", str(out), str(out)]) == 0
    with FakeAkhomed(park_akh, PeerState(own_tools=["http_fetch"])) as peer:
        assert B.main([*common, "--acp-endpoint", peer.endpoint, "--out", str(tmp_path / "leaky")]) == 2
    assert not (tmp_path / "leaky" / "provenance.json").exists()  # refused before anything was written
    printed = capsys.readouterr()
    assert "teacher_isolation" in printed.err and "one episode at a time" in printed.err
    assert TOKEN not in printed.out + printed.err
