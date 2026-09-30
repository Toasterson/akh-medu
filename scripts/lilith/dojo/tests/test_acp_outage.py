"""Transport robustness of the ACP teacher arm (Anima WI C4 01a0e270-475e-74d0-9701-c1806eab79d0).

The 2026-09-30 sealed run: the Roman core restarted for a deploy and 300 of 310 episodes were
recorded as `cannot connect` errors within seconds; six GLM-5.3 turns were cut at 300 s; the resume
was refused because a re-rendered TOML changed the teacher's file hash. Fake ACP peer only."""
import json
import time
from pathlib import Path

import pytest

from dojo import acp_teacher as T
from dojo import baseline as B
from dojo import model_profile as MP
from dojo import runner as R
from dojo.errors import RunProvenanceError
from dojo.mcp_gateway import ReplayMcpGateway
from dojo.replay_server import StaticAncestryOracle
from fake_acp import TOKEN, FakeAkhomed, PeerState
from test_acp_teacher import SECRET, SYSTEM, park_akh, park_world, servers_for, the_p_item, teacher  # noqa: F401
from test_runner import one_spec, provenance


class FakeTime:
    """A monotonic clock that only sleeping advances beyond real time, so backoff tests take no wall time."""

    def __init__(self, on_sleep=None):
        self.offset = 0.0
        self.slept: list[float] = []
        self.lines: list[str] = []
        self.on_sleep = on_sleep

    def clock(self) -> float:
        return time.monotonic() + self.offset

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.offset += seconds
        if self.on_sleep:
            self.on_sleep(len(self.slept))


def waiting_teacher(peer, gateway, fake: FakeTime, budget_s=900.0, **config):
    runner = T.AcpEpisodeRunner(
        T.tcp_connector(peer.endpoint), SECRET, SYSTEM, gateway,
        T.AcpTeacherConfig(akh="dojo-teacher", rpc_timeout_s=10, turn_timeout_s=20,
                           outage=T.OutagePolicy(budget_s=budget_s), **config),
        clock=fake.clock, sleep=fake.sleep, notify=fake.lines.append)
    return runner


# --- the outage policy ----------------------------------------------------------------------------------------


def test_backoff_doubles_to_the_cap_and_is_clipped_to_the_budget():
    policy = T.OutagePolicy(initial_s=5, max_backoff_s=60, factor=2, budget_s=900)
    assert [policy.wait_after(n, 0) for n in range(6)] == [5, 10, 20, 40, 60, 60]
    assert policy.wait_after(0, 899) == 1      # the last wait ends at the budget
    assert policy.wait_after(3, 900) is None   # spent
    assert T.OutagePolicy(budget_s=0).wait_after(0, 0) is None  # 0 = stop at the first outage


# --- an unreachable endpoint is an outage, not an episode result ----------------------------------------------


def test_the_same_episode_is_retried_until_the_core_is_back(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, _ = servers_for(park_world, item, tmp_path)
    with FakeAkhomed(park_akh) as peer, ReplayMcpGateway() as gateway:
        peer.go_away()
        fake = FakeTime(on_sleep=lambda n: peer.come_back() if n == 3 else None)  # up again after ~35 s
        runner = waiting_teacher(peer, gateway, fake)
        result = runner.run_traced(item.spec, servers)
    assert result.trace.status == "ok" and result.trace.answer_status == "document", result.trace.to_dict()
    assert fake.slept == [5, 10, 20]
    assert [o["attempt"] for o in runner.outages] == [1, 2, 3]
    assert all("Connection refused" in o["error"] or "refused" in o["error"].lower() for o in runner.outages)
    assert sum("waiting 5s" in line or "waiting 10s" in line or "waiting 20s" in line for line in fake.lines) == 3
    assert "is back" in fake.lines[-1]
    assert len(peer.sessions) == 1  # one ACP session: the episode ran once, after the core came back


def test_a_peer_that_drops_before_the_prompt_is_an_outage_and_the_preflight_runs_again(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, _ = servers_for(park_world, item, tmp_path)
    with FakeAkhomed(park_akh, PeerState(close_on_initialize=2)) as peer, ReplayMcpGateway() as gateway:
        fake = FakeTime()
        result = waiting_teacher(peer, gateway, fake).run_traced(item.spec, servers)
    assert result.trace.status == "ok" and fake.slept == [5, 10]
    methods = [m.get("method") for m in peer.received]
    # the third connection ran the whole isolation preflight (tools, memory) before the prompt
    assert methods.count("initialize") == 3 and methods.count("_akh/tools") == 1
    assert methods.index("_akh/tools") > max(i for i, m in enumerate(methods) if m == "initialize")
    assert len(peer.sessions) == 1


def test_a_refused_preflight_after_reconnect_still_stops_the_run(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, _ = servers_for(park_world, item, tmp_path)
    with FakeAkhomed(park_akh, PeerState(close_on_initialize=1, own_tools=["http_fetch"])) as peer, ReplayMcpGateway() as gateway:
        with pytest.raises(T.IsolationRefusal, match="tool\\(s\\) of its own"):
            waiting_teacher(peer, gateway, FakeTime()).run_traced(item.spec, servers)
    assert peer.sessions == []  # fail closed: no prompt reached a teacher that can reach anything else


def test_an_exhausted_budget_stops_the_batch_without_recording_the_rest(park_world, tmp_path):
    out = tmp_path / "out"
    chosen = one_spec(park_world)
    assert len(chosen) == 2

    def first_episode_then_the_core_dies(ctx):
        park_akh(ctx)
        ctx.peer.go_away(drop_connections=False)  # the restart begins after the first answer

    fake = FakeTime()
    with FakeAkhomed(first_episode_then_the_core_dies) as peer, ReplayMcpGateway() as gateway:
        runner = waiting_teacher(peer, gateway, fake, budget_s=30)
        with pytest.raises(T.AcpOutageExhausted, match="stayed unreachable") as stopped:
            B.run_batch(park_world.store(), chosen, None, SYSTEM, StaticAncestryOracle({}), B.BatchConfig(out, 1),
                        provenance(model_kind="acp", model_id=runner.model_id), "teacher", episode_runner=runner)
        assert fake.slept == pytest.approx([5, 10, 15], abs=0.5)  # clipped: 30 s in all
        assert stopped.value.code == "lilith::dojo::acp_outage_exhausted" and "--outage-budget" in stopped.value.help
        # one episode finished; the other was NOT recorded as an error
        records = [json.loads(line) for line in (out / "verification.jsonl").read_text().splitlines()]
        assert [r["status"] for r in records] == ["ok"]
        # resume once the core is back: the remaining episode runs, nothing else does
        peer.come_back()
        summary = B.run_batch(park_world.store(), chosen, None, SYSTEM, StaticAncestryOracle({}), B.BatchConfig(out, 1),
                              provenance(model_kind="acp", model_id=runner.model_id, started_at="t2"), "teacher",
                              episode_runner=waiting_teacher(peer, gateway, FakeTime(), budget_s=30))
    assert summary["overall"]["n"] == 2 and summary["overall"]["errors"] == 0
    assert len([s for s in peer.sessions if s["mcpServers"]]) == 2  # first run 1, resume 1: never a repeat of a finished episode


def test_a_zero_budget_stops_at_the_first_outage(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, _ = servers_for(park_world, item, tmp_path)
    with FakeAkhomed(park_akh) as peer, ReplayMcpGateway() as gateway:
        peer.go_away()
        fake = FakeTime()
        with pytest.raises(T.AcpOutageExhausted):
            waiting_teacher(peer, gateway, fake, budget_s=0).run_traced(item.spec, servers)
    assert fake.slept == []


def test_the_token_never_reaches_an_outage_line(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, _ = servers_for(park_world, item, tmp_path)
    with FakeAkhomed(park_akh, PeerState(close_on_initialize=1)) as peer, ReplayMcpGateway() as gateway:
        fake = FakeTime()
        runner = waiting_teacher(peer, gateway, fake)
        runner.run_traced(item.spec, servers)
    assert TOKEN not in "\n".join(fake.lines) + json.dumps(runner.outages)


# --- the connection is lost during the turn ---------------------------------------------------------------------------


def test_a_connection_lost_mid_turn_is_an_error_episode_retried_on_resume(park_world, tmp_path):
    out = tmp_path / "out"
    chosen = one_spec(park_world)
    with FakeAkhomed(park_akh, PeerState(close_on_prompt=1)) as peer, ReplayMcpGateway() as gateway:
        fake = FakeTime()
        runner = waiting_teacher(peer, gateway, fake)
        summary = B.run_batch(park_world.store(), chosen, None, SYSTEM, StaticAncestryOracle({}), B.BatchConfig(out, 1),
                              provenance(model_kind="acp", model_id=runner.model_id), "teacher", episode_runner=runner)
        assert summary["overall"]["n"] == 2 and summary["overall"]["errors"] == 1  # the batch was not stopped
        errored = [r for r in (json.loads(x) for x in (out / "verification.jsonl").read_text().splitlines())
                   if r["status"] != "ok"]
        assert len(errored) == 1 and "lost" in errored[0]["error"]
        episodes = [json.loads(x) for x in (out / "episodes.jsonl").read_text().splitlines()]
        assert {e["error_code"] for e in episodes if e["status"] != "ok"} == {"lilith::dojo::acp_connection_lost"}
        assert fake.slept == []  # the core was back for the next episode: no waiting
        # resume: only the lost episode runs again
        before = len([s for s in peer.sessions if s["mcpServers"]])
        summary = B.run_batch(park_world.store(), chosen, None, SYSTEM, StaticAncestryOracle({}), B.BatchConfig(out, 1),
                              provenance(model_kind="acp", model_id=runner.model_id, started_at="t2"), "teacher",
                              episode_runner=waiting_teacher(peer, gateway, FakeTime()))
        assert len([s for s in peer.sessions if s["mcpServers"]]) == before + 1
    assert summary["overall"]["errors"] == 0


def test_the_baseline_is_proven_again_after_a_restart(park_world, tmp_path):
    """A restarted akhomed interns its one-time init on the first prompt (515 -> 517); after an outage the
    warm-up runs again so the exact memory check still holds, instead of refusing the run."""
    item = the_p_item(park_world)
    servers, _ = servers_for(park_world, item, tmp_path)
    state = PeerState(symbols=515, triples=473, grow_first_prompt=2)
    with FakeAkhomed(park_akh, state) as peer, ReplayMcpGateway() as gateway:

        def restart(n):
            state.symbols, state.prompts_served = 515, 0  # fresh process: the init is owed again
            peer.come_back()

        fake = FakeTime(on_sleep=restart)
        runner = waiting_teacher(peer, gateway, fake, warm_up=True)
        assert runner.probe()["memory"] == {"symbols": 517, "triples": 473}
        peer.go_away()
        result = runner.run_traced(item.spec, servers)
    assert result.trace.status == "ok" and fake.slept == [5]
    assert len([x for x in peer.sessions if not x["mcpServers"]]) == 4  # two warm-ups, then two again after the restart
    assert len([x for x in peer.sessions if x["mcpServers"]]) == 1 and state.symbols == 517


def test_a_restart_that_writes_a_triple_is_still_refused(park_world, tmp_path):
    item = the_p_item(park_world)
    servers, _ = servers_for(park_world, item, tmp_path)
    state = PeerState(symbols=10, triples=20)
    with FakeAkhomed(park_akh, state) as peer, ReplayMcpGateway() as gateway:
        def restart(n):
            state.triples = 21  # the "restarted" akh holds a triple the baseline does not
            peer.come_back()
        fake = FakeTime(on_sleep=restart)
        runner = waiting_teacher(peer, gateway, fake, warm_up=True)
        runner.probe()
        peer.go_away()
        with pytest.raises(T.IsolationRefusal, match="memory changed between episodes"):
            runner.run_traced(item.spec, servers)


# --- the prompt timeout follows the akh's turn deadline ---------------------------------------------------------


CATALOG = MP.load_catalog(Path(R.REPO_ROOT) / MP.CATALOG_RELATIVE)


def test_the_repo_catalog_gives_glm_5_3_an_1800_s_deadline():
    assert CATALOG is not None
    resolved = MP.resolve_turn_timeout({"model": "glm-5.3"}, CATALOG)
    assert (resolved.deadline_s, resolved.seconds, resolved.source) == (1800.0, 1800.0 + MP.TURN_MARGIN_S, MP.TimeoutSource.CATALOG)
    assert resolved.catalog_row == "glm-5.3"


def test_effort_and_prefix_rows_and_the_cloud_tag():
    assert MP.resolve_turn_timeout({"model": "glm-5.3", "reasoning_effort": "low"}, CATALOG).deadline_s == 600.0
    assert MP.resolve_turn_timeout({"model": "GLM-5.3:cloud"}, CATALOG).deadline_s == 1800.0
    assert MP.resolve_turn_timeout({"model": "kimi-k3"}, CATALOG).deadline_s == 3000.0


def test_an_akh_override_beats_the_catalog_and_a_flag_beats_both():
    doc = {"model": "glm-5.3", "model_profile": {"turn_deadline_secs": 2400}}
    resolved = MP.resolve_turn_timeout(doc, CATALOG)
    assert (resolved.deadline_s, resolved.source) == (2400.0, MP.TimeoutSource.AKH_OVERRIDE)
    flagged = MP.resolve_turn_timeout(doc, CATALOG, flag=90.0)
    assert (flagged.seconds, flagged.source) == (90.0, MP.TimeoutSource.FLAG)


def test_unreadable_inputs_fall_back_to_1800_and_say_why():
    for doc, catalog, why in [(None, CATALOG, "config was not given"), ({}, CATALOG, "names no model"),
                              ({"model": "glm-5.3"}, None, "catalog is missing"),
                              ({"model": "no-such-model"}, CATALOG, "no row")]:
        resolved = MP.resolve_turn_timeout(doc, catalog)
        assert (resolved.seconds, resolved.source) == (1800.0, MP.TimeoutSource.FALLBACK)
        assert why in resolved.note and "--turn-timeout" in resolved.note


def test_the_in_process_http_timeout_follows_the_models_catalog_row():
    assert MP.resolve_http_timeout("lilith", CATALOG).seconds == 600.0   # gemma-4-12b row: served locally
    assert MP.resolve_http_timeout("kimi-k3", CATALOG).seconds == 600.0
    assert MP.resolve_http_timeout("glm-5.3", CATALOG).seconds == 300.0
    assert MP.resolve_http_timeout("claude-x", CATALOG).source is MP.TimeoutSource.FALLBACK
    assert MP.resolve_http_timeout("lilith", CATALOG, flag=45.0).seconds == 45.0


def teacher_toml(tmp_path, prompt, extra="", comment=""):
    path = tmp_path / "dojo-teacher.toml"
    top = ('name = "dojo-teacher"\nmodel = "glm-5.3"\nbackend = "ollama"\ncapture_transcripts = false\n'
           f'tool_surface = "session-only"\nsystem_prompt = {json.dumps(prompt)}\n')
    path.write_text(f"# {comment}\n{top}{extra}\n[mcp]\nservers = []\n")
    return path


def cli(park_world, tmp_path, peer, *extra):
    key = tmp_path / "dojo-teacher.key"
    key.write_text(TOKEN + "\n")
    key.chmod(0o600)
    return ["run", "--root", str(park_world.root), "--repo", str(Path(R.REPO_ROOT)), "--family", "park-triage",
            "--split", "all", "--model-kind", "acp", "--key-file", str(key), "--allow-dirty", "--label", "teacher",
            "--acp-endpoint", peer.endpoint, *extra]


def test_cli_records_the_turn_timeout_and_refuses_the_http_flag(park_world, tmp_path, capsys):
    config = teacher_toml(tmp_path, SYSTEM.text)
    out = tmp_path / "out"
    with FakeAkhomed(park_akh) as peer:
        args = cli(park_world, tmp_path, peer, "--akh-config", str(config), "--out", str(out))
        assert B.main(args) == 0
        assert B.main([*args[:-1], str(tmp_path / "x"), "--timeout", "300"]) == 2
    prov = json.loads((out / "provenance.json").read_text())
    turn = prov["timeouts"]["turn"]
    assert turn["seconds"] == 2100.0 and turn["deadline_s"] == 1800.0 and turn["source"] == "catalog"
    assert turn["catalog_row"] == "glm-5.3" and prov["timeouts"]["outage_budget_s"] == 900.0
    err = capsys.readouterr().err
    assert "teacher turn timeout 2100s" in err and "--turn-timeout" in err


def test_cli_turn_timeout_flag_is_taken_exactly(park_world, tmp_path):
    config = teacher_toml(tmp_path, SYSTEM.text)
    out = tmp_path / "out"
    with FakeAkhomed(park_akh) as peer:
        assert B.main(cli(park_world, tmp_path, peer, "--akh-config", str(config), "--turn-timeout", "77",
                          "--outage-budget", "0", "--out", str(out))) == 0
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["timeouts"]["turn"]["seconds"] == 77.0 and prov["timeouts"]["turn"]["source"] == "flag"
    assert prov["timeouts"]["outage_budget_s"] == 0.0


def test_cli_outage_stops_the_run_with_a_clear_error_and_records_nothing_per_episode(park_world, tmp_path, capsys):
    config = teacher_toml(tmp_path, SYSTEM.text)
    out = tmp_path / "out"
    with FakeAkhomed(park_akh) as peer:
        args = cli(park_world, tmp_path, peer, "--akh-config", str(config), "--outage-budget", "0", "--out", str(out))
        peer.go_away()
        assert B.main(args) == 2  # the probe cannot reach the core: nothing is written
    assert not (out / "verification.jsonl").exists()
    assert "cannot connect to the ACP endpoint" in capsys.readouterr().err


# --- resume identity covers only what changes the measurement --------------------------------------------------------


def teacher_record(**over):
    declared = {"backend": "ollama", "model": "glm-5.3", "reasoning_effort": None, "tool_surface": "session-only",
                "capture_transcripts": False, "system_prompt_sha256": "sp", "sha256": "file-1", "path": "/a/x.toml",
                "model_profile": {}}
    teacher = {"akh": "dojo-teacher", "agent_info": {"name": "janus", "version": "0.8.0"}, "own_tools": 0,
               "memory": {"symbols": 517, "triples": 473}, "prompt_delivery": "akh-system-prompt",
               "tool_transport": "client-pushed-http-mcp", "declared": declared}
    for key, value in over.items():
        if key.startswith("declared."):
            declared[key.split(".", 1)[1]] = value
        else:
            teacher[key] = value
    return {"teacher": teacher, "timeouts": {"turn": {"seconds": 2100.0}}}


def resume(tmp_path, first, second):
    out = tmp_path / "run"
    out.mkdir(exist_ok=True)
    (out / "provenance.json").write_text(json.dumps(provenance(**first)))
    return B.check_resume(out, provenance(**second))


def test_deployment_only_changes_do_not_block_a_resume(tmp_path):
    first = teacher_record()
    second = teacher_record(**{"declared.sha256": "file-2", "declared.path": "/b/x.toml",
                               "declared.model_profile": {"turn_deadline_secs": 2400},
                               "agent_info": {"name": "janus", "version": "0.9.0"},
                               "memory": {"symbols": 515, "triples": 473}})
    second["timeouts"] = {"turn": {"seconds": 2700.0}}
    earlier = resume(tmp_path, first, second)
    assert earlier is not None
    changed = B.deployment_changes(earlier, provenance(**second))
    assert set(changed) == {"teacher.declared.sha256", "teacher.declared.path", "teacher.declared.model_profile",
                            "teacher.agent_info", "teacher.memory", "timeouts"}
    assert changed["teacher.declared.sha256"] == {"was": '"file-1"', "now": '"file-2"'}


@pytest.mark.parametrize("change", [
    {"declared.model": "glm-5.4"}, {"declared.reasoning_effort": "low"}, {"declared.system_prompt_sha256": "other"},
    {"declared.tool_surface": "full"}, {"declared.capture_transcripts": True}, {"declared.backend": "openai"},
    {"own_tools": 1}, {"prompt_delivery": "acp-session-prompt"}, {"akh": "working-akh"},
])
def test_measurement_changes_block_a_resume_and_are_named(tmp_path, change):
    (key,) = change
    with pytest.raises(RunProvenanceError, match=f"different teacher.{key}") as refused:
        resume(tmp_path, teacher_record(), teacher_record(**change))
    assert "->" in str(refused.value)  # names the old and the new value


@pytest.mark.parametrize("change", [
    {"model_id": "akh:other"}, {"system_prompt_sha256": "x"}, {"prompt_sha256": "x"}, {"git_sha": "h" * 40},
    {"snapshot_root_sha256": "x"}, {"floors_sha256": "x"}, {"context_window": 1234}, {"tools_sha256": "x"},
])
def test_run_level_identity_still_blocks(tmp_path, change):
    with pytest.raises(RunProvenanceError, match="different"):
        resume(tmp_path, {**teacher_record(), "tools_sha256": "t", "context_window": 1}, {**teacher_record(), "tools_sha256": "t", "context_window": 1, **change})


def test_a_provenance_from_before_the_new_keys_still_resumes(tmp_path):
    """The 2026-09-30 sealed run's provenance has no effort, capture flag or tools hash, and a file hash that has since moved."""
    old = teacher_record()
    del old["teacher"]["declared"]["reasoning_effort"]
    del old["teacher"]["declared"]["capture_transcripts"]
    new = {**teacher_record(**{"declared.sha256": "file-2"}), "tools_sha256": "t"}
    assert resume(tmp_path, old, new) is not None


def test_a_resume_records_the_deployment_change_and_the_outages(park_world, tmp_path):
    out = tmp_path / "out"
    chosen = one_spec(park_world)
    with FakeAkhomed(park_akh) as peer, ReplayMcpGateway() as gateway:
        runner = waiting_teacher(peer, gateway, FakeTime())
        first = provenance(model_kind="acp", model_id=runner.model_id, **teacher_record())
        B.run_batch(park_world.store(), chosen, None, SYSTEM, StaticAncestryOracle({}), B.BatchConfig(out, 1), first,
                    "teacher", episode_runner=runner)
        (out / "verification.jsonl").write_text("")  # as if nothing had finished: the resume has work to do
        lines: list[str] = []
        second = provenance(model_kind="acp", model_id=runner.model_id, started_at="t2",
                            **teacher_record(**{"declared.sha256": "file-2"}))
        B.run_batch(park_world.store(), chosen, None, SYSTEM, StaticAncestryOracle({}), B.BatchConfig(out, 1), second,
                    "teacher", progress=lines.append, episode_runner=runner)
    record = json.loads((out / "provenance.json").read_text())
    assert record["deployment_history"] == [{"at": "t2", "changed": {
        "teacher.declared.sha256": {"was": '"file-1"', "now": '"file-2"'}}}]
    assert any("deployment-only changes" in line and "file-1" in line for line in lines)
