import collections
import json

from fakes import block, responder

from program.generate import canon, cli, guidance, lineage, records, sparring, variation
from program.generate.acp_worker import FakeTeacher, TeacherError, run_pool
from program.generate.checks import CitationResolver, LanguageProfiles
from program.generate.tasking import Context

REPO = cli.REPO_ROOT


def run_all(tasks, responder_fn=responder, workers=3):
    """`workers` fake akhs, one worker each."""
    outcomes = []
    run_pool(tasks, [lambda: FakeTeacher(responder_fn) for _ in range(workers)],
             lambda teacher, task: outcomes.append(task.run(teacher)))
    return outcomes


# --- variation ---------------------------------------------------------------------------------------------


def test_variation_plan_meets_the_quota_and_balances_katas():
    jobs = variation.plan_jobs(variation.load_seeds(cli.DRILLS))
    assert len(jobs) == 240
    by_variant = collections.Counter(j.variant for j in jobs)
    assert by_variant == {"paraphrase": 120, "paraphrase-2": 40, "multi-turn": 40, "budget-pressure": 40}
    assert len({j.key for j in jobs}) == 240
    for variant in by_variant:
        per_kata = collections.Counter(j.seed.kata for j in jobs if j.variant == variant)
        assert set(per_kata.values()) == {by_variant[variant] // 8}
    assert all(j.language != j.seed.language for j in jobs)
    assert all(j.seed.belt == "white" for j in jobs)
    second = {j.seed.id: j.language for j in jobs if j.variant == "paraphrase-2"}
    first = {j.seed.id: j.language for j in jobs if j.variant == "paraphrase"}
    assert all(second[s] != first[s] for s in second)


def test_variation_records_follow_the_format(ctx):
    tasks = variation.plan(ctx, cli.DRILLS)
    outcomes = run_all(tasks)
    assert len(outcomes) == 240 and all(o.status == "done" for o in outcomes)
    rows = ctx.run.read("variation.jsonl")
    assert len(rows) == 240 and not ctx.run.read("variation.excluded.jsonl")
    row = rows[0]
    assert row["schema"] == "lilith/program-record@1" and row["source"] == "variation" and row["stage"] == "yellow"
    assert row["kind"] == "sft" and row["split"] == "train" and row["weight"] == 1.0
    assert row["voice"] in canon.VOICE_IDS and row["kata"] in canon.KATA_IDS and row["language"] in canon.LANGUAGES
    assert row["source_ref"]["teacher"] == "dojo-teacher/glm-5.3" and row["source_ref"]["seed_record_id"] == row["component_id"]
    assert [m["role"] for m in row["messages"]][:2] == ["system", "user"] and row["messages"][-1]["role"] == "assistant"
    assert records.sha256_text(records.canonical_json({k: v for k, v in row.items() if k != "record_id"})) == row["record_id"]
    assert row["verification"]["checks_failed"] == [] and row["provenance"]["teacher_transcript_sha256"]
    multi = [r for r in rows if r["provenance"]["variant"] == "multi-turn"]
    assert all(len(r["messages"]) == 5 for r in multi)  # system + 4 turns
    # per-record transcripts exist
    assert len(list((ctx.run.out / "transcripts" / "variation").glob("*.json"))) == 240


def test_variation_is_resumable(ctx):
    tasks = variation.plan(ctx, cli.DRILLS)
    run_all(tasks[:50])
    assert len(variation.plan(ctx, cli.DRILLS)) == 190
    run_all(variation.plan(ctx, cli.DRILLS))
    assert len(ctx.run.read("variation.jsonl")) == 240
    assert variation.plan(ctx, cli.DRILLS) == []


def test_variation_excludes_failed_structural_checks(ctx):
    def bad(prompt):
        reply = json.loads(responder(prompt).split("```json\n")[1].split("\n```")[0])
        reply["kata"] = "silence" if reply["kata"] != "silence" else "mirror"  # wrong kata echo
        return block(reply)

    tasks = variation.plan(ctx, cli.DRILLS)[:3]
    outcomes = run_all(tasks, bad)
    assert all(o.status == "excluded" and "kata_echo" in o.failed_checks for o in outcomes)
    excluded = ctx.run.read("variation.excluded.jsonl")
    assert len(excluded) == 3 and all(r["weight"] == 0.0 for r in excluded)
    assert not ctx.run.read("variation.jsonl")
    assert variation.plan(ctx, cli.DRILLS) != [] and len(variation.plan(ctx, cli.DRILLS)) == 237  # excluded seeds are finished
    ctx.retry_excluded = True
    assert len(variation.plan(ctx, cli.DRILLS)) == 240


def test_variation_retry_uses_feedback_then_succeeds(ctx):
    seen = []

    def flaky(prompt):
        seen.append(prompt)
        return "no header at all" if "failed these structural checks" not in prompt else responder(prompt.split("\n\nYour previous")[0])

    outcome = run_all(variation.plan(ctx, cli.DRILLS)[:1], flaky, workers=1)[0]
    assert outcome.status == "done" and len(seen) == 2 and "json_header" in seen[1]


def test_variation_budget_must_be_shorter_and_multi_turn_needs_anchor(ctx):
    jobs = variation.plan_jobs(variation.load_seeds(cli.DRILLS))
    budget = next(j for j in jobs if j.variant == "budget-pressure")
    long_reply = responder(variation.build_prompt(budget)).replace('"ok"', json.dumps("x " * 2000))
    assessment = variation.assess(budget, ctx, long_reply, "p")
    assert "budget_shorter_than_seed" in [c.name for c in assessment.checks if not c.passed]
    multi = next(j for j in jobs if j.variant == "multi-turn")
    reply = json.loads(responder(variation.build_prompt(multi)).split("```json\n")[1].split("\n```")[0])
    reply["memory_anchor"] = "not in the first turn"
    assessment = variation.assess(multi, ctx, block(reply), "p")
    assert "memory_anchor_in_first_turn" in [c.name for c in assessment.checks if not c.passed]


# --- sparring and its paired lineage --------------------------------------------------------------------------


def test_sparring_plan_and_pairs(ctx):
    jobs = sparring.plan_jobs()
    assert len(jobs) == 120 and collections.Counter(j.adversary.id for j in jobs) == {a.id: 20 for a in canon.ADVERSARIES}
    assert {j.messages for j in jobs} == {4, 6} and {j.language for j in jobs} == set(canon.LANGUAGES)
    assert all(set(j.katas) <= set(j.adversary.katas) for j in jobs)
    outcomes = run_all(sparring.plan(ctx))
    assert all(o.status == "done" for o in outcomes)
    spar = ctx.run.read("sparring.jsonl")
    pairs = [r for r in ctx.run.read("lineage.jsonl") if r["source_ref"]["generator"].endswith(":pair")]
    assert len(spar) == 120 and len(pairs) == 120
    row = spar[0]
    assert row["stage"] == "green" and row["source"] == "sparring" and len(row["messages"]) - 1 in (4, 6)
    # the three commentary fields stay out of the sft target
    text = json.dumps(row["messages"])
    assert "why " not in text and "shadow " not in text
    pair = next(p for p in pairs if p["source_ref"]["seed_record_id"] == row["record_id"])
    assert pair["stage"] == "black" and pair["source"] == "lineage" and pair["component_id"] == row["component_id"]
    assert all(f"{k}:" in pair["messages"][-1]["content"] for k in (*lineage.FOUR_FIELDS, "shadow_avoided"))


def test_sparring_rejects_short_or_misordered_dialogue(ctx):
    job = sparring.plan_jobs()[0]
    reply = json.loads(responder(sparring.build_prompt(job)).split("```json\n")[1].split("\n```")[0])
    reply["dialogue"] = reply["dialogue"][:2]
    names = [c.name for c in sparring.assess(job, ctx, block(reply), "p").checks if not c.passed]
    assert "turns_at_least_3" in names and "turns_count" in names
    reply["dialogue"] = [{"role": "assistant", "content": "x"}] * 4
    assert "alternating_roles" in [c.name for c in sparring.assess(job, ctx, block(reply), "p").checks if not c.passed]
    reply["kata_under_test"] = ["not-a-kata"]
    assert "kata_ids_valid" in [c.name for c in sparring.assess(job, ctx, block(reply), "p").checks if not c.passed]


# --- lineage -----------------------------------------------------------------------------------------------


def test_lineage_drill_seeds_are_balanced_and_checked(ctx):
    seeds = lineage.drill_seeds(cli.DRILLS)
    assert len(seeds) == 60 and all(s.kata for s in seeds)
    counts = collections.Counter(s.kata for s in seeds)
    assert set(counts) == canon.KATA_IDS and max(counts.values()) - min(counts.values()) <= 1
    outcomes = run_all(lineage.plan(ctx, cli.DRILLS, ctx.run.out / "guidance.jsonl", origins=("kata-drill",)))
    assert all(o.status == "done" for o in outcomes)
    row = ctx.run.read("lineage.jsonl")[0]
    assert row["stage"] == "black" and row["source"] == "lineage" and row["voice"] in canon.VOICE_IDS
    assert all(f"{k}:" in row["messages"][-1]["content"] for k in lineage.FOUR_FIELDS)


def test_lineage_requires_naming_and_all_four_fields(ctx):
    seed = lineage.drill_seeds(cli.DRILLS)[0]
    reply = json.loads(responder(lineage.build_prompt(seed)).split("```json\n")[1].split("\n```")[0])
    reply["why_this_move"] = "a lesson that names nothing"
    assert "names_voice" in [c.name for c in lineage.assess(seed, ctx, block(reply), "p").checks if not c.passed]
    reply["observed_effect_on_person"] = ""
    assert "four_fields_present" in [c.name for c in lineage.assess(seed, ctx, block(reply), "p").checks if not c.passed]
    reply["voice"] = "not-a-voice"
    names = [c.name for c in lineage.assess(seed, ctx, block(reply), "p").checks if not c.passed]
    assert "voice_canonical" in names and "voice_echo" in names


# --- guidance ----------------------------------------------------------------------------------------------


def candidate(**over):
    base = dict(work_item_id="wi-1", code="WI-1", title="[illumos gate - Bug #1] thing", description="d" * 200, project_id="p",
                snapshot_id="20261001T213729Z", observed_at=1_790_000_000, detail_sha256="ab" * 32, language="en", audience="implementer")
    base.update(over)
    return guidance.Candidate(**base)


def resolver_ok():
    return CitationResolver(None, get=lambda url: (200, "<title>illumos: manual page: svcadm.8</title>") if "svcadm" in url else (404, ""))


def test_guidance_admits_resolved_findings_and_builds_the_pack(tmp_path):
    ctx = Context(records.RunDir(tmp_path), LanguageProfiles({}), resolver_ok(), REPO)
    cand = candidate()
    outcome = run_all(guidance.plan(ctx, [cand]), workers=1)[0]
    assert outcome.status == "done"
    row = ctx.run.read("guidance.jsonl")[0]
    assert row["stage"] == "black" and row["source_ref"]["work_item_id"] == "wi-1" and row["component_id"] == "wi-1"
    finding = json.loads(row["messages"][-1]["content"])
    assert finding["contract"] == "lilith/cos-guidance@1" and finding["subject"] == {"kind": "work-item", "id": "wi-1"}
    assert finding["receipts"][0]["digest"] == "ab" * 32 and finding["citations"][0]["reference"] == "svcadm(8)"
    assert "OpenIndiana VM" in row["provenance"]["native_check"]
    assert guidance.pack_entries([row]) == [{"work_item_id": "wi-1", "subsystem": "smf", "source": "panel"}]


def test_guidance_excludes_unresolved_citations(tmp_path):
    ctx = Context(records.RunDir(tmp_path), LanguageProfiles({}), CitationResolver(None, get=lambda url: (404, "")), REPO)
    outcome = run_all(guidance.plan(ctx, [candidate()]), workers=1)[0]
    assert outcome.status == "excluded" and "citations_resolve" in outcome.failed_checks
    excluded = ctx.run.read("guidance.excluded.jsonl")
    assert excluded[0]["weight"] == 0.0 and not ctx.run.read("guidance.jsonl")
    assert excluded[0]["provenance"]["citation_resolutions"][0]["method"] == "not-found"


def test_guidance_rejects_schema_violations(ctx):
    cand = candidate()
    reply = json.loads(responder(guidance.build_prompt(cand)).split("```json\n")[1].split("\n```")[0])
    for field, value, check in (("subsystem", "kernel", "subsystem_enum"), ("audience", "product-owner", "audience_requested"),
                                ("citations", [{"kind": "vm-check", "reference": "x", "extra": 1}], "citations_shape"),
                                ("surprise", 1, "no_unknown_keys")):
        broken = dict(reply, **{field: value})
        assert check in [c.name for c in guidance.assess(cand, ctx, block(broken), "p").checks if not c.passed]


def test_guidance_waves_stop_at_the_quota(tmp_path):
    ctx = Context(records.RunDir(tmp_path), LanguageProfiles({}), resolver_ok(), REPO)
    cands = [candidate(work_item_id=f"wi-{i}", code=f"WI-{i}") for i in range(10)]
    progress = cli.Progress(lambda line: None)
    cli.run_guidance_waves(ctx, cands, 4, [lambda: FakeTeacher(responder)] * 2, progress)
    assert len(ctx.run.read("guidance.jsonl")) == 4
    assert cli.write_pack(ctx) == 4
    pack = cli.guidance.pack_entries(ctx.run.read("guidance.jsonl"))
    assert all(set(p) == {"work_item_id", "subsystem", "source"} for p in pack)


def test_teacher_errors_are_not_recorded_and_retry_on_resume(ctx):
    def down(prompt):
        raise TeacherError("connection refused", retryable=True)

    outcomes = run_all(variation.plan(ctx, cli.DRILLS)[:2], down, workers=1)
    assert [o.status for o in outcomes] == ["error", "error"]
    assert len(variation.plan(ctx, cli.DRILLS)) == 240  # errors are not finished seeds


def cli_args(out, **over):
    base = dict(out=str(out), order="variation", akhs="dojo-teacher", guidance_quota=0, snapshots="", snapshot_cutoff="20261002T000000Z",
                gate=None, acp_bin=None, retry_excluded=False, allow_any_out=True)
    base.update(over)
    return cli.argparse.Namespace(**base)


# --- end to end ------------------------------------------------------------------------------------------------


def test_cli_run_end_to_end_without_network(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "language_profiles", lambda: LanguageProfiles({}))
    out = tmp_path / "out"
    args = cli_args(out, order="variation,sparring,lineage", akhs="dojo-teacher,dojo-teacher-2,dojo-teacher-3", guidance_quota=0,
                    snapshots=str(tmp_path))
    assert cli.run(args, make_teacher=lambda akh: FakeTeacher(responder)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["admitted"]["variation"] == 240 and report["admitted"]["sparring"] == 120
    assert report["lineage_by_generator"]["program.generate.lineage@1"] == 60
    assert report["lineage_by_generator"]["program.generate.sparring@1:pair"] == 120
    assert report["errors_pending"] == {k: 0 for k in records.KINDS}


def test_cli_refuses_outputs_outside_the_private_root(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "language_profiles", lambda: LanguageProfiles({}))
    args = cli_args(tmp_path / "repo-data", order="variation", allow_any_out=False)
    assert cli.run(args, make_teacher=lambda akh: FakeTeacher(responder)) == 2
    assert not (tmp_path / "repo-data").exists()


# --- retry classification, backoff, per-akh workers, order, error handling -----------------------------------------


def test_busy_workspace_and_refusal_are_retryable_other_errors_are_not():
    from program.generate import acp_worker as aw

    busy = aw.classify_rpc_error("session/prompt", {"code": -32603, "message": "Internal error",
                                                   "data": 'workspace busy: a previous turn is still running for "dojo-teacher" (running 516s)'})
    assert busy.retryable and busy.kind == aw.BUSY
    boom = aw.classify_rpc_error("session/prompt", {"code": -32603, "message": "Internal error", "data": "index out of bounds"})
    assert not boom.retryable and boom.kind == aw.OTHER
    assert not aw.classify_rpc_error("session/new", {"code": -32601, "message": "method not found"}).retryable
    assert aw.TeacherError("connection refused", retryable=True).kind == aw.TRANSPORT


def test_backoff_is_exponential_capped_jittered_and_50_attempts():
    from program.generate.acp_worker import Backoff

    delays = Backoff().delays(rand=lambda: 0.5)  # zero jitter
    assert len(delays) == 50 and delays[0] == 5.0 and delays[1] == 8.0 and max(delays) == 60.0
    low, high = Backoff().delays(rand=lambda: 0.0), Backoff().delays(rand=lambda: 1.0)
    assert low[0] == 3.75 and high[0] == 6.25


def test_teacher_retries_busy_without_dropping_the_connection_and_gives_up_after_the_cap():
    from program.generate import acp_worker as aw

    sleeps, retries = [], []
    teacher = aw.AcpTeacher(backoff=aw.Backoff(attempts=3), sleep=sleeps.append, on_retry=lambda k, d, n: retries.append((k, n)))
    state = {"calls": 0, "drops": 0}

    def once(prompt):
        state["calls"] += 1
        if state["calls"] <= 2:
            raise aw.TeacherError("workspace busy", retryable=True, kind=aw.BUSY)
        return aw.TeacherReply("ok", 0.0)

    teacher._once = once
    teacher._drop = lambda: state.__setitem__("drops", state["drops"] + 1)
    assert teacher.ask("p").text == "ok"
    assert state == {"calls": 3, "drops": 0} and retries == [("busy", 1), ("busy", 2)] and len(sleeps) == 2

    def refused(prompt):
        raise aw.TeacherError("connection refused", retryable=True)

    teacher._once = refused
    try:
        teacher.ask("p")
        raise AssertionError("expected TeacherError")
    except aw.TeacherError as error:
        assert error.retryable and state["drops"] == 4  # dropped on every transport failure; gave up after 3 retries

    def fatal(prompt):
        raise aw.TeacherError("bad request")

    teacher._once, slept = fatal, len(sleeps)
    try:
        teacher.ask("p")
    except aw.TeacherError as error:
        assert not error.retryable and len(sleeps) == slept  # not retried


def test_one_worker_per_akh_and_each_akh_runs_one_turn_at_a_time():
    import threading

    active, peak, handled = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    lock = threading.Lock()

    class Akh:
        def __init__(self, name):
            self.name = name

        def ask(self, prompt):
            with lock:
                active[self.name] += 1
                peak[self.name] = max(peak[self.name], active[self.name])
                handled[self.name].append(threading.get_ident())
            with lock:
                active[self.name] -= 1
            return FakeTeacher(responder).ask(prompt)

        def close(self):
            pass

    names = ["dojo-teacher", "dojo-teacher-2", "dojo-teacher-3"]
    tasks = variation.plan(records_ctx(), cli.DRILLS)[:30]
    run_pool(tasks, [lambda n=n: Akh(n) for n in names], lambda teacher, task: task.run(teacher))
    assert set(handled) == set(names) and all(len(set(ids)) == 1 for ids in handled.values())  # one thread per akh
    assert all(peak[n] == 1 for n in names)
    assert sum(len(v) for v in handled.values()) == 30


def records_ctx():
    import tempfile
    from pathlib import Path

    return Context(records.RunDir(Path(tempfile.mkdtemp())), LanguageProfiles({}), CitationResolver(None, get=lambda url: (404, "")), REPO)


def test_cli_validates_akhs_and_order(tmp_path):
    assert cli.validate(cli_args(tmp_path, akhs="a,a")) == "--akhs needs one or more distinct akh names, got ['a', 'a']"
    assert "at most 6" in cli.validate(cli_args(tmp_path, akhs="a,b,c,d,e,f,g"))
    assert "--order" in cli.validate(cli_args(tmp_path, order="guidance,nonsense"))
    assert cli.validate(cli_args(tmp_path, akhs="a,b", order=",".join(cli.DEFAULT_ORDER))) == (["a", "b"], list(cli.DEFAULT_ORDER))
    assert cli.DEFAULT_ORDER == ("guidance", "sparring", "lineage", "variation")


def test_cli_runs_kinds_in_the_requested_order_and_akh_names_reach_the_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "language_profiles", lambda: LanguageProfiles({}))
    seen = []

    def make(akh):
        def respond(prompt):
            seen.append((akh, "sparring" if "sparring round" in prompt else "transmit" if "transmitting a technique" in prompt else "variation"))
            return responder(prompt)

        return FakeTeacher(respond)

    args = cli_args(tmp_path / "o", order="sparring,variation", akhs="dojo-teacher,dojo-teacher-2")
    assert cli.run(args, make_teacher=make) == 0
    kinds = [k for _, k in seen]
    assert kinds.index("variation") > max(i for i, k in enumerate(kinds) if k == "sparring")  # sparring finished first
    assert {a for a, _ in seen} == {"dojo-teacher", "dojo-teacher-2"} and "transmit" not in kinds


def test_exhausted_seeds_stop_the_run_and_error_lines_are_dropped_on_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "language_profiles", lambda: LanguageProfiles({}))

    def dead(prompt):
        raise TeacherError("workspace busy", retryable=True, kind="busy")

    args = cli_args(tmp_path / "o", order="variation,sparring")
    assert cli.run(args, make_teacher=lambda akh: FakeTeacher(dead)) == 1  # stopped by the consecutive-error breaker
    run_dir = records.RunDir(tmp_path / "o")
    errors = [v for v in run_dir.progress("variation").values() if v["status"] == "error"]
    assert len(errors) == cli.MAX_CONSECUTIVE_ERRORS and errors[0]["error_kind"] == "busy"
    assert not run_dir.progress("sparring")  # never reached
    # restart: the errored seeds are not finished, their error lines are dropped, and the run completes
    assert cli.run(cli_args(tmp_path / "o", order="variation"), make_teacher=lambda akh: FakeTeacher(responder)) == 0
    assert len(records.RunDir(tmp_path / "o").read("variation.jsonl")) == 240
    assert all(v["status"] != "error" for v in records.RunDir(tmp_path / "o").progress("variation").values())


def test_guidance_waves_terminate_when_candidates_keep_erroring(tmp_path):
    ctx = Context(records.RunDir(tmp_path), LanguageProfiles({}), resolver_ok(), REPO)
    cands = [candidate(work_item_id=f"wi-{i}", code=f"WI-{i}") for i in range(30)]
    progress = cli.Progress(lambda line: None, max_consecutive_errors=1000)  # breaker off: termination must not depend on it

    def down(prompt):
        raise TeacherError("busy", retryable=True, kind="busy")

    cli.run_guidance_waves(ctx, cands, 5, [lambda: FakeTeacher(down)], progress)  # returns instead of looping forever
    assert len(progress.errored) == 30 and not ctx.run.read("guidance.jsonl")
