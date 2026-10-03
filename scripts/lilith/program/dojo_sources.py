"""Dojo-derived records: teacher decisions (Brown), captain-selection renders, memory-epistemics corrected
siblings and rejected sides, and on-policy conversion.

A finished run keeps steps and receipt digests but not receipt bodies. Every observation here comes from
asking the recorded calls again of fresh replay servers over the recorder snapshots (read-only), and an
episode is used only when each answer hashes to the digest the run recorded: the model would have read
exactly those bytes. Weights are re-derived under the CURRENT dojo builders and verifiers
(`baseline.score_episode`, the same path `runner.py reverify` uses), so a label change since the run is
picked up rather than inherited.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

LILITH = Path(__file__).resolve().parents[1]
if str(LILITH) not in sys.path:
    sys.path.insert(0, str(LILITH))

from dojo import baseline as B  # noqa: E402
from dojo import contracts as C  # noqa: E402
from dojo import memory_epistemics as ME  # noqa: E402
from dojo.episode import EmitDocument, Episode, EpisodeSpec, Family, SendMessage, StepRecord, ToolCall  # noqa: E402
from dojo.families import BuildContext, build_all, load_illumos_pack  # noqa: E402
from dojo.replay_server import DojoLedger, GitAncestryOracle, servers_for_spec  # noqa: E402
from dojo.reverify import contract_status, parse_step  # noqa: E402
from dojo.runner import (DOJO_ADDENDUM_FILE, DOJO_SERVERS, MCP_SERVERS_PLACEHOLDER, Broker, SystemPrompt, _snapshot_now,  # noqa: E402
                         compose_system_prompt, parse_answer, tool_name, user_prompt)
from dojo.snapshot import SnapshotStore, sha256_hex as dojo_sha  # noqa: E402

from program import schema as S  # noqa: E402
from program.schema import assistant, assistant_calls, make_record, tool_message, user  # noqa: E402
from program.splits import uuids_in  # noqa: E402

MIN_WEIGHT = 0.7

# Stages of the CoS families. All teacher decisions are Brown (Choice): the captain picks the move and the voice.
FAMILY_STAGE = {f.value: "brown" for f in Family}
FAMILY_STAGE.update({"memory-epistemics": "white"})
ONPOLICY_FAMILIES = {
    "green": ("memory-epistemics", "read-failure", "scope", "correction", "park-triage"),
    "brown": ("briefing", "chase", "commitment", "operator-action", "pm-po-steering", "captain-selection"),
    "black": ("illumos-guidance",),
}
ONPOLICY_QUOTA = {"green": 120, "brown": 150, "black": 60}


@dataclasses.dataclass
class World:
    store: SnapshotStore
    oracle: GitAncestryOracle
    repo: Path
    specs: dict[str, EpisodeSpec]
    system: SystemPrompt
    ledger_dir: Path
    memory: Any
    until: str
    exclusions: list[dict]

    def servers(self, spec: EpisodeSpec, tag: str):
        snapshot = self.store.get(spec.snapshot_id)
        ledger = self.ledger_dir / f"{dojo_sha(f'{spec.episode_id}:{tag}'.encode())[:16]}.jsonl"
        ledger.unlink(missing_ok=True)
        servers, log = servers_for_spec(spec, snapshot, DojoLedger(ledger, forbidden_roots=(self.store.root,)), self.oracle)
        return servers, log, Broker(servers)


def open_world(snapshot_root: Path, repo: Path, until: str, memory_sessions: Path | None, illumos_pack: Path | None = None) -> World:
    store = SnapshotStore.open(Path(snapshot_root), until)
    oracle = GitAncestryOracle(repo)
    pack = load_illumos_pack(illumos_pack) if illumos_pack else ()
    memory = ME.memory_world(memory_sessions)
    built = build_all(store, BuildContext(oracle=oracle, illumos_pack=tuple(pack), memory=memory),
                      [f for f in Family if f is not Family.CORRECTION])
    specs = {s.episode_id: s for res in built.values() for s in res.specs}
    excluded = [{"family": fam.value, **e.to_dict()} for fam, res in built.items() for e in res.excluded]
    return World(store, oracle, Path(repo), specs, compose_system_prompt(repo), Path(tempfile.mkdtemp(prefix="program-ledgers-")),
                 memory, store.cutoff or until, excluded)


PROMPT_PROFILES = ("compact", "full")


def system_text(world: World, profile: str = "compact") -> str:
    """The system prompt of dojo-derived records. `full` is the dojo runner's own prompt (role + chief-of-staff
    procedures + dojo addendum, ~4.5k tokens: most episodes then exceed 8192 tokens). `compact` keeps the role
    file and the dojo addendum and leaves the procedures to be internalized by training."""
    if profile == "full":
        return world.system.text
    if profile != "compact":
        raise S.ProgramError(f"unknown dojo prompt profile {profile!r}", help=f"one of {PROMPT_PROFILES}")
    names = ", ".join(DOJO_SERVERS)
    role = (world.repo / "deploy" / "roman-mini" / "roles" / "lilith.md").read_text()
    addendum = (world.repo / DOJO_ADDENDUM_FILE).read_text()
    return (role.replace(MCP_SERVERS_PLACEHOLDER, names).rstrip("\n") + "\n\n" + addendum.replace(MCP_SERVERS_PLACEHOLDER, names))


def tool_declarations(broker: Broker, used: Sequence[str]) -> list[dict]:
    """OpenAI-shaped declarations of the tools an episode used (the full 82-tool surface alone is ~15k tokens)."""
    specs = {s.name: s for s in broker.tool_specs()}
    return [{"type": "function", "function": {"name": n, "description": specs[n].description, "parameters": specs[n].schema()}}
            for n in sorted(set(used)) if n in specs]


@dataclasses.dataclass
class Rendered:
    messages: list[dict]
    tools: list[dict]
    user_prompt_sha256: str
    verification: dict


class Skip(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason} {detail}".strip())
        self.reason, self.detail = reason, detail


def final_text(objects: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(json.dumps(o, ensure_ascii=False) for o in objects)


def _document_objects(steps: Sequence) -> list[dict]:
    out: list[dict] = []
    for s in steps:
        a = s.action
        if isinstance(a, EmitDocument):
            out.append(dict(a.document))
        elif isinstance(a, SendMessage):
            out.append({"kind": "message", "recipient": a.recipient, "recipient_class": a.recipient_class.value,
                        "protocol": a.protocol, "payload": dict(a.payload), "follow_up_at": a.follow_up_at})
    return out


def render_steps(world: World, spec: EpisodeSpec, steps: Sequence, recorded: Sequence[Mapping[str, Any]] | None,
                 answer_objects: Sequence[Mapping[str, Any]] | None, system_text: str, tag: str):
    """Replay `steps` against fresh servers and render the conversation. `recorded` carries the run's receipt
    digests to check (None for dojo-made siblings, which are served by the same servers they were played on)."""
    servers, log, broker = world.servers(spec, tag)
    now = _snapshot_now(servers)
    prompt = user_prompt(spec, now)
    texts: dict[str, str] = {}
    if recorded is not None:
        for number, rec in enumerate(recorded, start=1):
            d = broker.call(rec["server"], rec["tool"], dict(rec.get("arguments") or {}))
            issued = log.receipts[-1] if len(log.receipts) == number else None
            if issued is None or issued.call_id != rec["call_id"] or issued.response_sha256 != rec["response_sha256"]:
                raise Skip("replay-answers-differ", f"{rec['call_id']} {rec['tool']}")
            texts[rec["call_id"]] = d.text
    msgs = [S.system(system_text), user(prompt)]
    used: list[str] = []
    replayed = 0
    for st in steps:
        a = st.action
        if not isinstance(a, ToolCall):
            continue
        name = tool_name(a.server, a.tool)
        if recorded is None:
            d = broker.call(a.server, a.tool, dict(a.arguments))
            replayed += 1
            if d.receipt_id != a.receipt_id:
                raise Skip("sibling-receipt-mismatch", a.receipt_id)
            texts[a.receipt_id] = d.text
        if a.receipt_id not in texts:
            raise Skip("step-without-receipt", a.receipt_id)
        used.append(name)
        msgs.append(assistant_calls([(a.receipt_id, name, dict(a.arguments))]))
        msgs.append(tool_message(a.receipt_id, texts[a.receipt_id]))
    objects = list(answer_objects) if answer_objects else _document_objects(steps)
    if not objects:
        raise Skip("no-answer")
    msgs.append(assistant(final_text(objects)))
    return Rendered(msgs, tool_declarations(broker, used), dojo_sha(prompt.encode("utf-8")), {"replayed_calls": len(texts)}), \
        servers, log, broker


def score(world: World, spec: EpisodeSpec, steps: Sequence, log, answer_status: str) -> dict:
    episode = Episode(spec, tuple(log.receipts), tuple(steps))
    ok, errors = contract_status(spec, steps)
    base = {"episode_id": spec.episode_id, "family": spec.family.value, "split": "retrospective", "status": "ok",
            "answer_status": answer_status, "rounds": None, "contract_ok": ok, "schema_errors": errors, "error": None}
    return B.score_episode(base, spec, episode, world.store.get(spec.snapshot_id), world.oracle)


def obs_hashes(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    return S.tool_response_hashes(messages)


def _provenance(world: World, spec: EpisodeSpec, rec: Mapping[str, Any], extra: Mapping[str, Any]) -> dict:
    return {"subject_ids": sorted(uuids_in(list(spec.subject_ids))), "snapshot_id": spec.snapshot_id,
            "snapshot_until": world.until, "spec_sha256": spec.spec_sha256,
            "world_status": "retrospective_development",  # examined during development; never a sealed exam
            "verifier_weight": rec.get("weight"), "hard_flags": rec.get("hard", []),
            "step_weights": [w["weight"] for w in rec.get("step_weights", [])], "weight_policy": "cos-v1@1", **extra}


def _dojo_record(world: World, spec: EpisodeSpec, rendered: Rendered, rec: Mapping[str, Any], *, source: str, ref: Mapping[str, Any],
                 stage: str, weight: float, extra: Mapping[str, Any]) -> dict:
    return make_record(
        source=source, source_ref=ref, stage=stage, families=[spec.family.value], language=spec.language,
        voice=spec.voice_called_for.value if spec.voice_called_for else None, weight=weight, messages=rendered.messages,
        tools=rendered.tools, provenance=_provenance(world, spec, rec, {"observation_hashes": obs_hashes(rendered.messages), **extra}),
        verification={"structural": "episode replayed over recorded snapshots; every receipt matched its recorded digest"
                      if source == "teacher_dojo" else "dojo-played sibling over the replay servers",
                      "metrics": rec.get("metrics", {}), "answer_step": rec.get("answer_step"),
                      "user_prompt_sha256": rendered.user_prompt_sha256})


# -- teacher decisions --------------------------------------------------------------------------------------


def read_run(run: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    episodes, verifications = {}, {}
    for line in (run / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            episodes[r["episode_id"]] = r
    for line in (run / "verification.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            verifications[r["episode_id"]] = r
    return episodes, verifications


def teacher_records(world: World, runs: Sequence[Path], system_text: str | None = None, limit: int | None = None) -> tuple[list[dict], dict]:
    """Eligible decisions: stored status ok, then re-scored under the current labels: no hard flag, weight >= 0.7.
    Episodes are deduplicated by episode_id across runs (first run listed wins)."""
    system_text = system_text or world.system.text
    out: list[dict] = []
    seen: set[str] = set()
    report: dict[str, Any] = {"runs": {}, "skips": collections.Counter(), "relabelled": 0, "weight_changed": 0,
                              "eligible_by_stored_label": 0, "eligible_by_current_label": 0, "duplicates": 0}
    for run in runs:
        episodes, verifications = read_run(Path(run))
        stat = collections.Counter()
        for eid in sorted(episodes):
            src, old = episodes[eid], verifications.get(eid, {})
            stat["episodes"] += 1
            if eid in seen:
                report["duplicates"] += 1
                continue
            if src.get("status") != "ok" or old.get("status") != "ok":
                report["skips"]["source-not-ok"] += 1
                continue
            stored_ok = not old.get("hard") and float(old.get("weight") or 0) >= MIN_WEIGHT
            report["eligible_by_stored_label"] += int(stored_ok)
            spec = world.specs.get(eid)
            if spec is None:
                report["skips"]["not-built-by-current-builders"] += 1
                continue
            try:
                steps = [parse_step(s, eid) for s in src.get("steps") or []]
                transcript = [t for t in src.get("transcript") or [] if t.get("role") == "assistant" and t.get("content")]
                parsed = parse_answer(transcript[-1]["content"]).objects if transcript else ()
                rendered, _s, log, _b = render_steps(world, spec, steps, src.get("receipts") or [], parsed, system_text, "t")
                if rendered.user_prompt_sha256 != src.get("user_prompt_sha256"):
                    raise Skip("prompt-changed", "the user prompt differs from what the model read")
                record = score(world, spec, steps, log, src.get("answer_status") or "document")
            except Skip as skip:
                report["skips"][skip.reason] += 1
                continue
            except (KeyError, ValueError, TypeError) as error:
                report["skips"]["unreadable-steps"] += 1
                continue
            if src.get("spec_sha256") != spec.spec_sha256:
                report["relabelled"] += 1
            if abs(float(record["weight"]) - float(old.get("weight") or 0)) > 1e-9:
                report["weight_changed"] += 1
            if record["hard"] or float(record["weight"]) < MIN_WEIGHT:
                report["skips"]["below-admission-under-current-labels"] += 1
                continue
            report["eligible_by_current_label"] += 1
            seen.add(eid)
            stat["admitted"] += 1
            out.append(_dojo_record(world, spec, rendered, record, source="teacher_dojo",
                                    ref={"run": str(run), "episode_id": eid}, stage=FAMILY_STAGE[spec.family.value],
                                    weight=float(record["weight"]),
                                    extra={"run_provenance": str(Path(run) / "provenance.json"), "previous_weight": old.get("weight")}))
            if limit and len(out) >= limit:
                break
        report["runs"][str(run)] = dict(stat)
    report["skips"] = dict(report["skips"])
    return out, report


# -- captain-selection renders ----------------------------------------------------------------------------------


def captain_records(world: World, covered: set[str], system_text_: str | None = None) -> list[dict]:
    """Situation -> voice + reason for the specs the teacher did not answer. The voice is the dojo label; the
    reason is the structural feature code of that label (no prose is written here). The document must cite a
    receipt, so the sibling first reads the work item through the replay and cites that read; the dojo verifier
    then scores it like any episode (voice match, validity, provenance)."""
    system_text_ = system_text_ or world.system.text
    out = []
    for eid, spec in sorted(world.specs.items()):
        if spec.family is not Family.CAPTAIN_SELECTION or eid in covered:
            continue
        label = spec.label
        wi = spec.situation.get("work_item")
        if not wi:
            continue
        _servers, log, broker = world.servers(spec, "cap")
        broker.call("anima", "get_work_item_detail", {"id": wi})
        rc = log.receipts[-1] if log.receipts else None
        if rc is None or rc.is_error:
            continue
        ref = {"kind": "tool-call", "reference": rc.call_id, "observed_at": rc.observed_at, "digest": f"sha256:{rc.response_sha256}"}
        doc = {"contract": C.CAPTAIN_SELECTION, "language": spec.language, "voice": label["voice"],
               "reason": label["feature"], "receipts": [ref]}
        steps = [StepRecord(eid, 0, ToolCall("get_work_item_detail", {"id": wi}, rc.call_id, "anima")),
                 StepRecord(eid, 1, EmitDocument(doc))]
        try:
            rendered, _s, log2, _b = render_steps(world, spec, steps, None, None, system_text_, "caprender")
            rec = score(world, spec, steps, log2, "document")
        except Skip:
            continue
        weight = float(rec["weight"])
        if rec["hard"] or weight < MIN_WEIGHT:
            continue
        out.append(_dojo_record(world, spec, rendered, rec, source="dojo_sibling", ref={"episode_id": eid, "render": "label"},
                                stage="brown", weight=weight,
                                extra={"feature": label["feature"], "render": "label voice + structural feature code as reason"}))
    return out


# -- memory-epistemics: corrected siblings and rejected sides ---------------------------------------------------------


def note_sessions(world: World) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for note in world.memory.notes:
        out[note.episode_ref] = sorted({m.session for m in note.memories})
    return out


def spec_ref(spec: EpisodeSpec) -> str:
    return spec.subject_ids[0] if spec.subject_ids else ""


def raw_turn_messages(turn: Mapping[str, Any]) -> list[dict]:
    """The raw C2 turn as the rejected continuation: its own tool calls (live outputs) and its answer."""
    msgs: list[dict] = []
    for n, call in enumerate(turn.get("tool_calls") or [], 1):
        cid = f"r{n}"
        try:
            args = json.loads(call.get("args") or "{}")
        except json.JSONDecodeError:
            args = call.get("args") or {}
        msgs.append(assistant_calls([(cid, call["tool"], args)]))
        msgs.append(tool_message(cid, call.get("output") or ""))
    msgs.append(assistant(turn["agent_answer"]))
    return msgs


def memory_records(world: World, sessions_dir: Path, verdicts: Mapping[str, Any], system_text: str | None = None) -> tuple[list[dict], dict]:
    """Corrected siblings of the memory-epistemics episodes at their verifier weight, plus a preference record
    where a known-wrong raw C2 turn is the raw attempt (`raw_of`) of the sibling's episode."""
    system_text = system_text or world.system.text
    sessions = note_sessions(world)
    raw_for = {t["raw_of"]: t for t in verdicts["turns"] if t.get("raw_of")}
    out: list[dict] = []
    report = collections.Counter()
    for eid, spec in sorted(world.specs.items()):
        if spec.family is not Family.MEMORY_EPISTEMICS:
            continue
        report["episodes"] += 1
        servers, log, broker = world.servers(spec, "sib")
        sibling = ME.corrected_episode(spec, servers, log)
        steps = list(sibling.steps)
        try:
            rendered, _s, log2, _b = render_steps(world, spec, steps, None, None, system_text, "sibrender")
            rec = score(world, spec, steps, log2, "document")
        except Skip as skip:
            report["skip:" + skip.reason] += 1
            continue
        weight = float(rec["weight"])
        if rec["hard"] or weight <= 0:
            report["skip:sibling-not-earned"] += 1
            continue
        ref = spec_ref(spec)
        links = [f"c2:{s}" for s in sessions.get(ref, [])]
        extra = {"episode_ref": ref, "session_ids": sessions.get(ref, []), "link_keys": [], "round_basis": "c2-derived memory seeds"}
        extra["session_ids"] = sessions.get(ref, [])
        out.append(_dojo_record(world, spec, rendered, rec, source="c2_sibling", ref={"episode_ref": ref, "episode_id": eid},
                                stage="white", weight=weight, extra=extra))
        report["siblings"] += 1
        raw = raw_for.get(ref)
        if raw is None:
            report["no_raw_episode"] += 1
            continue
        path = Path(sessions_dir) / f"{raw['session']}.json"
        if not path.is_file():
            report["raw_session_not_cached"] += 1
            continue
        turn = next((t for t in json.loads(path.read_text())["turns"] if int(t["seq"]) == int(raw["seq"])), None)
        if turn is None:
            report["raw_turn_missing"] += 1
            continue
        context = rendered.messages[:2]
        chosen = rendered.messages
        rejected = [*context, *raw_turn_messages(turn)]
        out.append(make_record(
            source="c2_sibling", source_ref={"episode_ref": ref, "raw_turn": {"session": raw["session"], "seq": raw["seq"]}},
            stage="white", families=["memory-epistemics"], voice=spec.voice_called_for.value if spec.voice_called_for else None,
            language=spec.language, kind="preference", weight=weight, chosen_messages=chosen, rejected_messages=rejected,
            tools=rendered.tools,
            provenance=_provenance(world, spec, rec, {"observation_hashes": obs_hashes(chosen) + obs_hashes(rejected),
                                                      "session_ids": [raw["session"], *sessions.get(ref, [])],
                                                      "rejected_weight": 0.0, "rejected_reason": raw["reason"],
                                                      "pair_basis": "raw C2 turn (known wrong) vs corrected sibling, shared dojo prompt"}),
            verification={"structural": "sibling-contrast pair; the rejected side trains at weight 0 and only as the dispreferred side"}))
        report["pairs"] += 1
    return out, dict(report)


# -- on-policy conversion -------------------------------------------------------------------------------------------


def onpolicy_records(world: World, run: Path, stage: str, system_text: str | None = None, siblings: Path | None = None) -> tuple[list[dict], dict]:
    """A finished dojo run into stage records: raw episodes at their recorded weight, a corrected sibling (from
    `siblings.jsonl` in the run, or `corrected_episode` for memory-epistemics) as SFT plus a sibling-contrast
    pair when the raw episode failed. Families outside the stage's list are ignored; the quota caps the rest."""
    system_text = system_text or world.system.text
    allowed = ONPOLICY_FAMILIES[stage]
    episodes, verifications = read_run(Path(run))
    sib_rows: dict[str, dict] = {}
    sib_path = siblings or (Path(run) / "siblings.jsonl")
    if sib_path.is_file():
        for line in sib_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                sib_rows[r["episode_id"]] = r
    out: list[dict] = []
    report: collections.Counter = collections.Counter()
    for eid in sorted(episodes, key=lambda e: dojo_sha(f"0:{e}".encode())):
        src, old = episodes[eid], verifications.get(eid, {})
        if src.get("family") not in allowed:
            report["family_outside_stage"] += 1
            continue
        spec = world.specs.get(eid)
        if spec is None or src.get("status") != "ok" or old.get("status") != "ok":
            report["skip_unbuilt_or_not_ok"] += 1
            continue
        try:
            steps = [parse_step(s, eid) for s in src.get("steps") or []]
            transcript = [t for t in src.get("transcript") or [] if t.get("role") == "assistant" and t.get("content")]
            parsed = parse_answer(transcript[-1]["content"]).objects if transcript else ()
            raw_msgs, _s, _l, _b = render_steps(world, spec, steps, src.get("receipts") or [], parsed, system_text, "op")
        except (Skip, KeyError, ValueError, TypeError) as error:
            report["skip_replay"] += 1
            continue
        weight = float(old.get("weight") or 0.0)
        failed = bool(old.get("hard")) or weight <= 0
        sibling_msgs, sib_weight, sib_rec = None, 0.0, None
        if failed:
            if eid in sib_rows:
                row = sib_rows[eid]
                try:
                    sst = [parse_step(s, eid) for s in row["steps"]]
                    sibling_msgs, _s2, slog, _b2 = render_steps(world, spec, sst, row.get("receipts") or [], None, system_text, "ops")
                    sib_rec = score(world, spec, sst, slog, "document")
                except (Skip, KeyError, ValueError, TypeError):
                    report["sibling_unreadable"] += 1
            elif spec.family is Family.MEMORY_EPISTEMICS:
                servers, log, _b = world.servers(spec, "opsib")
                sib = ME.corrected_episode(spec, servers, log)
                sibling_msgs, _s2, slog, _b2 = render_steps(world, spec, list(sib.steps), None, None, system_text, "opsib2")
                sib_rec = score(world, spec, list(sib.steps), slog, "document")
            if sibling_msgs is not None and sib_rec and not sib_rec["hard"] and sib_rec["weight"] > 0:
                sib_weight = float(sib_rec["weight"])
            else:
                sibling_msgs = None
        common = {"run": str(run), "episode_id": eid, "on_policy_weight": weight, "run_hard": old.get("hard", [])}
        if not failed and weight >= MIN_WEIGHT:
            out.append(_dojo_record(world, spec, raw_msgs, old, source="onpolicy", ref=common, stage=stage, weight=weight,
                                    extra={"role": "on-policy episode"}))
            report["sft_raw"] += 1
        if sibling_msgs is not None:
            out.append(_dojo_record(world, spec, sibling_msgs, sib_rec, source="onpolicy", ref={**common, "role": "corrected-sibling"},
                                    stage=stage, weight=sib_weight, extra={"role": "corrected sibling"}))
            report["sft_sibling"] += 1
            if sib_weight - weight >= 0.1:
                chosen, rejected = sibling_msgs.messages, raw_msgs.messages
                out.append(make_record(
                    source="onpolicy", source_ref={**common, "role": "sibling-contrast"}, stage=stage, families=[spec.family.value],
                    language=spec.language, voice=spec.voice_called_for.value if spec.voice_called_for else None, kind="preference",
                    weight=sib_weight, chosen_messages=chosen, rejected_messages=rejected, tools=sibling_msgs.tools,
                    provenance=_provenance(world, spec, sib_rec, {"observation_hashes": obs_hashes(chosen) + obs_hashes(rejected),
                                                                  "rejected_weight": weight, "margin": sib_weight - weight}),
                    verification={"structural": "same situation; chosen = corrected sibling, rejected = the raw on-policy episode"}))
                report["pairs"] += 1
    quota = ONPOLICY_QUOTA[stage]
    sft = [r for r in out if r["kind"] == "sft"]
    keep_ids = {r["record_id"] for r in sft[:quota]}
    kept = [r for r in out if r["kind"] == "preference" or r["record_id"] in keep_ids]
    report["over_quota_sft"] = len(sft) - len(keep_ids)
    return kept, dict(report)
