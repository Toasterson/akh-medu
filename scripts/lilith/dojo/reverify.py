"""Re-score a finished run against the current spec builders and verifiers, without calling a model.

    runner.py reverify --run DIR --root SNAPSHOTS [--out DIR2] [--repo .] [--illumos-pack FILE]
                       [--allow-prompt-drift] [--allow-dirty]

A run records what the model did (steps) and which receipts it got (ids, arguments, response digests), not the
receipt bodies. The bodies come from the replay: every recorded call is asked again, in order, of fresh replay
servers for the episode's snapshot, and an episode is scored only when each answer hashes to the recorded
digest, i.e. when the model would have read exactly the same bytes. Per episode:

- the same prompt and tool surface and the same label  -> re-verified;
- the same prompt and tool surface, a changed label (or spec) -> re-verified, `relabelled: true`;
- a changed prompt (task text, situation, answer schema) -> skipped, unless `--allow-prompt-drift` re-verifies
  it and records `prompt_drift: true` (a what-if: the model never saw the new instruction);
- a changed tool surface, replay bytes that differ now, an episode the builders no longer produce, or an
  episode the source run did not score -> skipped, with the reason.

Nothing in the source run is touched: results go to `--out` (default `<run>-reverify-<short sha>`), which must
be new or empty. `baseline-summary.json` there covers the re-verified episodes only; skipped ones are listed in
`reverify-skipped.jsonl`. `provenance.json` keeps the source run's provenance and adds `reverify` (source run,
the reverify commit, the current snapshot-root and prompt hashes, counts, and the per-family before/after table).
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dojo import DOJO_VERSION
from dojo import baseline as B
from dojo import contracts as C
from dojo.errors import DojoError
from dojo.episode import (
    EmitDocument,
    Episode,
    EpisodeSpec,
    Family,
    RecipientClass,
    SendMessage,
    StepBudget,
    StepRecord,
    ToolCall,
    Voice,
)
from dojo import memory_epistemics as ME
from dojo.families import BuildContext, build_all, load_illumos_pack
from dojo.replay_server import DojoLedger, GitAncestryOracle, servers_for_spec
from dojo.runner import Broker, _snapshot_now, compose_system_prompt, expected_contracts, prompt_bundle_sha256, user_prompt
from dojo.snapshot import SnapshotStore, sha256_hex

REVERIFY_SCHEMA = "lilith/dojo-reverify@1"
SKIPPED_FILE = "reverify-skipped.jsonl"


class SkipReason:
    NOT_SCORED = "not-scored-in-source"
    NOT_BUILT = "not-built-by-current-builders"
    NO_SNAPSHOT = "snapshot-not-under-root"
    TOOLS_CHANGED = "tool-surface-changed"
    PROMPT_CHANGED = "prompt-changed"
    NO_PROMPT_HASH = "prompt-hash-not-recorded"
    REPLAY_DIFFERS = "replay-answers-differ"
    UNREADABLE = "recorded-steps-unreadable"


@dataclasses.dataclass(frozen=True)
class Skipped:
    episode_id: str
    family: str
    reason: str
    detail: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# --- recorded steps back into step records ---------------------------------------------------------------------


def parse_step(record: Mapping[str, Any], episode_id: str) -> StepRecord:
    """The inverse of `StepRecord.to_dict` (`follows_correction` is not recorded: runs have no correction turns)."""
    action = record["action"]
    kind = action["kind"]
    if kind == "tool_call":
        parsed: Any = ToolCall(action["tool"], dict(action.get("arguments") or {}), action["receipt"],
                               action.get("server") or "anima")
    elif kind == "emit":
        parsed = EmitDocument(dict(action["document"]))
    elif kind == "send":
        parsed = SendMessage(action["recipient"], RecipientClass(action["recipient_class"]), action["protocol"],
                             dict(action["payload"]), int(action["sent_at"]), action.get("follow_up_at"))
    else:
        raise ValueError(f"unknown step kind {kind!r}")
    budget = StepBudget(**{k: int(v) for k, v in (record.get("budget") or {}).items()})
    voice = Voice(record["voice_shown"]) if record.get("voice_shown") else None
    return StepRecord(episode_id, int(record["step"]), parsed, budget, voice, record.get("observation_ref"),
                      record.get("why_this_move"))


def contract_status(spec: EpisodeSpec, steps: Sequence[StepRecord]) -> tuple[bool, list[str]]:
    """Did the episode answer with its family's (or variant's) contract, and do its documents pass the current
    schemas. Mirrors `runner.finish_answer`."""
    ok = False
    errors: list[str] = []
    wanted = expected_contracts(spec)
    for step in steps:
        action = step.action
        if isinstance(action, EmitDocument):
            found = C.validate_document(action.document)
            errors.extend(found)
            ok = ok or (action.document.get("contract") in wanted and not found)
        elif isinstance(action, SendMessage):
            schema = C.schema_for(action.protocol)
            found = C.validate(dict(action.payload), schema) if schema else [f"unknown protocol {action.protocol}"]
            errors.extend(found)
            ok = ok or (action.protocol in wanted and not found)
    return ok, errors


# --- one episode ------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Reverified:
    record: dict  # the new verification record
    relabelled: bool
    prompt_drift: bool


def reverify_episode(source: Mapping[str, Any], old: Mapping[str, Any], spec: EpisodeSpec | None, store: SnapshotStore,
                     oracle, ledger_dir: Path, allow_prompt_drift: bool) -> Reverified | Skipped:
    eid, family = source["episode_id"], source["family"]

    def skip(reason: str, detail: str = "") -> Skipped:
        return Skipped(eid, family, reason, detail)

    if old.get("status") != "ok":
        return skip(SkipReason.NOT_SCORED, str(old.get("error") or old.get("status")))
    if spec is None:
        return skip(SkipReason.NOT_BUILT)
    if spec.snapshot_id not in store.ids:
        return skip(SkipReason.NO_SNAPSHOT, spec.snapshot_id)
    snapshot = store.get(spec.snapshot_id)
    ledger = ledger_dir / f"{sha256_hex(eid.encode())[:16]}.jsonl"
    ledger.unlink(missing_ok=True)
    servers, log = servers_for_spec(spec, snapshot, DojoLedger(ledger, forbidden_roots=(store.root,)), oracle)
    broker = Broker(servers)
    if source.get("tools_sha256") and source["tools_sha256"] != broker.tools_sha256:
        return skip(SkipReason.TOOLS_CHANGED)
    if not source.get("user_prompt_sha256"):
        return skip(SkipReason.NO_PROMPT_HASH)
    prompt_now = sha256_hex(user_prompt(spec, _snapshot_now(servers)).encode("utf-8"))
    drift = prompt_now != source["user_prompt_sha256"]
    if drift and not allow_prompt_drift:
        return skip(SkipReason.PROMPT_CHANGED, "task text, situation or answer schema differs from what the model read")
    # Ask every recorded call again; the answers must hash to the recorded digests.
    for number, receipt in enumerate(source.get("receipts") or [], start=1):
        broker.call(receipt["server"], receipt["tool"], dict(receipt.get("arguments") or {}))
        issued = log.receipts[-1] if len(log.receipts) == number else None
        if issued is None or issued.call_id != receipt["call_id"] or issued.response_sha256 != receipt["response_sha256"]:
            return skip(SkipReason.REPLAY_DIFFERS, f"{receipt['call_id']} {receipt['tool']}")
    try:
        steps = tuple(parse_step(s, eid) for s in source.get("steps") or [])
    except (KeyError, ValueError, TypeError) as error:
        return skip(SkipReason.UNREADABLE, f"{type(error).__name__}: {error}")
    episode = Episode(spec, tuple(log.receipts), steps)
    contract_ok, schema_errors = contract_status(spec, steps)
    base = {"episode_id": eid, "family": family, "split": source.get("split") or old.get("split"),
            "status": "ok", "answer_status": old.get("answer_status"), "rounds": old.get("rounds"),
            "contract_ok": contract_ok, "schema_errors": schema_errors,
            "weight_policy": old.get("weight_policy"), "error": None}
    record = B.score_episode(base, spec, episode, snapshot, oracle)
    relabelled = source.get("spec_sha256") != spec.spec_sha256
    record |= {"relabelled": relabelled, "prompt_drift": drift, "previous_weight": old.get("weight"),
               "previous_hard": old.get("hard"), "previous_spec_sha256": source.get("spec_sha256"),
               "spec_sha256": spec.spec_sha256}
    return Reverified(record, relabelled, drift)


# --- the run ---------------------------------------------------------------------------------------------------------


def _short(sha: str) -> str:
    return sha[:8]


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def answer_effect(record: Mapping[str, Any]) -> float | None:
    """The family effect of the answer step, before any hard constraint zeroes the episode weight: it shows what a
    label or verifier change did to the answers even when a fabrication still holds the weight at 0."""
    steps = (record.get("verdict") or {}).get("steps") or []
    return next((s["vector"]["effect"] for s in reversed(steps) if s["vector"].get("effect") is not None), None)


def metric_means(old: Sequence[Mapping[str, Any]], new: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean of each family metric the old and the new records both carry, over the same episodes."""
    out: dict[str, dict[str, float]] = {}
    keys = sorted(set().union(*(r.get("metrics", {}).keys() for r in new)) & set().union(*(r.get("metrics", {}).keys() for r in old))
                  if new and old else [])
    for key in keys:
        pairs = [(o["metrics"][key], n["metrics"][key]) for o, n in zip(old, new)
                 if key in o.get("metrics", {}) and key in n.get("metrics", {})]
        if pairs:
            out[key] = {"before": statistics.fmean(p[0] for p in pairs), "after": statistics.fmean(p[1] for p in pairs)}
    return out


def family_table(source_summary: Mapping[str, Any], old: Mapping[str, Mapping[str, Any]],
                 new: Mapping[str, Mapping[str, Any]], skipped: Sequence[Skipped]) -> list[dict]:
    """Per family: the source run's mean over all its episodes, the same episodes' mean before, and after."""
    rows = []
    families = sorted({r["family"] for r in new.values()} | {s.family for s in skipped}
                      | set(source_summary.get("families", {})))
    for family in families:
        done = [e for e, r in new.items() if r["family"] == family]
        was = source_summary.get("families", {}).get(family, {})
        old_done = [old[e] for e in done]
        new_done = [new[e] for e in done]
        rows.append({
            "family": family,
            "effect_before": _mean([x for x in map(answer_effect, old_done) if x is not None]),
            "effect_after": _mean([x for x in map(answer_effect, new_done) if x is not None]),
            "metrics": metric_means(old_done, new_done),
            "source_n": was.get("n"), "source_mean": was.get("mean_weight"),
            "reverified_n": len(done),
            "before_same_episodes": _mean([float(old[e]["weight"]) for e in done if old[e].get("weight") is not None]),
            "after": _mean([float(new[e]["weight"]) for e in done]),
            "relabelled": sum(1 for e in done if new[e].get("relabelled")),
            "prompt_drift": sum(1 for e in done if new[e].get("prompt_drift")),
            "fabrication_before": sum("fabrication" in (old[e].get("hard") or []) for e in done),
            "fabrication_after": sum("fabrication" in (new[e].get("hard") or []) for e in done),
            "skipped": sum(1 for s in skipped if s.family == family),
        })
    return rows


def render_table(rows: Sequence[Mapping[str, Any]], overall: Mapping[str, Any]) -> str:
    def fmt(value: Any) -> str:
        return "-" if value is None else f"{value:.3f}"

    lines = [f"{'family':<20} {'src n':>5} {'src mean':>8} {'n':>4} {'before':>7} {'after':>7} {'eff b':>6} {'eff a':>6} "
             f"{'fab b>a':>8} {'relab':>5} {'drift':>5} {'skip':>4}"]
    for r in rows:
        lines.append(f"{r['family']:<20} {r['source_n'] if r['source_n'] is not None else '-':>5} {fmt(r['source_mean']):>8} "
                     f"{r['reverified_n']:>4} {fmt(r['before_same_episodes']):>7} {fmt(r['after']):>7} "
                     f"{fmt(r['effect_before']):>6} {fmt(r['effect_after']):>6} "
                     f"{str(r['fabrication_before']) + '>' + str(r['fabrication_after']):>8} {r['relabelled']:>5} "
                     f"{r['prompt_drift']:>5} {r['skipped']:>4}")
    lines.append(f"{'OVERALL':<20} {overall['source_n']:>5} {fmt(overall['source_mean']):>8} {overall['reverified_n']:>4} "
                 f"{fmt(overall['before_same_episodes']):>7} {fmt(overall['after']):>7}")
    return "\n".join(lines)


def run_reverify(run: Path, root: Path, repo: Path, out: Path | None, illumos_pack: Path | None = None,
                 allow_prompt_drift: bool = False, allow_dirty: bool = False, progress=lambda line: None,
                 memory_sessions: Path | None = None, memory_annotations: Path | None = ME.ANNOTATIONS_FILE) -> dict:
    run = Path(run)
    for name in (B.PROVENANCE_FILE, B.EPISODES_FILE, B.VERIFICATION_FILE, B.SUMMARY_FILE):
        if not (run / name).is_file():
            raise DojoError(f"{run} has no {name}", help="pass a finished run directory written by `runner.py run`")
    git = B.git_provenance(repo, allow_dirty)
    out = Path(out) if out else run.with_name(f"{run.name}-reverify-{_short(git['git_sha'])}")
    if out.resolve() == run.resolve():
        raise DojoError("--out is the source run", help="a reverify never overwrites the run it reads")
    if out.exists() and any(out.iterdir()):
        raise DojoError(f"{out} already holds files", help="give a new directory; a reverify is written once")
    out.mkdir(parents=True, exist_ok=True)
    source_prov = json.loads((run / B.PROVENANCE_FILE).read_text())
    source_summary = json.loads((run / B.SUMMARY_FILE).read_text())
    episodes = B.read_jsonl_latest(run / B.EPISODES_FILE)
    verifications = B.read_jsonl_latest(run / B.VERIFICATION_FILE)
    store = SnapshotStore(root)
    oracle = GitAncestryOracle(repo)
    families = [Family(f) for f in source_prov.get("families") or [f.value for f in Family]]
    pack = load_illumos_pack(illumos_pack) if illumos_pack else ()
    memory = ME.memory_world(memory_sessions, memory_annotations)
    built = build_all(store, BuildContext(oracle=oracle, illumos_pack=tuple(pack), memory=memory), families)
    current = {spec.episode_id: spec for res in built.values() for spec in res.specs}
    ledger_dir = out / "ledgers"
    ledger_dir.mkdir()
    new: dict[str, dict] = {}
    skipped: list[Skipped] = []
    for eid in sorted(episodes):
        source, old = episodes[eid], verifications.get(eid, {})
        result = reverify_episode(source, old, current.get(eid), store, oracle, ledger_dir, allow_prompt_drift)
        if isinstance(result, Skipped):
            skipped.append(result)
            progress(f"[skip] {eid}: {result.reason} {result.detail}".rstrip())
            continue
        new[eid] = result.record
        B._jsonl_append(out / B.EPISODES_FILE, _NO_LOCK, source | {"reverified_from": str(run)})
        B._jsonl_append(out / B.VERIFICATION_FILE, _NO_LOCK, result.record)
        weight = result.record["weight"]
        progress(f"[ok] {eid} {'-' if old.get('weight') is None else format(old['weight'], '.3f')} -> {weight:.3f}"
                 f"{' relabelled' if result.relabelled else ''}{' prompt-drift' if result.prompt_drift else ''}")
    with (out / SKIPPED_FILE).open("w", encoding="utf-8") as stream:
        for s in skipped:
            stream.write(json.dumps(s.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
    old_same = {e: verifications[e] for e in new}
    rows = family_table(source_summary, verifications, new, skipped)
    overall = {
        "source_n": source_summary.get("overall", {}).get("n"),
        "source_mean": source_summary.get("overall", {}).get("mean_weight"),
        "reverified_n": len(new),
        "before_same_episodes": _mean([float(v["weight"]) for v in old_same.values() if v.get("weight") is not None]),
        "after": _mean([float(r["weight"]) for r in new.values()]),
    }
    system = compose_system_prompt(repo)
    reasons = collections.Counter(s.reason for s in skipped)
    provenance = {
        **source_prov,
        "reverify": {
            "schema": REVERIFY_SCHEMA,
            "dojo": DOJO_VERSION,
            "source_run": str(run),
            "source_label": source_summary.get("label"),
            "source_git_sha": source_prov.get("git_sha"),
            "source_snapshot_root_sha256": source_prov.get("snapshot_root_sha256"),
            "source_prompt_sha256": source_prov.get("prompt_sha256"),
            "git_sha": git["git_sha"], "git_dirty": git["git_dirty"], "on_origin_main": git["on_origin_main"],
            "snapshot_root_sha256": B.snapshot_root_sha256(store),
            "prompt_sha256": prompt_bundle_sha256(system),
            "system_prompt_sha256": system.sha256,
            "allow_prompt_drift": allow_prompt_drift,
            "episodes_in_source": len(episodes), "reverified": len(new), "skipped": len(skipped),
            "relabelled": sum(1 for r in new.values() if r["relabelled"]),
            "prompt_drift": sum(1 for r in new.values() if r["prompt_drift"]),
            "skip_reasons": dict(reasons),
            "table": rows, "overall": overall,
        },
    }
    (out / B.PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    label = f"{source_summary.get('label') or run.name} (reverified {_short(git['git_sha'])})"
    summary = B.build_summary(out, provenance, label)
    (out / B.SUMMARY_FILE).write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return {"out": str(out), "table": rows, "overall": overall, "skip_reasons": dict(reasons), "reverified": len(new),
            "skipped": len(skipped), "git_sha": git["git_sha"]}


class _NoLock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


_NO_LOCK = _NoLock()


def cmd_reverify(args: argparse.Namespace) -> int:
    root = args.root
    if root is None:
        raise DojoError("--root is required", help="the snapshot root the run was made from (it may hold newer snapshots)")
    done = run_reverify(args.run, root, args.repo, args.out, args.illumos_pack, args.allow_prompt_drift,
                        args.allow_dirty, progress=lambda line: print(line, file=sys.stderr, flush=True),
                        memory_sessions=args.memory_sessions, memory_annotations=args.memory_annotations)
    print(render_table(done["table"], done["overall"]))
    print(f"\nreverified {done['reverified']}, skipped {done['skipped']} {done['skip_reasons']}\nwritten to {done['out']}")
    return 0


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", type=Path, required=True, help="a finished run directory (read-only source)")
    p.add_argument("--root", type=Path, default=None, help="snapshot root")
    p.add_argument("--repo", type=Path, default=B.REPO_ROOT, help="local akh-medu clone: ancestry oracle and provenance")
    p.add_argument("--out", type=Path, default=None, help="new output directory (default <run>-reverify-<short sha>)")
    p.add_argument("--illumos-pack", type=Path, default=None)
    p.add_argument("--memory-sessions", type=Path, default=None, help="the harvested C2 sessions (memory-epistemics seeds)")
    p.add_argument("--memory-annotations", type=Path, default=ME.ANNOTATIONS_FILE)
    p.add_argument("--allow-prompt-drift", action="store_true",
                   help="also re-score episodes whose prompt changed since the run (recorded as prompt_drift: a "
                        "what-if, the model never read the new instruction)")
    p.add_argument("--allow-dirty", action="store_true")
