"""Export the mixed, tokenized training archive and the frozen stage manifest of the guided Lilith program.

    cd scripts/lilith
    python3 -m program.export --out ~/lilith-private/program-20261003/corpus [--generated-dir D]
                              [--snapshot-root R] [--memory-sessions S]
    python3 -m program.export --out ... --onpolicy RUN_DIR --stage green

Needs the pinned tokenizer stack (transformers 5.10.4, tokenizers 0.22.2, jinja2 3.1.6), e.g.
    uv run --no-project --python 3.12 --with transformers==5.10.4 --with tokenizers==0.22.2 --with jinja2==3.1.6 \\
        --with blake3 python -m program.export --out ...
Tests: the same, plus --with pytest, then `python -m pytest program/tests -q`.

Idempotent and deterministic (seed 0): the same inputs write the same bytes. Private data stays under the
output root; nothing here writes into the repository.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

LILITH = Path(__file__).resolve().parents[1]
if str(LILITH) not in sys.path:
    sys.path.insert(0, str(LILITH))

from program import coverage as COV  # noqa: E402
from program import schema as S  # noqa: E402
from program import sources as SRC  # noqa: E402
from program import splits as SPL  # noqa: E402
from program.encode import ChatCodec, Unfit, encode_preference, encode_sft  # noqa: E402
from program.schema import ProgramError, canonical, sha256_hex  # noqa: E402

HOME = Path.home()
PRIVATE = HOME / "lilith-private"
DEFAULT_PARENT = Path("/Volumes/akh-medu-runs/hephaistos-20260927/lilith/parent-gemma4-12b-5926caa4")
DEFAULT_SNAPSHOTS = HOME / "Library" / "Application Support" / "akh-medu" / "lilith" / "cos-snapshots"
SEED = 0
REHEARSAL_CAP = 0.10

# Frozen stage plan: passes, learning-rate hint and the on-policy slot (families, quota).
STAGE_PLAN = {
    "white": {"sft_passes": 2, "preference_passes": 0, "lr_hint": 1e-4},
    "yellow": {"sft_passes": 2, "preference_passes": 0, "lr_hint": 5e-5},
    "green": {"sft_passes": 1, "preference_passes": 1, "lr_hint": 5e-5},
    "brown": {"sft_passes": 1, "preference_passes": 1, "lr_hint": 5e-5},
    "black": {"sft_passes": 1, "preference_passes": 1, "lr_hint": 5e-5},
}
ONPOLICY = {
    "green": {"families": ["memory-epistemics", "read-failure", "scope", "correction", "park-triage"], "quota": 120},
    "brown": {"families": ["briefing", "chase", "commitment", "operator-action", "pm-po-steering", "captain-selection"], "quota": 150},
    "black": {"families": ["illumos-guidance"], "quota": 60},
}


@dataclasses.dataclass
class Config:
    out: Path
    history: Path = PRIVATE / "history-20261001" / "corpus-v3"
    curriculum: Path = PRIVATE / "followup-20261003" / "curriculum"
    behavior_cases: Path = PRIVATE / "followup-20261003" / "behavior-cases.json"
    cos_trajectories: Path = PRIVATE / "c5" / "cos-trajectories-v3.jsonl"
    teacher_runs: tuple[Path, ...] = (PRIVATE / "c4" / "teacher-sealed-20261001-reverify-e214584e",
                                      PRIVATE / "c4" / "teacher-sealed-cut171956")
    c2_sessions: Path = PRIVATE / "c2-cache" / "sessions"
    memory_sessions: Path | None = None
    generated_dir: Path | None = PRIVATE / "program-20261003" / "generated"
    snapshot_root: Path | None = DEFAULT_SNAPSHOTS
    snapshot_until: str = S.SNAPSHOT_UNTIL
    parent: Path = DEFAULT_PARENT
    illumos_pack: Path | None = None
    sentinel_exclusions: Path | None = PRIVATE / "program-20261003" / "exam" / "sentinel-exclusions.jsonl"
    repo: Path = SRC.REPO_ROOT
    skip_dojo: bool = False
    dojo_prompt: str = "compact"
    context_limit: int = S.CONTEXT_LIMIT


# -- collection -----------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Collected:
    records: list[dict]
    exposure: SPL.Exposure
    exclusions: list[dict]
    sources: dict[str, Any]
    reports: dict[str, Any]
    prompts: dict[str, Any]


def merge_exposure(into: SPL.Exposure, other: SPL.Exposure) -> None:
    into.work_items |= other.work_items
    into.sessions |= other.sessions
    into.responses |= other.responses
    into.origins.update(other.origins)
    for k, v in other.wi_origin.items():
        into.wi_origin.setdefault(k, v)


MAX_CONTEXT_LIMIT = 32768


def check_context_limit(limit: int) -> int:
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_CONTEXT_LIMIT:
        raise ProgramError(f"--context-limit {limit!r} is outside 1..{MAX_CONTEXT_LIMIT}",
                           help="the trainer core supports up to 32K chunked attention; pick a limit within that")
    return limit


def collect(cfg: Config, log: Callable[[str], None] = lambda s: None) -> Collected:
    exposure = SPL.Exposure()
    exclusions: list[dict] = []
    sources: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    records: list[dict] = []

    def hashed(name: str, path: Path):
        sources[name] = {"path": str(path), "sha256": SRC.file_sha(path)}

    log("character corpus")
    records += SRC.kata_records() + SRC.sparring_records() + SRC.lineage_records()
    hashed("kata-drills", SRC.KATA_PATH)
    hashed("sparring-green", SRC.SPARRING_PATH)
    hashed("convert_corpus", SRC.CONVERT_PATH)

    log("historical lessons")
    hist, hist_exposure, hist_info = SRC.history_records(cfg.history)
    records += hist
    merge_exposure(exposure, hist_exposure)
    for name in ("train.jsonl", "train.tokens.jsonl", "dev.jsonl", "historical_eval.jsonl", "receipts.jsonl", "tokenization.json"):
        hashed("historical/" + name, cfg.history / name)
    reports["historical"] = hist_info

    log("behavior spots")
    spots = SRC.behavior_exposure(cfg.behavior_cases)
    merge_exposure(exposure, spots)
    hashed("behavior-cases", cfg.behavior_cases)

    log("cos-style transcripts")
    train, val = cfg.curriculum / "cos-style-train.jsonl", cfg.curriculum / "cos-style-validation.jsonl"
    records += SRC.cos_records(train, val)
    hashed("cos-style-train", train)
    hashed("cos-style-validation", val)
    hashed("curriculum/recipe", cfg.curriculum / "recipe.json")
    cos_x = SRC.cos_exposure(cfg.curriculum / "recipe.json", cfg.cos_trajectories)
    merge_exposure(exposure, cos_x)
    hashed("cos-trajectories-v3", cfg.cos_trajectories)

    log("c2 sessions")
    verdicts = SRC.load_verdicts()
    c2, c2_excluded = SRC.c2_records(cfg.c2_sessions, verdicts)
    records += c2
    reports["c2"] = {"positive_records": len(c2), "turns_not_used": collections.Counter(e["reason"] for e in c2_excluded)}
    reports["c2"]["turns_not_used"] = dict(reports["c2"]["turns_not_used"])
    sources["c2-turn-verdicts"] = {"path": str(SRC.VERDICTS_PATH), "sha256": SRC.file_sha(SRC.VERDICTS_PATH),
                                   "analysis_sha256": verdicts["analysis_sha256"]}

    log("rehearsal")
    records += SRC.rehearsal_records(cfg.repo)

    generated, rejected = SRC.generated_records(cfg.generated_dir)
    records += generated
    reports["generated"] = {"accepted": len(generated), "rejected": len(rejected)}
    for r in rejected:
        exclusions.append({"record_id": None, **r})

    prompts: dict[str, Any] = {"role_prompt_sha256": sha256_hex(SRC.role_prompt()), "role_file": "deploy/roman-mini/roles/lilith.md",
                               "persona_sha256": sha256_hex(SRC._convert_module().PERSONA)}
    reports["dojo"] = {"status": "skipped"}
    if not cfg.skip_dojo and cfg.snapshot_root and Path(cfg.snapshot_root).is_dir():
        from program import dojo_sources as DS  # imports the dojo; only the real export needs it

        log("dojo world (replay over recorder snapshots <= %s)" % cfg.snapshot_until)
        world = DS.open_world(cfg.snapshot_root, cfg.repo, cfg.snapshot_until, cfg.memory_sessions or cfg.c2_sessions, cfg.illumos_pack)
        sys_text = DS.system_text(world, cfg.dojo_prompt)
        prompts["dojo_prompt_profile"] = cfg.dojo_prompt
        prompts["dojo_system_prompt_sha256"] = sha256_hex(sys_text)
        prompts["dojo_full_system_prompt_sha256"] = world.system.sha256
        prompts["dojo_prompt_sources"] = dict(world.system.sources)
        log("teacher decisions")
        teacher, treport = DS.teacher_records(world, cfg.teacher_runs, sys_text)
        covered = {r["source_ref"]["episode_id"] for r in teacher}
        log("captain-selection renders")
        captain = DS.captain_records(world, covered, sys_text)
        log("memory-epistemics siblings")
        memory, mreport = DS.memory_records(world, cfg.c2_sessions, verdicts, sys_text)
        records += teacher + captain + memory
        reports["dojo"] = {"status": "ok", "snapshot_until": world.until, "snapshots_in_store": len(world.store.ids),
                           "specs_built": len(world.specs), "builder_exclusions": len(world.exclusions),
                           "teacher": treport, "captain_label_rendered": len(captain), "memory": mreport,
                           "world_status": "retrospective_development",
                           "note": "pre-cutoff dojo worlds examined during development; filenames that say sealed are historical, "
                                   "results and manifests are preserved, and none may later be called an untouched release exam"}
        for run in cfg.teacher_runs:
            if (Path(run) / "provenance.json").is_file():
                hashed("teacher-run/" + Path(run).name + "/provenance", Path(run) / "provenance.json")
    return Collected(records, exposure, exclusions, sources, reports, prompts)


# -- boundary ---------------------------------------------------------------------------------------------------


def apply_boundary(records: Sequence[dict], exposure: SPL.Exposure, sentinels: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Sentinel, snapshot-cutoff and exposure exclusions. Returns (kept, exclusion rows)."""
    kept, rows = [], []
    for rec in records:
        if sentinels:
            hit = sorted(SPL.record_keys_for_sentinels(rec) & sentinels)
            if hit:
                rows.append(_exclusion(rec, "sentinel", {"matched": len(hit), "sample": [h[:12] for h in hit[:3]]}))
                continue
        prov = rec.get("provenance") or {}
        snaps = [prov["snapshot_id"]] if prov.get("snapshot_id") else list(prov.get("snapshot_ids", []))
        late = [s for s in snaps if s >= S.SNAPSHOT_EXCLUDED_FROM]
        if late:
            rows.append(_exclusion(rec, "snapshot_after_cutoff", {"snapshots": late}))
            continue
        hits = SPL.exposure_hits(rec, exposure)
        if hits:
            for kind, vals in hits.items():
                detail = {"matched": len(vals), "sample": [v[:12] for v in vals[:3]]}
                if kind == "work_item":
                    detail["origins"] = SPL.hit_origins(hits, exposure)
                rows.append(_exclusion(rec, "exposure_" + kind, detail))
            continue
        kept.append(rec)
    return kept, rows


def _exclusion(rec: Mapping[str, Any], reason: str, detail: Mapping[str, Any]) -> dict:
    return {"record_id": rec["record_id"], "source": rec["source"], "stage": rec["stage"], "source_ref": rec["source_ref"],
            "reason": reason, "detail": dict(detail)}


# -- tokenization -----------------------------------------------------------------------------------------------


def tokenize_records(records: Sequence[dict], codec: ChatCodec | None, limit: int = S.CONTEXT_LIMIT):
    """(token rows by record_id, unfit rows, historical re-encode mismatch count)."""
    toks: dict[str, dict] = {}
    unfit: list[dict] = []
    mismatches = 0
    for rec in records:
        head = {k: rec[k] for k in ("record_id", "stage", "kind", "weight", "source", "families", "voice", "language",
                                    "component_id", "split")}
        try:
            if rec.get("_tokens") is not None:
                pre = rec["_tokens"]
                ids, mask = list(pre["input_ids"]), [int(bool(m)) for m in pre["target_mask"]]
                if codec is not None:
                    again = encode_sft(codec, rec["messages"], rec.get("tools"), limit)
                    if again.input_ids != ids or again.target_mask != mask:
                        mismatches += 1
                if len(ids) > limit:
                    raise Unfit(len(ids), limit, 0)
                toks[rec["record_id"]] = {**head, "input_ids": ids, "target_mask": mask}
            elif codec is None:
                raise ProgramError("a tokenizer is required to encode non-pretokenized records")
            elif rec["kind"] == "sft":
                e = encode_sft(codec, rec["messages"], rec.get("tools"), limit)
                toks[rec["record_id"]] = {**head, **e.to_dict()}
            else:
                c, r = encode_preference(codec, rec["chosen_messages"], rec["rejected_messages"], rec.get("tools"), limit)
                toks[rec["record_id"]] = {**head, "chosen": c.to_dict(), "rejected": r.to_dict()}
        except Unfit as u:
            unfit.append({"record_id": rec["record_id"], "source": rec["source"], "stage": rec["stage"], "kind": rec["kind"],
                          "families": rec["families"], "tokens": u.length, "limit": u.limit, "prompt_tokens": u.prompt_tokens,
                          "source_ref": rec["source_ref"]})
    return toks, unfit, mismatches


def apply_rehearsal_cap(records: list[dict], toks: Mapping[str, dict], cap: float = REHEARSAL_CAP) -> list[dict]:
    """Rehearsal stays at most `cap` of every stage: the surplus (highest record-id hash first) is dropped."""
    out = [r for r in records]
    for stage in S.STAGES:
        stage_recs = [r for r in out if r["stage"] == stage and r["record_id"] in toks and r["split"] != "excluded"]
        rehearsal = sorted((r for r in stage_recs if r["source"] == "rehearsal"), key=lambda r: sha256_hex(f"{SEED}:{r['record_id']}"))
        others = len(stage_recs) - len(rehearsal)
        allowed = int(cap * others / (1 - cap)) if cap < 1 else len(rehearsal)
        for r in rehearsal[allowed:]:
            r["split"] = "excluded"
            r.setdefault("verification", {})["rehearsal_cap"] = f"over {cap:.0%} of the stage"
    return out


# -- output -----------------------------------------------------------------------------------------------------


def write_private(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical(r) + "\n" for r in rows)


def order_key(rec_id: str) -> str:
    return sha256_hex(f"{SEED}:{rec_id}")


def stage_file(i: int, stage: str) -> str:
    return f"stage-{i}-{stage}.tokens.jsonl"


def _clean(rec: Mapping[str, Any]) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def counts_of(records: Sequence[Mapping[str, Any]]) -> dict:
    def tally(rows, key):
        c = collections.Counter(key(r) for r in rows)
        return dict(sorted(c.items()))

    live = [r for r in records if r["split"] != "excluded"]
    out = {"by_stage": {}, "by_source": tally(live, lambda r: r["source"]), "by_kind": tally(live, lambda r: r["kind"]),
           "by_split": tally(live, lambda r: r["split"])}
    for st in S.STAGES:
        rows = [r for r in live if r["stage"] == st]
        out["by_stage"][st] = {"total": len(rows), "by_source": tally(rows, lambda r: r["source"]),
                               "by_kind": tally(rows, lambda r: r["kind"]), "by_split": tally(rows, lambda r: r["split"]),
                               "by_source_kind": tally(rows, lambda r: f"{r['source']}/{r['kind']}")}
    return out


def build_manifest(cfg: Config, col: Collected, kept: Sequence[dict], toks: Mapping[str, dict], unfit: Sequence[dict],
                   exclusions: Sequence[dict], codec_info: Mapping[str, Any], coverage_doc: Mapping[str, Any],
                   hashes: Mapping[str, str], mismatches: int, template_sha: str | None) -> dict:
    stages = []
    for i, st in enumerate(S.STAGES, 1):
        plan = STAGE_PLAN[st]
        op = ONPOLICY.get(st)
        rows = [toks[r["record_id"]] for r in kept if r["stage"] == st and r["split"] != "excluded" and r["record_id"] in toks]
        counts = {"sft": sum(1 for t in rows if t["kind"] == "sft"), "preference": sum(1 for t in rows if t["kind"] == "preference"),
                  "train": sum(1 for t in rows if t["split"] == "train"), "dev": sum(1 for t in rows if t["split"] == "dev")}
        stages.append({"name": st, "order": i, "archive": stage_file(i, st), "archive_sha256": hashes.get(stage_file(i, st)),
                       "counts": counts,
                       "sft_passes": plan["sft_passes"], "preference_passes": plan["preference_passes"],
                       "onpolicy": ({"families": op["families"], "quota": op["quota"], "snapshot_until": S.SNAPSHOT_UNTIL} if op else None),
                       "lr_hint": plan["lr_hint"]})
    kept_ids = {r["record_id"] for r in kept}
    reasons = collections.Counter(e["reason"] for e in exclusions)
    by_unfit = collections.Counter(u["stage"] for u in unfit)
    live = [r for r in kept if r["split"] != "excluded" and r["record_id"] in toks]
    return {
        "schema": S.MANIFEST_SCHEMA, "run_id": S.RUN_ID, "context_limit": cfg.context_limit,
        "parent": {"model": "gemma-4-12b", **{k: v for k, v in codec_info.items() if k != "enable_thinking"},
                   "revision": codec_info.get("parent_revision"), "enable_thinking": codec_info.get("enable_thinking"),
                   "chat_template_sha256": template_sha},
        "parent_revision": codec_info.get("parent_revision"), "tokenizer_pins": codec_info.get("tokenizer_pins"),
        "files_sha256": {k: v for k, v in sorted(hashes.items())},
        "stages": stages,
        "counts": counts_of(live),
        "preference_records": sum(1 for r in live if r["kind"] == "preference"),
        "exclusions": {"total": len(exclusions), "by_reason": dict(sorted(reasons.items())), "file": "exclusions.jsonl",
                       "exposure_index": col.exposure.summary()},
        "unfit": {"count": len(unfit), "by_stage": dict(by_unfit), "limit": cfg.context_limit, "file": "unfit.jsonl",
                  "policy": "reported, never truncated"},
        "reused_tokens": {"historical_reencode_mismatches": mismatches},
        "sources": col.sources, "coverage": "coverage.json", "coverage_sha256": hashes.get("coverage.json"),
        "uncovered_cells": len(coverage_doc["uncovered"]),
        "prompts": col.prompts,
        "snapshot_boundary": {"until": S.SNAPSHOT_UNTIL, "excluded_from": S.SNAPSHOT_EXCLUDED_FROM,
                              "hindsight_labels": "built from snapshots at or before the cutoff only"},
        "dojo_worlds": col.reports.get("dojo"),
        "reports": {k: v for k, v in col.reports.items() if k != "dojo"},
        "split_policy": {"dev_fraction": SPL.DEV_FRACTION, "seed": SEED, "unit": "connected component over session ids and "
                         "subject work-item ids across all sources", "forced": "historical train lessons and cos-style train stay "
                         "train; cos-style validation stays dev; a forced-dev record in a forced-train component is excluded"},
        "rehearsal": {"cap_per_stage": REHEARSAL_CAP, "sources": list(SRC.REHEARSAL_SOURCES)},
        "onpolicy_archives": [],
        "limitations": [
            "The full 82-tool surface is not declared in dojo-derived records: it is ~15k tokens against an 8192 context. "
            "Each record declares the tools it calls.",
            "Preference pairs at export time exist only where a raw attempt was recorded; the Green/Brown/Black preference "
            "passes are filled by the on-policy rounds (python3 -m program.export --onpolicy RUN --stage S).",
            "Teacher record weight is the episode (answer-step) cos-v1 weight; per-step weights are kept in provenance.",
            "Captain-selection records without a teacher answer carry the dojo label voice and the structural feature code as reason, at the 0.7 admission floor.",
        ],
        "record_count_kept": len(kept_ids),
    }


def export_corpus(cfg: Config, codec: ChatCodec | None, log: Callable[[str], None] = lambda s: None) -> dict:
    col = collect(cfg, log)
    log("boundary")
    sentinels = SPL.load_sentinels(cfg.sentinel_exclusions) if cfg.sentinel_exclusions else None
    kept, boundary_rows = apply_boundary(col.records, col.exposure, sentinels)
    # Rehearsal copies per stage, before components so they take part in the splits.
    base_rehearsal = [r for r in kept if r["source"] == "rehearsal"]
    for st in S.STAGES:
        if st != "white":
            kept += [SRC.stage_copy(r, st) for r in base_rehearsal]
    log("components and splits")
    conflicts = SPL.assign_splits(kept)
    exclusions = col.exclusions + boundary_rows + conflicts
    live = [r for r in kept if r["split"] != "excluded"]
    log("tokenize %d records" % len(live))
    toks, unfit, mismatches = tokenize_records(live, codec, check_context_limit(cfg.context_limit))
    kept = apply_rehearsal_cap(kept, toks)
    unfit_ids = {u["record_id"] for u in unfit}
    for r in kept:
        if r["record_id"] in unfit_ids:
            r.setdefault("verification", {})["unfit"] = True
    for r in kept:
        if r["split"] == "excluded" and r["source"] == "rehearsal":
            exclusions.append(_exclusion(r, "rehearsal_cap", {"cap": REHEARSAL_CAP}))
    archive_rows = {st: [] for st in S.STAGES}
    for r in kept:
        if r["split"] != "excluded" and r["record_id"] in toks:
            archive_rows[r["stage"]].append(toks[r["record_id"]])
    out = Path(cfg.out)
    hashes: dict[str, str] = {}
    for i, st in enumerate(S.STAGES, 1):
        rows = sorted(archive_rows[st], key=lambda t: order_key(t["record_id"]))
        for t in rows:
            problems = S.validate_tokens(t)
            if problems:
                raise ProgramError(f"token record {t['record_id']} is invalid: {problems}")
        text = jsonl(rows)
        write_private(out / stage_file(i, st), text)
        hashes[stage_file(i, st)] = sha256_hex(text)
        recs = sorted((_clean(r) for r in kept if r["stage"] == st), key=lambda r: order_key(r["record_id"]))
        rtext = jsonl(recs)
        write_private(out / "records" / f"{st}.jsonl", rtext)
        hashes[f"records/{st}.jsonl"] = sha256_hex(rtext)
    kept_live = [r for r in kept if r["split"] != "excluded" and r["record_id"] in toks]
    cov = COV.compute(kept_live, {k: v["families"] for k, v in ONPOLICY.items()})
    cov_text = json.dumps(cov, indent=2, sort_keys=True) + "\n"
    write_private(out / "coverage.json", cov_text)
    hashes["coverage.json"] = sha256_hex(cov_text)
    ex_text = jsonl(sorted(exclusions, key=lambda e: (e["reason"], str(e.get("record_id")), canonical(e.get("detail")))))
    write_private(out / "exclusions.jsonl", ex_text)
    hashes["exclusions.jsonl"] = sha256_hex(ex_text)
    unfit_text = jsonl(sorted(unfit, key=lambda u: u["record_id"]))
    write_private(out / "unfit.jsonl", unfit_text)
    hashes["unfit.jsonl"] = sha256_hex(unfit_text)
    info = getattr(codec, "info", {}) if codec is not None else {}
    template_sha = info.get("tokenizer_pins", {}).get("chat_template.jinja")
    info = {**info, "context_limit": cfg.context_limit} if info else info
    manifest = build_manifest(cfg, col, kept, toks, unfit, exclusions, info, cov, hashes, mismatches, template_sha)
    manifest["exclusions"]["sentinels"] = (None if sentinels is None else sum(1 for e in exclusions if e["reason"] == "sentinel"))
    manifest["exclusions"]["sentinel_file"] = ({"path": str(cfg.sentinel_exclusions), "sha256": SRC.file_sha(cfg.sentinel_exclusions),
                                                "keys": len(sentinels)} if sentinels is not None else
                                               {"path": str(cfg.sentinel_exclusions) if cfg.sentinel_exclusions else None,
                                                "status": "absent: re-export once the exam freeze writes it"})
    if not SPL.components_isolated(kept):
        raise ProgramError("a connected component spans two splits")
    write_private(out / "program-manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if codec is not None:
        write_private(out / "tokenization.json", json.dumps({"schema": "lilith/program-tokenization@1", **info,
                      "archives": {k: v for k, v in hashes.items() if k.endswith(".tokens.jsonl")}}, indent=2, sort_keys=True) + "\n")
    return manifest


# -- on-policy mode ---------------------------------------------------------------------------------------------


def onpolicy_export(cfg: Config, codec: ChatCodec, run: Path, stage: str, log: Callable[[str], None] = lambda s: None) -> dict:
    from program import dojo_sources as DS

    out = Path(cfg.out)
    manifest_path = out / "program-manifest.json"
    if not manifest_path.is_file():
        raise ProgramError("no program-manifest.json under --out", help="run the base export first")
    manifest = json.loads(manifest_path.read_text())
    col = collect(dataclasses.replace(cfg, skip_dojo=True), log)  # only the exposure index is needed here
    world = DS.open_world(cfg.snapshot_root, cfg.repo, cfg.snapshot_until, cfg.memory_sessions or cfg.c2_sessions, cfg.illumos_pack)
    recs, report = DS.onpolicy_records(world, Path(run), stage, DS.system_text(world, cfg.dojo_prompt))
    kept, rows = apply_boundary(recs, col.exposure, SPL.load_sentinels(cfg.sentinel_exclusions) if cfg.sentinel_exclusions else None)
    existing: dict[str, str] = {}
    for path in (out / "records").glob("*.jsonl"):
        for r in SRC.read_jsonl(path):
            if r["split"] == "excluded":
                continue
            for k in SPL.link_keys(r):
                existing[k] = r["split"]
    for r in kept:
        keys = SPL.link_keys(r)
        splits = {existing[k] for k in keys if k in existing}
        r["component_id"] = "c-onpolicy-" + sha256_hex(sorted(keys) or [r["record_id"]])[:12]
        r["split"] = "dev" if splits == {"dev"} else "train"
    toks, unfit, _ = tokenize_records(kept, codec, check_context_limit(cfg.context_limit))
    rows_sorted = sorted((toks[r["record_id"]] for r in kept if r["record_id"] in toks), key=lambda t: order_key(t["record_id"]))
    for t in rows_sorted:
        problems = S.validate_tokens(t)
        if problems:
            raise ProgramError(f"token record {t['record_id']} is invalid: {problems}")
    name = f"onpolicy-{stage}.tokens.jsonl"
    text = jsonl(rows_sorted)
    write_private(out / name, text)
    write_private(out / "records" / f"onpolicy-{stage}.jsonl", jsonl(sorted((_clean(r) for r in kept), key=lambda r: order_key(r["record_id"]))))
    entry = {"stage": stage, "run": str(run), "archive": name, "archive_sha256": sha256_hex(text), "sha256": sha256_hex(text),
             "records": len(rows_sorted), "by_kind": dict(collections.Counter(t["kind"] for t in rows_sorted)),
             "unfit": len(unfit), "boundary_exclusions": len(rows), "report": report, "world_status": "retrospective_development"}
    manifest["onpolicy_archives"] = [e for e in manifest.get("onpolicy_archives", []) if (e["stage"], e["run"]) != (stage, str(run))] + [entry]
    for s in manifest["stages"]:
        if s["name"] == stage and s["onpolicy"] is not None:
            s["onpolicy"]["archive"] = name
            s["onpolicy"]["archive_sha256"] = entry["archive_sha256"]
    manifest.setdefault("files_sha256", {})[name] = entry["archive_sha256"]
    write_private(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_private(out / f"onpolicy-{stage}.unfit.jsonl", jsonl(sorted(unfit, key=lambda u: u["record_id"])))
    return entry


# -- CLI --------------------------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--generated-dir", type=Path, default=Config.generated_dir)
    p.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOTS)
    p.add_argument("--snapshot-until", default=S.SNAPSHOT_UNTIL)
    p.add_argument("--memory-sessions", type=Path, default=None)
    p.add_argument("--c2-sessions", type=Path, default=Config.c2_sessions)
    p.add_argument("--history", type=Path, default=Config.history)
    p.add_argument("--curriculum", type=Path, default=Config.curriculum)
    p.add_argument("--behavior-cases", type=Path, default=Config.behavior_cases)
    p.add_argument("--cos-trajectories", type=Path, default=Config.cos_trajectories)
    p.add_argument("--teacher-run", type=Path, action="append", default=None)
    p.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--illumos-pack", type=Path, default=None)
    p.add_argument("--sentinel-exclusions", type=Path, default=Config.sentinel_exclusions,
                   help="exam-freeze sentinel ids/hashes to exclude from every stage (null in the manifest when the file is absent)")
    p.add_argument("--context-limit", type=int, default=S.CONTEXT_LIMIT,
                   help=f"token limit per record (default {S.CONTEXT_LIMIT}, at most {MAX_CONTEXT_LIMIT}); longer records are listed in unfit.jsonl")
    p.add_argument("--skip-dojo", action="store_true")
    p.add_argument("--dojo-prompt", choices=("compact", "full"), default="compact",
                   help="system prompt of dojo-derived records: compact = role + dojo addendum, full = + chief-of-staff procedures")
    p.add_argument("--onpolicy", type=Path, default=None, metavar="RUN_DIR")
    p.add_argument("--stage", choices=S.STAGES, default=None)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    illumos = a.illumos_pack
    gen = a.generated_dir
    if illumos is None and gen and (Path(gen) / "illumos_pack.json").is_file():
        illumos = Path(gen) / "illumos_pack.json"
    cfg = Config(out=a.out, history=a.history, curriculum=a.curriculum, behavior_cases=a.behavior_cases,
                 cos_trajectories=a.cos_trajectories,
                 teacher_runs=tuple(a.teacher_run) if a.teacher_run else Config.teacher_runs, c2_sessions=a.c2_sessions,
                 memory_sessions=a.memory_sessions, generated_dir=gen, snapshot_root=a.snapshot_root,
                 snapshot_until=a.snapshot_until, parent=a.parent, illumos_pack=illumos, sentinel_exclusions=a.sentinel_exclusions, skip_dojo=a.skip_dojo, dojo_prompt=a.dojo_prompt, context_limit=a.context_limit)
    log = lambda s: print(f"[export] {s}", file=sys.stderr, flush=True)
    try:
        check_context_limit(cfg.context_limit)
        from program.encode import load_gemma_codec

        codec = load_gemma_codec(cfg.parent)
        if a.onpolicy:
            if not a.stage:
                raise ProgramError("--onpolicy needs --stage")
            entry = onpolicy_export(cfg, codec, a.onpolicy, a.stage, log)
            print(json.dumps(entry, indent=2, sort_keys=True))
        else:
            manifest = export_corpus(cfg, codec, log)
            print(json.dumps({"counts": manifest["counts"], "exclusions": manifest["exclusions"], "unfit": manifest["unfit"],
                              "uncovered_cells": manifest["uncovered_cells"]}, indent=2, sort_keys=True))
    except ProgramError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
