"""The coverage matrix: stage x family x voice x language x source, the eight voices and eight kata, Double
Diamond / ERRC presence (structural: records whose source labels them, or whose answer document carries the
optional `diamond` / `memory_review` sections) and the cells that stay uncovered."""
from __future__ import annotations

import collections
import json
from typing import Any, Iterable, Mapping, Sequence

from program.schema import STAGES, VOICES

KATA_IDS = ("clarify-before-answer", "double-diamond", "mirror", "naming-the-move", "redirect-refusal", "silence",
            "stubborn-affirmation", "yes-and")
LANGUAGES = ("en", "de", "fr")
DIAMOND_KATA = "double-diamond"
DIAMOND_KEYS = ("diamond",)
ERRC_KEYS = ("memory_review",)

# The inherited ADAM / Kaine capability groups (checklist 2026-09-27). `rehearsal_sources` lists the exact
# on-main record sets that feed the rehearsal source; a group with none is reported as uncovered by rehearsal.
CAPABILITIES: Mapping[str, Mapping[str, Any]] = {
    "tool_native_calls": {"checklist_rows": [1, 2],
                          "note": "no chat-shaped tool-call record set exists on main under training/lilith; ADAM tool-native data lives in the grug-data corpus under another tokenizer"},
    "provenance_revision": {"checklist_rows": [5, 6],
                            "note": "no rehearsal records; the historical memory_revision lessons and the memory-epistemics siblings exercise it inside the CoS role"},
    "retrieval_long_context": {"checklist_rows": [7],
                               "note": "recall episodes are measurement tasks under experiments/native-recall, not training records"},
    "coordination": {"checklist_rows": [24],
                     "note": "training/lilith/coordination-* hold probes and reports, not records; chase and pm-po-steering teacher decisions cover it inside the role"},
    "coding": {"checklist_rows": [11, 12], "rehearsal_sources": ["training/lilith/rl-20260921/tasks.json", "training/lilith/rl-20260922/tasks.json"],
               "note": "training_targets of train-split tasks (assistant-authored references with replay receipts)"},
    "languages": {"checklist_rows": [13],
                  "note": "no rehearsal records; EN/DE/FR appear in the kata, sparring and lineage sources (see language cells)"},
}


def _answer_keys(rec: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    msgs = rec.get("messages") or rec.get("chosen_messages") or []
    for m in msgs:
        if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"].lstrip().startswith("{"):
            try:
                doc = json.loads(m["content"])
            except json.JSONDecodeError:
                continue
            if isinstance(doc, dict):
                keys |= set(doc)
    return keys


def compute(records: Sequence[Mapping[str, Any]], onpolicy_families: Mapping[str, Sequence[str]] | None = None) -> dict:
    live = [r for r in records if r["split"] in ("train", "dev")]
    cells = collections.Counter()
    by_stage_source: dict = collections.defaultdict(collections.Counter)
    by_stage_family: dict = collections.defaultdict(collections.Counter)
    by_stage_voice: dict = collections.defaultdict(collections.Counter)
    by_stage_language: dict = collections.defaultdict(collections.Counter)
    kata_stage: dict = collections.defaultdict(collections.Counter)
    voice_kata: dict = collections.defaultdict(collections.Counter)
    diamond = collections.Counter()
    errc = collections.Counter()
    capability = collections.Counter()
    for r in live:
        st, src, voice, lang = r["stage"], r["source"], r["voice"] or "-", r["language"]
        fams = r["families"] or ["-"]
        by_stage_source[st][f"{src}/{r['kind']}"] += 1
        by_stage_voice[st][voice] += 1
        by_stage_language[st][lang] += 1
        for f in fams:
            by_stage_family[st][f] += 1
            cells[(st, f, voice, lang, src)] += 1
        if r.get("kata"):
            kata_stage[r["kata"]][st] += 1
            voice_kata[f"{voice}|{r['kata']}"][st] += 1
        keys = _answer_keys(r)
        if r.get("kata") == DIAMOND_KATA or DIAMOND_KATA in fams:
            diamond[f"{st}/source-labelled-kata"] += 1
        if keys & set(DIAMOND_KEYS):
            diamond[f"{st}/answer-document"] += 1
        if keys & set(ERRC_KEYS):
            errc[f"{st}/answer-document"] += 1
        cap = (r.get("provenance") or {}).get("capability")
        if cap:
            capability[f"{st}/{cap}"] += 1
    uncovered: list[dict] = []
    for st in STAGES:
        for v in VOICES:
            if by_stage_voice[st][v] == 0:
                uncovered.append({"cell": "stage x voice", "stage": st, "voice": v, "reason": "no record carries this voice at this stage"})
        for lang in LANGUAGES:
            if by_stage_language[st][lang] == 0:
                uncovered.append({"cell": "stage x language", "stage": st, "language": lang, "reason": "no record in this language at this stage"})
    for st in ("white", "yellow"):
        for k in KATA_IDS:
            if kata_stage[k][st] == 0:
                uncovered.append({"cell": "stage x kata", "stage": st, "kata": k, "reason": "no drill record"})
    for v in VOICES:
        for k in KATA_IDS:
            if sum(voice_kata[f"{v}|{k}"].values()) == 0:
                uncovered.append({"cell": "voice x kata", "voice": v, "kata": k, "reason": "this voice never drills this kata"})
    if not diamond:
        uncovered.append({"cell": "double-diamond", "reason": "no record is labelled Double Diamond"})
    if not errc:
        uncovered.append({"cell": "errc", "reason": "no answer document carries the memory_review (ERRC) section"})
    for st, fams in (onpolicy_families or {}).items():
        for f in fams:
            if by_stage_family[st][f] == 0:
                uncovered.append({"cell": "stage x family", "stage": st, "family": f,
                                  "reason": "no static record; filled by the on-policy slot after generation"})
    cap_status = {}
    for name, info in CAPABILITIES.items():
        n = sum(v for k, v in capability.items() if k.endswith("/" + name))
        status = "rehearsal_records" if n else "uncovered_by_rehearsal"
        cap_status[name] = {"status": status, "rehearsal_records": n, **info}
        if not n:
            uncovered.append({"cell": "rehearsal capability", "capability": name, "reason": info["note"]})
    return {
        "schema": "lilith/program-coverage@1",
        "records_counted": len(live),
        "stage_x_source_kind": {s: dict(sorted(c.items())) for s, c in by_stage_source.items()},
        "stage_x_family": {s: dict(sorted(c.items())) for s, c in by_stage_family.items()},
        "stage_x_voice": {s: dict(sorted(c.items())) for s, c in by_stage_voice.items()},
        "stage_x_language": {s: dict(sorted(c.items())) for s, c in by_stage_language.items()},
        "cells": [{"stage": k[0], "family": k[1], "voice": k[2], "language": k[3], "source": k[4], "count": v}
                  for k, v in sorted(cells.items())],
        "voices": {v: {st: by_stage_voice[st][v] for st in STAGES} for v in VOICES},
        "kata": {k: dict(kata_stage[k]) for k in KATA_IDS},
        "double_diamond": dict(diamond),
        "errc": dict(errc),
        "rehearsal_capabilities": cap_status,
        "uncovered": uncovered,
    }
