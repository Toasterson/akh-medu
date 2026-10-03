"""Boundary and splits: exposure exclusion and connected-component splitting.

Exposure is checked through the FULL record text, not only subject ids. A record is excluded when any of
its messages contains a work-item id, a session id or the exact text of a tool response (by sha256) that
belongs to a held-out source (the historical dev/eval lessons and their sessions, the four behavior spots,
the cos-style session excluded for holdout overlap, anything derived from snapshots at or after the cutoff).

Connected components link records that share a session id or a work-item id, across all sources. One
component lands in exactly one split. Pre-assigned splits from a source (`provenance.force_split`) win; a
component that holds both a forced-train and a forced-dev record keeps train and excludes the dev records.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import re
from typing import Iterable, Mapping, MutableSequence, Sequence

from program.schema import canonical, sha256_hex, tool_response_hashes

UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
DEV_FRACTION = 0.10
GIANT_SHARE = 0.25  # a component holding more than this share of the records trains (never a tiny dev sample)
SEED = 0


def uuids_in(value) -> set[str]:
    return {u.lower() for u in UUID.findall(canonical(value))}


SCOPE_ID = re.compile(r'(?:project_id|plan_id)\\*"\s*:\s*\\*"(' + UUID.pattern + ')')


def work_item_uuids(value) -> set[str]:
    """UUIDs that are not project or plan ids. Those are shared by whole projects, so holding them out would
    exclude every record of the project; structurally they are the values of `project_id` / `plan_id` keys."""
    text = value if isinstance(value, str) else canonical(value)
    scope = {u.lower() for u in SCOPE_ID.findall(text)}
    return {u.lower() for u in UUID.findall(text)} - scope


def record_messages(rec: Mapping) -> list[Mapping]:
    return [*(rec.get("messages") or []), *(rec.get("chosen_messages") or []), *(rec.get("rejected_messages") or [])]


@dataclasses.dataclass
class Exposure:
    """What a held-out source exposes: work-item ids, session ids and tool-response text hashes."""

    work_items: set[str] = dataclasses.field(default_factory=set)
    sessions: set[str] = dataclasses.field(default_factory=set)
    responses: set[str] = dataclasses.field(default_factory=set)
    origins: dict[str, int] = dataclasses.field(default_factory=lambda: collections.Counter())
    wi_origin: dict[str, str] = dataclasses.field(default_factory=dict)

    def add(self, *, origin: str, work_items: Iterable[str] = (), sessions: Iterable[str] = (),
            responses: Iterable[str] = ()) -> None:
        for w in work_items:
            self.work_items.add(w.lower())
            self.wi_origin.setdefault(w.lower(), origin)
        self.sessions |= {str(s) for s in sessions}
        self.responses |= set(responses)
        self.origins[origin] += 1

    def add_text(self, origin: str, value, *, sessions: Iterable[str] = ()) -> None:
        """A held-out artifact given as JSON-able text: its ids and its own sha256 count as exposed."""
        self.add(origin=origin, work_items=uuids_in(value), sessions=sessions, responses=[sha256_hex(value)])

    def summary(self) -> dict:
        return {"work_items": len(self.work_items), "sessions": len(self.sessions), "response_hashes": len(self.responses),
                "origins": dict(self.origins)}


def record_exposure(rec: Mapping) -> tuple[set[str], set[str], set[str]]:
    """System messages are the shared role/procedure prompt (identical for every record and for the held-out
    evaluation); the boundary is about the situation content the model reads: user, assistant and tool turns."""
    prov = rec.get("provenance") or {}
    wis = uuids_in([m for m in record_messages(rec) if m.get("role") != "system"])
    sessions = {str(s) for s in prov.get("session_ids", [])}
    responses = set(tool_response_hashes(record_messages(rec))) | set(prov.get("observation_hashes", []))
    return wis, sessions, responses


def exposure_hits(rec: Mapping, exposure: Exposure) -> dict[str, list[str]]:
    wis, sessions, responses = record_exposure(rec)
    hits = {"work_item": sorted(wis & exposure.work_items), "session": sorted(sessions & exposure.sessions),
            "tool_response": sorted(responses & exposure.responses)}
    return {k: v for k, v in hits.items() if v}


def hit_origins(hits: Mapping[str, Sequence[str]], exposure: Exposure) -> list[str]:
    return sorted({exposure.wi_origin.get(w, "?") for w in hits.get("work_item", [])})


def link_keys(rec: Mapping) -> set[str]:
    prov = rec.get("provenance") or {}
    keys = {f"session:{s}" for s in prov.get("session_ids", [])}
    keys |= {f"wi:{w.lower()}" for w in prov.get("subject_ids", [])}
    keys |= {f"key:{k}" for k in prov.get("link_keys", [])}
    return keys


class _Union:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def assign_components(records: Sequence[MutableSequence | dict]) -> dict[str, list[dict]]:
    """Set `component_id` on every record; returns component id -> records."""
    uf = _Union()
    for rec in records:
        node = f"rec:{rec['record_id']}"
        uf.find(node)
        for key in link_keys(rec):
            uf.union(node, key)
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for rec in records:
        groups[uf.find(f"rec:{rec['record_id']}")].append(rec)
    out: dict[str, list[dict]] = {}
    for root, members in groups.items():
        cid = "c-" + hashlib.sha256(root.encode()).hexdigest()[:16]
        for rec in members:
            rec["component_id"] = cid
        out[cid] = members
    return out


def _dev_by_hash(cid: str, fraction: float, seed: int) -> bool:
    return int(sha256_hex(f"{seed}:{cid}")[:8], 16) / 0xFFFFFFFF < fraction


def assign_splits(records: Sequence[dict], *, dev_fraction: float = DEV_FRACTION, seed: int = SEED) -> list[dict]:
    """Splits per component; returns exclusion rows for records dropped by a component conflict.

    Excluded records already carry split `excluded` and are left out of component linking by the caller."""
    comps = assign_components(records)
    total = max(1, len(records))
    exclusions: list[dict] = []
    for cid, members in comps.items():
        forced = {(m.get("provenance") or {}).get("force_split") for m in members} - {None}
        if "train" in forced:
            split = "train"
        elif "dev" in forced:
            split = "dev"
        elif len(members) / total > GIANT_SHARE:
            split = "train"
        else:
            split = "dev" if _dev_by_hash(cid, dev_fraction, seed) else "train"
        for m in members:
            want = (m.get("provenance") or {}).get("force_split")
            if want == "dev" and split == "train" and "train" in forced:
                m["split"] = "excluded"
                exclusions.append({"record_id": m["record_id"], "source": m["source"], "reason": "component_conflict",
                                   "detail": "a forced-dev record shares a session or work item with a forced-train record",
                                   "component_id": cid})
            else:
                m["split"] = split
    return exclusions


def components_isolated(records: Iterable[Mapping]) -> bool:
    """True when no component spans two splits (the invariant tests assert)."""
    seen: dict[str, set[str]] = collections.defaultdict(set)
    for r in records:
        if r["split"] != "excluded":
            seen[r["component_id"]].add(r["split"])
    return all(len(v) == 1 for v in seen.values())


# -- sentinel exclusions (acceptance-exam freeze) -------------------------------------------------------------------


def load_sentinels(path) -> set[str] | None:
    """Ids and content hashes the exam freeze holds out. None when the file does not exist (a re-export is then
    required once it does). Every string value of at least 8 characters on a row counts as a candidate key, so the
    exam writer's exact field names do not matter."""
    from pathlib import Path
    import json

    path = Path(path)
    if not path.is_file():
        return None
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stack = [row]
        while stack:
            v = stack.pop()
            if isinstance(v, str):
                if len(v) >= 8:
                    keys.add(v.lower() if len(v) == 64 else v)
            elif isinstance(v, dict):
                stack.extend(v.values())
            elif isinstance(v, list):
                stack.extend(v)
    return keys


def record_keys_for_sentinels(rec: Mapping) -> set[str]:
    """Every identity a held-out item could be named by: this record's id, its source row's ids and hashes, and the
    sha256 of each message text (so a sentinel given as a drill's situation or move text matches)."""
    prov = rec.get("provenance") or {}
    ref = rec.get("source_ref") if isinstance(rec.get("source_ref"), dict) else {}
    keys = {rec["record_id"], prov.get("historical_record_id"), ref.get("original_id"), ref.get("episode_id"),
            prov.get("source_row_sha256"), *prov.get("source_text_sha256", [])}
    for m in record_messages(rec):
        if isinstance(m.get("content"), str) and m["content"] and m.get("role") != "system":
            keys.add(sha256_hex(m["content"]))
    # a generated record names its seed in provenance; a held-out seed taints its variations
    keys |= {v for v in prov.values() if isinstance(v, str) and len(v) >= 8}
    return {k for k in keys if k}
