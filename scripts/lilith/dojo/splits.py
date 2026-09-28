"""Train / dev / sealed splits by time window and Anima Plan, plus decontamination.

- Windows are fixed-width from a fixed origin (the recorder's first day), so a
  window's membership never changes as snapshots accumulate.
- sealed = the newest window + every episode whose plans are all held out.
  dev = the window before the newest; train = the rest.
- Leak guard: a train/dev episode whose hindsight labels read a snapshot inside a
  later split's window is dropped (listed with the reason), never reassigned.
- Episodes spanning several plans (the briefing) are split by time only.
- Decontamination follows P6.3 (`src/bin/akh_episode_generator.rs`): NFKC,
  per-character lowercase, whitespace split, alphanumeric-only words; exact-text
  and 13-word-shingle BLAKE3 hashes; baseline subtraction for boilerplate.
"""
from __future__ import annotations

import dataclasses
import enum
import json
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

from dojo.blake3_compat import blake3_digest
from dojo.episode import EpisodeSpec
from dojo.errors import SplitLeakError
from dojo.snapshot import canonical_json, parse_snapshot_time, sha256_hex

SPLITS_SCHEMA = "lilith/cos-splits@1"
RECORDER_ORIGIN = datetime(2026, 9, 28, tzinfo=timezone.utc)
SHINGLE_WORDS = 13


class Split(enum.StrEnum):
    TRAIN = "train"
    DEV = "dev"
    SEALED = "sealed"


@dataclasses.dataclass(frozen=True)
class SplitConfig:
    window: timedelta = timedelta(days=7)
    origin: datetime = RECORDER_ORIGIN
    held_out_plans: tuple[str, ...] = ()
    held_out_plan_fraction: float = 0.0  # additionally hold out plans whose salted hash falls below this
    salt: str = "cos-splits-v1"

    def to_dict(self) -> dict:
        return {"window_hours": self.window.total_seconds() / 3600, "origin": self.origin.isoformat(),
                "held_out_plans": sorted(self.held_out_plans), "held_out_plan_fraction": self.held_out_plan_fraction,
                "salt": self.salt}

    def plan_held_out(self, plan_id: str) -> bool:
        if plan_id in self.held_out_plans:
            return True
        if self.held_out_plan_fraction <= 0:
            return False
        bucket = int(sha256_hex(f"{self.salt}:{plan_id}".encode())[:8], 16) / 0xFFFFFFFF
        return bucket < self.held_out_plan_fraction

    def window_of(self, snapshot_id: str) -> int:
        return int((parse_snapshot_time(snapshot_id) - self.origin) // self.window)

    def window_start(self, index: int) -> datetime:
        return self.origin + index * self.window


@dataclasses.dataclass(frozen=True)
class Assignment:
    spec: EpisodeSpec
    split: Split
    window: int
    reason: str

    def to_dict(self) -> dict:
        return {"episode_id": self.spec.episode_id, "spec_sha256": self.spec.spec_sha256,
                "family": self.spec.family.value, "snapshot": self.spec.snapshot_id,
                "horizon_end": self.spec.horizon_end, "plans": list(self.spec.plan_ids),
                "window": self.window, "reason": self.reason}


@dataclasses.dataclass(frozen=True)
class SplitManifest:
    config: SplitConfig
    assignments: tuple[Assignment, ...]
    dropped: tuple[tuple[str, str], ...]  # (episode_id, reason)

    def members(self, split: Split) -> list[Assignment]:
        return [a for a in self.assignments if a.split is split]

    def to_dict(self) -> dict:
        splits = {}
        for split in Split:
            rows = sorted((a.to_dict() for a in self.members(split)), key=lambda r: r["episode_id"])
            splits[split.value] = {"count": len(rows), "sha256": sha256_hex(canonical_json(rows)), "episodes": rows}
        body = {"schema": SPLITS_SCHEMA, "config": self.config.to_dict(), "splits": splits,
                "dropped": [{"episode_id": e, "reason": r} for e, r in sorted(self.dropped)]}
        return body | {"manifest_sha256": sha256_hex(canonical_json(body))}

    def check_disjoint(self) -> None:
        seen: dict[str, Split] = {}
        for a in self.assignments:
            if a.spec.episode_id in seen and seen[a.spec.episode_id] is not a.split:
                raise SplitLeakError(f"{a.spec.episode_id} is in both {seen[a.spec.episode_id]} and {a.split}")
            seen[a.spec.episode_id] = a.split
        sealed_plans = {p for a in self.members(Split.SEALED) if a.reason == "held-out-plan" for p in a.spec.plan_ids}
        for a in self.assignments:
            if a.split is not Split.SEALED and len(a.spec.plan_ids) == 1 and a.spec.plan_ids[0] in sealed_plans:
                raise SplitLeakError(f"{a.spec.episode_id} ({a.split}) belongs to held-out plan {a.spec.plan_ids[0]}")


def assign_splits(specs: Sequence[EpisodeSpec], config: SplitConfig = SplitConfig()) -> SplitManifest:
    if not specs:
        return SplitManifest(config, (), ())
    newest = max(config.window_of(s.snapshot_id) for s in specs)
    sealed_start = config.window_start(newest)
    dev_start = config.window_start(newest - 1)
    assignments: list[Assignment] = []
    dropped: list[tuple[str, str]] = []
    unique: dict[str, EpisodeSpec] = {}
    for spec in specs:
        unique.setdefault(spec.episode_id, spec)
    for spec in unique.values():
        window = config.window_of(spec.snapshot_id)
        horizon = parse_snapshot_time(spec.horizon_end)
        plan_sealed = bool(spec.plan_ids) and all(config.plan_held_out(p) for p in spec.plan_ids)
        if window == newest:
            assignments.append(Assignment(spec, Split.SEALED, window, "newest-window"))
        elif plan_sealed:
            assignments.append(Assignment(spec, Split.SEALED, window, "held-out-plan"))
        elif window == newest - 1:
            if horizon >= sealed_start:
                dropped.append((spec.episode_id, "dev labels read the sealed window"))
            else:
                assignments.append(Assignment(spec, Split.DEV, window, "previous-window"))
        else:
            if horizon >= dev_start:
                dropped.append((spec.episode_id, "train labels read the dev or sealed window"))
            else:
                assignments.append(Assignment(spec, Split.TRAIN, window, "older-window"))
    manifest = SplitManifest(config, tuple(assignments), tuple(dropped))
    manifest.check_disjoint()
    return manifest


# --- decontamination (P6.3-compatible) ----------------------------------------------------


def normalized_words(text: str) -> list[str]:
    folded = "".join(c.lower() for c in unicodedata.normalize("NFKC", text))
    words = ("".join(c for c in w if c.isalnum()) for w in folded.split())
    return [w for w in words if w]


def text_hash(text: str) -> bytes:
    return blake3_digest(" ".join(normalized_words(text)).encode("utf-8"))


def shingle_hashes(text: str, size: int = SHINGLE_WORDS) -> list[bytes]:
    words = normalized_words(text)
    if len(words) < size:
        return []
    return [blake3_digest(" ".join(words[i:i + size]).encode("utf-8")) for i in range(len(words) - size + 1)]


@dataclasses.dataclass
class SealedIndex:
    exact: set[bytes] = dataclasses.field(default_factory=set)
    shingles: set[bytes] = dataclasses.field(default_factory=set)
    texts: int = 0
    removed_by_baseline: int = 0

    def add(self, text: str) -> None:
        self.exact.add(text_hash(text))
        self.shingles.update(shingle_hashes(text))
        self.texts += 1

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "SealedIndex":
        index = cls()
        for t in texts:
            index.add(t)
        return index

    @classmethod
    def from_jsonl(cls, paths: Iterable[Path]) -> "SealedIndex":
        """Reads sentinel files the way P6.3's `SentinelIndex::load` does."""
        index = cls()
        for path in paths:
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in ("prompt", "target", "user_request", "request", "text"):
                    if isinstance(value.get(key), str):
                        index.add(value[key])
                if "expected_call" in value:
                    index.add(json.dumps(value["expected_call"], separators=(",", ":")))
        return index

    def subtract_baseline(self, baseline: Iterable[str]) -> "SealedIndex":
        base = SealedIndex.from_texts(baseline)
        before = len(self.exact) + len(self.shingles)
        self.exact -= base.exact
        self.shingles -= base.shingles
        self.removed_by_baseline = before - len(self.exact) - len(self.shingles)
        return self

    def collision(self, text: str) -> str | None:
        if text_hash(text) in self.exact:
            return "exact"
        if any(h in self.shingles for h in shingle_hashes(text)):
            return f"{SHINGLE_WORDS}-word-shingle"
        return None


def spec_texts(spec: EpisodeSpec) -> list[str]:
    """What a sealed spec contributes to the index: its situation (prompt side) and label (target side)."""
    return [spec.prompt_text(), json.dumps(spec.label, sort_keys=True, ensure_ascii=False)]


def sealed_index(manifest: SplitManifest, baseline: Iterable[str] = ()) -> SealedIndex:
    index = SealedIndex.from_texts(t for a in manifest.members(Split.SEALED) for t in spec_texts(a.spec))
    return index.subtract_baseline(baseline)


@dataclasses.dataclass(frozen=True)
class DecontaminationReport:
    checked: int
    dropped: tuple[tuple[str, str], ...]  # (record id, reason)
    sealed_texts: int
    removed_by_baseline: int

    def to_dict(self) -> dict:
        return {"method": f"nfkc-lower-alnum exact + {SHINGLE_WORDS}-word shingle, blake3", "checked": self.checked,
                "dropped": [{"id": i, "reason": r} for i, r in self.dropped], "sealed_texts": self.sealed_texts,
                "removed_by_baseline": self.removed_by_baseline}


def decontaminate(records: Iterable[tuple[str, Sequence[str]]], index: SealedIndex) -> tuple[list[str], DecontaminationReport]:
    """`records`: (id, texts) of train/dev material (rendered prompts, targets, siblings)."""
    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    checked = 0
    for record_id, texts in records:
        checked += 1
        reason = next((r for r in (index.collision(t) for t in texts) if r), None)
        if reason:
            dropped.append((record_id, reason))
        else:
            kept.append(record_id)
    return kept, DecontaminationReport(checked, tuple(dropped), index.texts, index.removed_by_baseline)

