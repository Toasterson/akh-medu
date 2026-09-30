"""Per-model turn deadline and HTTP timeout for the dojo arms (ADR 123; Anima WI C4 01a0e270-475e).

The akh resolves its own limits as `[model_profile]` override > catalog row for (model, effort) >
global default (`src/agent/model_profile.rs`). The dojo must not cut a turn shorter than the akh
would allow it, so it reads the same two sources:

- the teacher akh's rendered workspace TOML: `model`, `reasoning_effort`, `[model_profile]`;
- `data/model-profiles/catalog.toml` (one catalog, compiled into akhomed and read here).

`resolve_turn_timeout` returns the wall-clock budget for one `session/prompt`: the akh's resolved turn
deadline plus a margin (the akh checks its deadline between model rounds, so a turn can overrun by one
round), its source, and every input, so provenance records where the number came from. Nothing here
changes what is measured, so none of it is part of a run's resume identity.

stdlib only (`tomllib`).
"""
from __future__ import annotations

import dataclasses
import enum
import tomllib
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dojo.contracts import REPO_ROOT

CATALOG_RELATIVE = Path("data/model-profiles/catalog.toml")
CATALOG_SCHEMA_VERSION = 1
# ADR 123: glm-5.3's row. Used when neither the akh's config nor the catalog names a deadline.
FALLBACK_TURN_TIMEOUT_S = 1800.0
# The akh checks `turn_deadline_secs` between model rounds, so a turn can overrun it by the round in
# flight: one HTTP timeout (300 s for glm-5.3) plus tool time.
TURN_MARGIN_S = 300.0
FALLBACK_HTTP_TIMEOUT_S = 300.0


class TimeoutSource(enum.StrEnum):
    FLAG = "flag"                      # --turn-timeout / --timeout: taken exactly as given
    AKH_OVERRIDE = "akh-model-profile"  # the akh's own `[model_profile]`
    CATALOG = "catalog"                # the catalog row for the akh's model (and effort)
    FALLBACK = "fallback"              # neither was readable


@dataclasses.dataclass(frozen=True)
class ResolvedTimeout:
    seconds: float
    source: TimeoutSource
    deadline_s: float | None = None     # the akh's own limit, before the margin
    margin_s: float = 0.0
    model: str | None = None
    effort: str | None = None
    catalog_row: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: (v.value if isinstance(v, enum.Enum) else v) for k, v in dataclasses.asdict(self).items()
                if v is not None}


def normalize_model_id(model: str) -> str:
    """Same normalization as `model_profile::normalize_model_id` in Rust (and `full_akh.py`)."""
    base = model.strip().lower().rsplit("/", 1)[-1]
    for tag in (":cloud", "-cloud"):
        if base.endswith(tag):
            return base[: -len(tag)]
    return base


def load_catalog(path: Path | None = None) -> dict | None:
    """The model catalog, or None when it is missing or unreadable (the caller falls back and says so)."""
    file = Path(path) if path is not None else REPO_ROOT / CATALOG_RELATIVE
    try:
        catalog = tomllib.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return catalog if catalog.get("schema_version") == CATALOG_SCHEMA_VERSION else None


def catalog_profile(model: str, effort: str | None, catalog: Mapping[str, Any]) -> tuple[str | None, dict]:
    """(row id, profile) for `model` at `effort`: an exact id wins, else the longest prefix; the
    `[models.effort.<level>]` override is laid over the row. (None, {}) when the catalog has no row."""
    model_id = normalize_model_id(model)
    rows = catalog.get("models", [])
    row = next((r for r in rows if model_id in {normalize_model_id(i) for i in r.get("ids", [])}), None)
    if row is None:
        hits = [(len(normalize_model_id(p)), r) for r in rows for p in r.get("prefixes", [])
                if model_id.startswith(normalize_model_id(p))]
        row = max(hits, key=lambda hit: hit[0])[1] if hits else None
    if row is None:
        return None, {}
    profile = dict(row.get("profile", {}))
    profile.update(row.get("effort", {}).get(effort or "", {}))
    return row.get("id"), profile


def _seconds(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None


def resolve_turn_timeout(akh_doc: Mapping[str, Any] | None, catalog: Mapping[str, Any] | None, *,
                         flag: float | None = None, margin_s: float = TURN_MARGIN_S,
                         fallback_s: float = FALLBACK_TURN_TIMEOUT_S) -> ResolvedTimeout:
    """The wall-clock budget for one teacher `session/prompt`.

    Order: `flag` (exactly as given, no margin) > the akh's `[model_profile] turn_deadline_secs` >
    the catalog row for the akh's `model` and `reasoning_effort` > `fallback_s`. The last two add
    `margin_s`; a value from the akh's own config is its deadline, so the dojo waits the margin
    beyond it before it calls the turn lost."""
    doc = akh_doc or {}
    model = doc.get("model") if isinstance(doc.get("model"), str) else None
    effort = doc.get("reasoning_effort") if isinstance(doc.get("reasoning_effort"), str) else None
    if flag is not None:
        return ResolvedTimeout(flag, TimeoutSource.FLAG, model=model, effort=effort)
    override = _seconds((doc.get("model_profile") or {}).get("turn_deadline_secs"))
    if override is not None:
        return ResolvedTimeout(override + margin_s, TimeoutSource.AKH_OVERRIDE, override, margin_s, model, effort)
    if model is not None and catalog is not None:
        row, profile = catalog_profile(model, effort, catalog)
        deadline = _seconds(profile.get("turn_deadline_secs"))
        if deadline is not None:
            return ResolvedTimeout(deadline + margin_s, TimeoutSource.CATALOG, deadline, margin_s, model, effort, row)
    why = ("the akh config was not given" if akh_doc is None else
           "the akh config names no model" if model is None else
           "the catalog is missing" if catalog is None else f"the catalog has no row with a deadline for {model!r}")
    return ResolvedTimeout(fallback_s, TimeoutSource.FALLBACK, None, 0.0, model, effort, None,
                           f"{why}; pass --turn-timeout to set the prompt timeout")


def resolve_http_timeout(model: str, catalog: Mapping[str, Any] | None, *, flag: float | None = None,
                         effort: str | None = None,
                         fallback_s: float = FALLBACK_HTTP_TIMEOUT_S) -> ResolvedTimeout:
    """The in-process arm's per-request HTTP timeout: `flag` > the catalog row's `http_timeout_secs`
    for `model` > `fallback_s` (the dojo's historical 300 s)."""
    if flag is not None:
        return ResolvedTimeout(flag, TimeoutSource.FLAG, model=model, effort=effort)
    if catalog is not None:
        row, profile = catalog_profile(model, effort, catalog)
        seconds = _seconds(profile.get("http_timeout_secs"))
        if seconds is not None:
            return ResolvedTimeout(seconds, TimeoutSource.CATALOG, seconds, 0.0, model, effort, row)
    return ResolvedTimeout(fallback_s, TimeoutSource.FALLBACK, None, 0.0, model, effort, None,
                           "the catalog has no http_timeout_secs for this model")
