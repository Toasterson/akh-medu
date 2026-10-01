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


# --- sampling (ADR 123 amendment 2026-10-01) ---------------------------------------------------------------


REPEAT_PENALTY_MIN = 1.0
REPEAT_PENALTY_MAX = 2.0
# `--model-profile` values that mean "resolve from --model" and "send no catalog sampling".
PROFILE_AUTO = "auto"
PROFILE_NONE = "none"


class SamplingSource(enum.StrEnum):
    FLAG = "flag"        # --repeat-penalty: taken as given (range-checked)
    CATALOG = "catalog"  # the catalog row's `repeat_penalty`
    NONE = "none"        # nothing is sent: the server's default sampling
    AKH = "akh"          # the akh samples (the acp arm); the dojo sends nothing


@dataclasses.dataclass(frozen=True)
class ResolvedSampling:
    """What the openai arm samples with, and where each value came from. Recorded in provenance so
    two baselines are comparable only when they sampled alike (the deployed Lilith sends
    `repeat_penalty` from the same catalog row; a baseline without it measures a different model
    behaviour: 5/10 empty-thought loops at default sampling, 0/10 at 1.1)."""
    repeat_penalty: float | None
    source: SamplingSource
    catalog_row: str | None = None
    # The row's per-request output cap, recorded for comparison with --max-tokens; not applied here.
    catalog_max_output_tokens: int | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: (v.value if isinstance(v, enum.Enum) else v) for k, v in dataclasses.asdict(self).items()
                if v is not None or k == "repeat_penalty"}


def check_repeat_penalty(value: float) -> float:
    """The value, or a DojoError naming the accepted range (same range as `ModelProfile::validate`)."""
    from dojo.errors import DojoError

    if not (REPEAT_PENALTY_MIN <= value <= REPEAT_PENALTY_MAX):  # also false for NaN
        raise DojoError(f"--repeat-penalty {value} is outside {REPEAT_PENALTY_MIN}..={REPEAT_PENALTY_MAX}",
                        help="1.0 is off, 1.05 to 1.15 is the useful band (the Gemma 4 12B loop guard uses 1.1); "
                             "omit the flag to take the catalog row, or pass --model-profile none to send nothing")
    return value


def resolve_sampling(model: str | None, catalog: Mapping[str, Any] | None, *, profile: str = PROFILE_AUTO,
                     repeat_penalty: float | None = None) -> ResolvedSampling:
    """The openai arm's sampling: `--repeat-penalty` > the catalog row's `repeat_penalty` > none.

    `profile` names the catalog row to read: `auto` (the default) looks up `model` the way the akh
    resolves its own model id, `none` sends no catalog sampling, anything else is a row id or alias
    (`gemma-4-12b`, `lilith`). A named row that the catalog lacks, or a missing catalog, is refused
    rather than skipped: a silently unsampled baseline would not be comparable with the deployed akh."""
    from dojo.errors import DojoError

    row: str | None = None
    found: dict = {}
    if profile != PROFILE_NONE:
        lookup = model if profile == PROFILE_AUTO else profile
        if profile != PROFILE_AUTO and catalog is None:
            raise DojoError("--model-profile needs data/model-profiles/catalog.toml, which is missing or unreadable",
                            help="run from a full akh-medu checkout (--repo) or pass --repeat-penalty directly")
        if lookup is not None and catalog is not None:
            row, found = catalog_profile(lookup, None, catalog)
            if row is None and profile != PROFILE_AUTO:
                raise DojoError(f"--model-profile {profile!r} matches no row in the model catalog",
                                help="use a row id or alias from data/model-profiles/catalog.toml "
                                     "(for Lilith: gemma-4-12b or lilith), `auto`, or `none`")
    cap = found.get("max_output_tokens")
    if repeat_penalty is not None:
        return ResolvedSampling(check_repeat_penalty(float(repeat_penalty)), SamplingSource.FLAG, row, cap)
    penalty = found.get("repeat_penalty")
    if isinstance(penalty, (int, float)) and not isinstance(penalty, bool):
        return ResolvedSampling(float(penalty), SamplingSource.CATALOG, row, cap)
    note = None if profile == PROFILE_NONE else "the catalog row sets no repeat_penalty"
    return ResolvedSampling(None, SamplingSource.NONE, row, cap, note)
