#!/usr/bin/env python3
"""Isolate why GPU-resident and CPU-offloaded tokenwise KV caches diverge on CUDA.

Follow-up to the first C6 H100 smoke (Anima WI 01a0e270-6021): the cache control
(``lilith_training_cache_control.py``) failed equivalence on 1x H100 (logit max_abs
3.81/0.875/0.5625 at 1023/1024/1025) while the same control matched to 2.9e-11 on
ROCm. This script does not change the qualified engine arithmetic; it re-runs the
resident-versus-offload comparison under controlled variations and records, per
variation, the max_abs at every probe position:

  e  baseline + repeat: resident vs offload, resident vs resident, offload vs offload
     (the run-to-run nondeterminism floor)
  a  TF32 off (matmul + cuDNN); a2: bf16 reduced-precision GEMM reductions off
  b  each SDPA backend forced (MATH, EFFICIENT_ATTENTION, FLASH_ATTENTION,
     CUDNN_ATTENTION); a backend with no kernel for a call (flash on the head_dim
     512 global layers) is recorded and retried with MATH as the fallback
  c  dense cache storage in both variants (resident stored sliding-window views
     and offloaded tensors on the way back become fresh contiguous copies with
     storage offset 0 and the same dtype), plus the strides / storage offsets /
     is_pinned of every stored K/V in every pass (``layout``)
  d  torch.use_deterministic_algorithms(True) with CUBLAS_WORKSPACE_CONFIG=:4096:8;
     a strict failure names the nondeterministic op and is retried warn-only

Two prefix modes keep the whole run inside ``--budget-seconds`` (default 1800):
  chunked    one no-grad prefill up to ``--tail-tokens`` (default 8) before the first
             probe, then tokenwise; every variation (with a resident repeat) runs here
             first, in information order: e (resident twice, then offload), a TF32 off,
             d deterministic, b SDPA MATH, c contiguous, then a2 and the other backends
  tokenwise  exactly the control's path (every token a separate forward): resident
             twice, the best chunked-restoring variation, the baseline offload pass,
             then the other variations, until ``--budget-seconds`` is spent
Cases whose estimated cost does not fit the remaining budget are recorded as
``skipped_budget``; a pass that overruns the deadline stops at the next token and is
recorded as ``timeout``, so the sweep always ends inside the budget.

The JSON output (``lilith/cache-path-diagnostic@1``) is rewritten after every case,
including ``modes_summary`` and a provisional ``verdict`` (``"final": false``), so a
step timeout or kill still leaves a verdict on disk. The final ``verdict`` gives the
cause class, which variations restore equivalence (confirmed on the tokenwise path or
chunked-only), and the numerics configuration
to try (``lilith_numerics`` flags) when the fix is configuration-only. The 1e-4
tolerance is the control's and is never loosened here.

``--from-report FILE`` recomputes the verdict from a saved report without a GPU or
model (for reports cut short before the provisional verdict existed).

Exit status: 0 when the diagnostic completed (whatever the verdict), 1 on error.
"""
from __future__ import annotations

import os

# cuBLAS reads this when its handle is created; set it before torch initialises CUDA.
_CUBLAS_PRESET = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import collections  # noqa: E402
from dataclasses import asdict, dataclass, field, replace  # noqa: E402
import enum  # noqa: E402
import functools  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable, Iterable, Sequence  # noqa: E402
import warnings  # noqa: E402

from lilith_numerics import NumericsConfig, SdpaBackend, Toggle  # noqa: E402

SCHEMA = "lilith/cache-path-diagnostic@1"
TOLERANCE = 1e-4          # the cache control's tolerance; never loosened
DEFAULT_POSITIONS = (1023, 1024, 1025)
SELECTED_TOKEN = 101      # the control's scored token
DEFAULT_BUDGET_SECONDS = 1800.0
DEFAULT_TAIL_TOKENS = 8
# Initial per-case duration guesses (seconds) until a case of the mode has run.
INITIAL_ESTIMATE = {"chunked": 20.0, "tokenwise": 150.0}


class PassDeadlineExceeded(RuntimeError):
    """A pass reached the diagnostic deadline; recorded as ``timeout``."""


class DiagnosticError(RuntimeError):
    """The diagnostic could not run; the message names the failed precondition."""


class PrefixMode(str, enum.Enum):
    CHUNKED = "chunked"
    TOKENWISE = "tokenwise"


class CacheKind(str, enum.Enum):
    RESIDENT = "resident"                        # engine._cache(), cache_offloading=False
    OFFLOAD = "offload"                          # engine._cache(), cache_offloading=True
    RESIDENT_CONTIGUOUS = "resident_contiguous"  # engine layers, stored K/V made contiguous
    OFFLOAD_CONTIGUOUS = "offload_contiguous"    # engine layers, contiguous on the way back


@dataclass(frozen=True)
class Variation:
    name: str
    letter: str
    description: str
    numerics: NumericsConfig = NumericsConfig()
    resident: CacheKind = CacheKind.RESIDENT
    offload: CacheKind = CacheKind.OFFLOAD
    # Configuration-only fixes can be applied through lilith_numerics flags; the
    # contiguity variation would need an engine change and a new qualification.
    configuration_only: bool = True


BASELINE = Variation("baseline", "e", "qualified defaults; resident twice (the repeat floor), then offload")
# Information order (round-3 request): a, d, b MATH, c first, then the rest.
VARIATIONS: tuple[Variation, ...] = (
    Variation("tf32_off", "a", "torch.backends.cuda.matmul.allow_tf32=False, cudnn.allow_tf32=False",
              NumericsConfig(tf32=Toggle.OFF)),
    Variation("deterministic", "d", "torch.use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG=:4096:8",
              NumericsConfig(deterministic=True)),
    Variation("sdpa_math", "b", "sdpa_kernel(MATH)", NumericsConfig(sdpa_backend=SdpaBackend.MATH)),
    Variation("contiguous", "c", "stored K/V contiguous in both variants (engine change if it is the fix)",
              resident=CacheKind.RESIDENT_CONTIGUOUS, offload=CacheKind.OFFLOAD_CONTIGUOUS,
              configuration_only=False),
    Variation("bf16_reduction_off", "a2", "matmul.allow_bf16_reduced_precision_reduction=False",
              NumericsConfig(bf16_reduced_precision_reduction=Toggle.OFF)),
    Variation("sdpa_efficient", "b", "sdpa_kernel(EFFICIENT_ATTENTION)",
              NumericsConfig(sdpa_backend=SdpaBackend.EFFICIENT)),
    Variation("sdpa_flash", "b", "sdpa_kernel(FLASH_ATTENTION); head_dim 512 global layers have no flash kernel",
              NumericsConfig(sdpa_backend=SdpaBackend.FLASH)),
    Variation("sdpa_cudnn", "b", "sdpa_kernel(CUDNN_ATTENTION)", NumericsConfig(sdpa_backend=SdpaBackend.CUDNN)),
)


def fallback_for(variation: Variation) -> Variation | None:
    """The retry scheduled when a variation's forced setting cannot run at all."""
    numerics = variation.numerics
    if numerics.deterministic and not numerics.deterministic_warn_only:
        return replace(variation, name=variation.name + "_warn_only",
                       description=variation.description + "; warn-only after a strict failure",
                       numerics=replace(numerics, deterministic_warn_only=True))
    if numerics.sdpa_backend not in (SdpaBackend.DEFAULT, SdpaBackend.MATH) and not numerics.sdpa_math_fallback:
        return replace(variation, name=variation.name + "_math_fallback",
                       description=variation.description + "; MATH where the backend has no kernel",
                       numerics=replace(numerics, sdpa_math_fallback=True))
    return None


@dataclass(frozen=True)
class Case:
    mode: PrefixMode
    variation: Variation
    role: str             # "resident" | "offload"
    repeat: int = 0

    @property
    def key(self) -> str:
        return f"{self.mode.value}/{self.variation.name}/{self.role}" + (f"#{self.repeat + 1}" if self.repeat else "")

    @property
    def cache_kind(self) -> CacheKind:
        return self.variation.resident if self.role == "resident" else self.variation.offload


@dataclass
class PassResult:
    """One resident or offloaded tokenwise pass; tensors stay in memory, summaries go to JSON."""
    case: Case
    status: str                                   # ok | error | skipped_budget
    seconds: float = 0.0
    logits: dict[int, Any] = field(default_factory=dict)
    gradients: dict[str, Any] = field(default_factory=dict)
    loss: float | None = None
    layout: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    peak_gpu_bytes: int | None = None

    def _layout_summary(self) -> dict[str, Any] | None:
        """Per-layer detail for baseline and contiguity passes; distinct layouts for the rest."""
        if self.layout is None or self.case.variation.name in (BASELINE.name, "contiguous"):
            return self.layout
        return {"distinct_key_layouts": self.layout["distinct_key_layouts"]}

    def summary(self) -> dict[str, Any]:
        return {"mode": self.case.mode.value, "variation": self.case.variation.name,
                "role": self.case.role, "repeat": self.case.repeat,
                "cache_kind": self.case.cache_kind.value,
                "numerics": self.case.variation.numerics.requested(),
                "status": self.status, "seconds": round(self.seconds, 3), "loss": self.loss,
                "gradient_norms": {name: float(value.norm()) for name, value in self.gradients.items()},
                "layout": self._layout_summary(), "warnings": self.warnings, "error": self.error,
                "peak_gpu_bytes": self.peak_gpu_bytes}


# ------------------------------------------------------------------- comparisons
def compare(left: PassResult, right: PassResult, tolerance: float = TOLERANCE) -> dict[str, Any] | None:
    """The control's hard values for two passes, or None when either pass has no data."""
    if left.status != "ok" or right.status != "ok":
        return None
    logits = {str(position): float((left.logits[position] - right.logits[position]).abs().max())
              for position in sorted(left.logits)}
    gradients = {name: float((left.gradients[name] - right.gradients[name]).abs().max())
                 for name in sorted(left.gradients)}
    loss = abs(left.loss - right.loss)
    hard = list(logits.values()) + [loss] + list(gradients.values())
    bitwise = all(bool((left.logits[p] == right.logits[p]).all()) for p in left.logits)
    return {"left": left.case.key, "right": right.case.key, "logit_max_abs": logits, "loss_abs": loss,
            "gradient_max_abs": gradients, "max_abs": max(hard), "within_tolerance": max(hard) <= tolerance,
            "logits_bitwise_equal": bitwise}


def layout_differences(left: dict[str, Any] | None, right: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Per-layer K/V storage attributes that differ between two cache layouts."""
    if not left or not right:
        return []
    differences = []
    for a, b in zip(left["layers"], right["layers"]):
        for tensor in ("keys", "values"):
            fields = {name: [a[tensor].get(name), b[tensor].get(name)]
                      for name in ("stride", "storage_offset", "is_contiguous", "is_pinned", "dtype", "shape")
                      if a[tensor].get(name) != b[tensor].get(name)}
            if fields:
                differences.append({"layer": a["index"], "tensor": tensor, "is_sliding": a["is_sliding"],
                                    "fields": fields})
    return differences


def _tensor_layout(tensor: Any) -> dict[str, Any]:
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "storage_offset": int(tensor.storage_offset()), "is_contiguous": bool(tensor.is_contiguous()),
            "dtype": str(tensor.dtype), "device": str(tensor.device),
            "is_pinned": bool(tensor.is_pinned()) if tensor.device.type == "cpu" else None}


def cache_layout(cache: Any) -> dict[str, Any]:
    layers = []
    for index, layer in enumerate(cache.layers):
        if not layer.is_initialized:
            continue
        layers.append({"index": index, "is_sliding": bool(getattr(layer, "is_sliding", False)),
                       "keys": _tensor_layout(layer.keys), "values": _tensor_layout(layer.values)})
    signatures = collections.Counter(
        json.dumps({"is_sliding": layer["is_sliding"],
                    **{k: layer["keys"][k] for k in ("stride", "storage_offset", "is_contiguous",
                                                      "is_pinned", "dtype", "device")}}, sort_keys=True)
        for layer in layers)
    return {"layers": layers,
            "distinct_key_layouts": [dict(json.loads(sig), layers=count) for sig, count in sorted(signatures.items())]}


# ------------------------------------------------------------------ cache kinds
def _dense(tensor: Any) -> Any:
    """A fresh dense copy: contiguous strides and storage offset 0 (``.contiguous()`` is a no-op
    on a view that is already contiguous but still offset into a larger storage)."""
    import torch

    return tensor.clone(memory_format=torch.contiguous_format)


@functools.cache
def _diagnostic_cache_classes() -> tuple[type, type]:
    from transformers import Cache

    class ContiguousResidentCache(Cache):
        """The resident cache, with every stored K/V made a dense contiguous tensor."""

        def update(self, key_states, value_states, layer_idx, *args, **kwargs):
            layer = self.layers[layer_idx]
            keys, values = layer.update(key_states, value_states, *args, **kwargs)
            layer.keys = _dense(layer.keys)
            layer.values = _dense(layer.values)
            return keys, values

    class ContiguousOffloadedCache(Cache):
        """The engine's synchronous offload with contiguous, dtype-checked copies both ways."""

        def update(self, key_states, value_states, layer_idx, *args, **kwargs):
            layer = self.layers[layer_idx]
            if layer.is_initialized:
                keys = _dense(layer.keys.to(key_states.device, non_blocking=False))
                values = _dense(layer.values.to(value_states.device, non_blocking=False))
                if keys.dtype != key_states.dtype or values.dtype != value_states.dtype:
                    raise DiagnosticError("offloaded K/V changed dtype on the way back")
                layer.keys, layer.values = keys, values
            keys, values = layer.update(key_states, value_states, *args, **kwargs)
            layer.keys = _dense(layer.keys.to("cpu", non_blocking=False))
            layer.values = _dense(layer.values.to("cpu", non_blocking=False))
            return keys, values

    return ContiguousResidentCache, ContiguousOffloadedCache


def make_cache(engine: Any, kind: CacheKind) -> Any:
    """Build the cache for ``kind``; the layer list always comes from the engine."""
    original = engine.cache_offloading
    try:
        engine.cache_offloading = kind is CacheKind.OFFLOAD
        cache = engine._cache()
    finally:
        engine.cache_offloading = original
    if kind in (CacheKind.RESIDENT, CacheKind.OFFLOAD):
        return cache
    resident_cls, offload_cls = _diagnostic_cache_classes()
    cls = resident_cls if kind is CacheKind.RESIDENT_CONTIGUOUS else offload_cls
    return cls(layers=cache.layers)


# ---------------------------------------------------------------------- passes
def probe_tokens(positions: Sequence[int]) -> list[int]:
    """The control's token stream."""
    return [2] + [100 + index % 40 for index in range(max(positions) - 1)]


def chunked_prefix_length(positions: Sequence[int], tail_tokens: int) -> int:
    """Tokens prefilled in one no-grad forward before the tokenwise tail (at least one)."""
    return max(1, positions[0] - 1 - tail_tokens)


def run_pass(engine: Any, case: Case, positions: Sequence[int], tail_tokens: int,
             deadline: float | None = None, clock: Callable[[], float] = time.monotonic) -> PassResult:
    """One pass; stops at the next token once ``deadline`` (on ``clock``) has passed."""
    import torch

    def check_deadline() -> None:
        if deadline is not None and clock() > deadline:
            raise PassDeadlineExceeded("diagnostic budget exhausted inside the pass")

    model = engine.model
    tokens = probe_tokens(positions)
    started = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    result = PassResult(case=case, status="error")
    cache = current = loss = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with case.variation.numerics.applied():
                cache = make_cache(engine, case.cache_kind)
                model.zero_grad(set_to_none=True)
                start = 0
                if case.mode is PrefixMode.CHUNKED:
                    start = chunked_prefix_length(positions, tail_tokens)
                    engine._forward(tokens[:start], cache, 0, grad=False)
                for index in range(start, len(tokens)):
                    check_deadline()
                    position = index + 1
                    selected = position in positions
                    current = engine._forward([tokens[index]], cache, index, grad=selected)
                    if selected:
                        result.logits[position] = current.detach().float().cpu()
                        term = -torch.log_softmax(current, dim=-1)[SELECTED_TOKEN] / len(positions)
                        loss = term if loss is None else loss + term
                if loss is None:
                    raise DiagnosticError("no probe position was reached")
                result.loss = float(loss.detach().cpu())
                loss.backward()
                result.gradients = {name: parameter.grad.detach().float().cpu().clone()
                                    for name, parameter in model.named_parameters()
                                    if parameter.requires_grad and parameter.grad is not None}
                result.layout = cache_layout(cache)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        result.warnings = sorted({str(item.message)[:400] for item in caught
                                  if "deterministic" in str(item.message).lower()})
        result.status = "ok"
    except PassDeadlineExceeded as error:
        result.status = "timeout"
        result.error = f"{type(error).__name__}: {error}"
    except Exception as error:  # recorded per case; the sweep continues
        result.error = f"{type(error).__name__}: {error}"[:2000]
    finally:
        del cache, current, loss
        model.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            result.peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
            torch.cuda.empty_cache()
        result.seconds = time.monotonic() - started
    return result


# ------------------------------------------------------------------- planning
def variation_cases(mode: PrefixMode, variation: Variation, *, resident_repeat: bool,
                    offload_repeat: bool = False) -> list[Case]:
    cases = [Case(mode, variation, "resident"), Case(mode, variation, "offload")]
    if resident_repeat:
        cases.append(Case(mode, variation, "resident", 1))
    if offload_repeat:
        cases.append(Case(mode, variation, "offload", 1))
    return cases


def summarize_mode(mode: PrefixMode, results: dict[str, PassResult], variations: Iterable[Variation],
                   tolerance: float = TOLERANCE) -> dict[str, Any]:
    """Resident-vs-offload and repeat comparisons for one prefix mode."""
    def get(variation: Variation, role: str, repeat: int = 0) -> PassResult | None:
        return results.get(Case(mode, variation, role, repeat).key)

    def pair(a: PassResult | None, b: PassResult | None) -> dict[str, Any] | None:
        return compare(a, b, tolerance) if a is not None and b is not None else None

    base_r, base_o = get(BASELINE, "resident"), get(BASELINE, "offload")
    summary: dict[str, Any] = {
        "baseline": {
            "resident_vs_offload": pair(base_r, base_o),
            "resident_vs_resident_repeat": pair(base_r, get(BASELINE, "resident", 1)),
            "offload_vs_offload_repeat": pair(base_o, get(BASELINE, "offload", 1)),
            "layout_differences": layout_differences(base_r.layout if base_r else None,
                                                     base_o.layout if base_o else None),
        },
        "variations": {},
    }
    for variation in variations:
        resident, offload = get(variation, "resident"), get(variation, "offload")
        if resident is None and offload is None:
            continue
        entry = {
            "letter": variation.letter, "description": variation.description,
            "numerics": variation.numerics.requested(), "configuration_only": variation.configuration_only,
            "status": {"resident": resident.status if resident else "not_run",
                       "offload": offload.status if offload else "not_run"},
            "errors": sorted({r.error for r in (resident, offload) if r is not None and r.error}),
            "resident_vs_offload": pair(resident, offload),
            "resident_vs_resident_repeat": pair(resident, get(variation, "resident", 1)),
            "resident_vs_baseline_resident": pair(resident, base_r),
            "layout_differences": layout_differences(resident.layout if resident else None,
                                                     offload.layout if offload else None),
        }
        entry["restores_equivalence"] = restores(entry)
        summary["variations"][variation.name] = entry
    return summary


def restores(entry: dict[str, Any]) -> bool:
    """Resident == offload within tolerance, and repeatable when a repeat ran."""
    main = entry.get("resident_vs_offload")
    repeat = entry.get("resident_vs_resident_repeat")
    return bool(main and main["within_tolerance"] and (repeat is None or repeat["within_tolerance"]))


def _within(comparison: dict[str, Any] | None) -> bool | None:
    return None if comparison is None else comparison["within_tolerance"]


def verdict(modes: dict[str, dict[str, Any]], variations: Sequence[Variation]) -> dict[str, Any]:
    """Classify the cause and name the variation(s) that restore equivalence."""
    by_name = {variation.name: variation for variation in variations}
    per_mode = {}
    for mode, summary in modes.items():
        baseline = summary["baseline"]
        per_mode[mode] = {
            "reproduced": (None if baseline["resident_vs_offload"] is None
                           else not baseline["resident_vs_offload"]["within_tolerance"]),
            "resident_repeatable": _within(baseline["resident_vs_resident_repeat"]),
            "offload_repeatable": _within(baseline["offload_vs_offload_repeat"]),
            "restoring_variations": sorted(
                (name for name, entry in summary["variations"].items() if entry["restores_equivalence"]),
                key=lambda name: summary["variations"][name]["resident_vs_offload"]["max_abs"]),
        }
    tokenwise = per_mode.get(PrefixMode.TOKENWISE.value, {})
    chunked = per_mode.get(PrefixMode.CHUNKED.value, {})
    primary = tokenwise if tokenwise.get("reproduced") is not None else chunked
    reproduced = [value["reproduced"] for value in per_mode.values() if value["reproduced"] is not None]
    if not reproduced:
        cause = "inconclusive"
        explanation = "no baseline resident/offload pair completed"
    elif not any(reproduced):
        cause = "not_reproduced"
        explanation = "resident and offloaded caches matched within tolerance in every mode that ran"
    elif any(value.get("resident_repeatable") is False or value.get("offload_repeatable") is False
             for value in per_mode.values()):
        cause = "run_to_run_nondeterminism"
        explanation = ("the same cache path run twice already differs beyond tolerance, so the offload "
                       "copy is not the cause; the CUDA kernels are not repeatable at this tolerance")
    elif primary.get("resident_repeatable") is True:
        cause = "offload_path_specific"
        explanation = ("each path repeats bit-for-bit within tolerance but resident and offloaded differ, "
                       "so the offload round trip changes what the kernels see (layout/alignment/backend)")
    else:
        cause = "inconclusive"
        explanation = "divergence reproduced but no repeat pair completed to separate the two causes"

    confirmed = list(tokenwise.get("restoring_variations", []))
    chunked_only = [name for name in chunked.get("restoring_variations", []) if name not in confirmed]
    divergent = cause not in ("not_reproduced", "inconclusive")
    recommended = (next((by_name[name] for name in confirmed if by_name[name].configuration_only), None)
                   if divergent else None)
    engine_change = divergent and bool(confirmed) and recommended is None
    return {
        "cause": cause,
        "explanation": explanation,
        "tolerance": TOLERANCE,
        "per_mode": per_mode,
        "restoring_variations_confirmed_tokenwise": confirmed,
        "restoring_variations_chunked_only": chunked_only,
        "recommended_numerics": recommended.numerics.requested() if recommended else None,
        "recommended_numerics_flags": numerics_flags(recommended.numerics) if recommended else None,
        "fix_requires_engine_change": engine_change,
        "next_step": _next_step(cause, recommended, engine_change, chunked_only),
    }


def numerics_flags(config: NumericsConfig) -> list[str]:
    """``lilith_numerics.add_arguments`` flags reproducing ``config`` (warn-only is never a fix)."""
    flags = []
    if config.sdpa_backend is not SdpaBackend.DEFAULT:
        flags += ["--numerics-sdpa-backend", config.sdpa_backend.value]
        if config.sdpa_math_fallback:
            flags.append("--numerics-sdpa-math-fallback")
    if config.tf32 is not Toggle.DEFAULT:
        flags += ["--numerics-tf32", config.tf32.value]
    if config.bf16_reduced_precision_reduction is not Toggle.DEFAULT:
        flags += ["--numerics-bf16-reduced-precision-reduction", config.bf16_reduced_precision_reduction.value]
    if config.deterministic:
        flags.append("--numerics-deterministic")
    return flags


def _next_step(cause: str, recommended: Variation | None, engine_change: bool, chunked_only: list[str]) -> str:
    if cause == "not_reproduced":
        return "rerun the cache control; the round-1 divergence did not reproduce here"
    if recommended is not None:
        if recommended.numerics.deterministic_warn_only:
            return ("only the warn-only deterministic run restored equivalence; the listed ops need a "
                    "deterministic replacement before this is a fix")
        return ("rerun the cache control with " + " ".join(numerics_flags(recommended.numerics)) +
                " (LILITH_NUMERICS_* in run_smoke.sh); it is recorded in the run manifest")
    if engine_change:
        return "only the contiguity variation restored equivalence: an engine change plus a new qualification"
    if chunked_only:
        return ("restoring variations were seen on the chunked path only; rerun with a larger "
                "--budget-seconds to confirm them tokenwise: " + ", ".join(chunked_only))
    return "no variation restored equivalence; escalate (kernel-level investigation, or accept offload off)"


def known_variations() -> list[Variation]:
    """Every variation a report can contain: the sweep plus each scheduled fallback."""
    known = list(VARIATIONS)
    for variation in VARIATIONS:
        fallback = fallback_for(variation)
        while fallback is not None:
            known.append(fallback)
            fallback = fallback_for(fallback)
    return known


def verdict_from_report(report: dict[str, Any]) -> dict[str, Any]:
    """The verdict of a saved report's ``modes_summary`` (e.g. one cut short by a step timeout)."""
    summary = report.get("modes_summary")
    if not summary:
        raise ValueError("the report has no modes_summary (no case finished)")
    return {**verdict(summary, known_variations()), "final": report.get("status") == "finished",
            "recomputed_from_report": True, "report_status": report.get("status"),
            "passes_recorded": len(report.get("passes", {}))}


class Planner:
    """Budget-aware case queue: chunked sweep first, then tokenwise confirmation."""

    def __init__(self, modes: Sequence[PrefixMode], variations: Sequence[Variation], budget_seconds: float,
                 clock: Callable[[], float] = time.monotonic):
        self.modes = list(modes)
        self.variations = list(variations)
        self.clock = clock
        self.deadline = clock() + budget_seconds
        self.queue: collections.deque[Case] = collections.deque()
        self.durations: dict[PrefixMode, list[float]] = collections.defaultdict(list)
        # Observed seconds per (mode, variation name, role), for per-case estimates.
        self.case_seconds: dict[tuple[PrefixMode, str, str], float] = {}
        # (mode, failed variation name) pairs whose fallback has been scheduled.
        self.scheduled_fallbacks: set[tuple[PrefixMode, str]] = set()
        self.all_variations: list[Variation] = list(variations)
        if PrefixMode.CHUNKED in self.modes:
            self._schedule_sweep(PrefixMode.CHUNKED, self.variations, repeat=True)
        elif PrefixMode.TOKENWISE in self.modes:
            self._schedule_sweep(PrefixMode.TOKENWISE, self.variations, repeat=False)

    def _schedule_sweep(self, mode: PrefixMode, variations: Iterable[Variation], repeat: bool) -> None:
        # e first: resident twice is the nondeterminism floor, then the offload pass.
        self.queue.extend([Case(mode, BASELINE, "resident"), Case(mode, BASELINE, "resident", 1),
                           Case(mode, BASELINE, "offload")])
        for variation in variations:
            self.queue.extend(variation_cases(mode, variation, resident_repeat=repeat))

    def estimate(self, case: Case) -> float:
        """Seconds a case is expected to take.

        Same (mode, variation, role) observed: that. Tokenwise with a chunked observation
        of the same variation/role: the tokenwise baseline resident time scaled by that
        case's chunked cost relative to the chunked baseline resident (offload and
        deterministic passes are several times slower). Otherwise the recent maximum of
        the mode, or the initial guess."""
        key = (case.mode, case.variation.name, case.role)
        if key in self.case_seconds:
            return self.case_seconds[key] * 1.1
        if case.mode is PrefixMode.TOKENWISE:
            chunked = self.case_seconds.get((PrefixMode.CHUNKED, case.variation.name, case.role))
            chunked_base = self.case_seconds.get((PrefixMode.CHUNKED, BASELINE.name, "resident"))
            tokenwise_base = self.case_seconds.get((PrefixMode.TOKENWISE, BASELINE.name, "resident"))
            if chunked and chunked_base and tokenwise_base:
                return tokenwise_base * max(1.0, chunked / chunked_base) * 1.1
        observed = self.durations[case.mode]
        return max(observed[-3:]) * 1.1 if observed else INITIAL_ESTIMATE[case.mode.value]

    def fits(self, case: Case) -> bool:
        return self.clock() + self.estimate(case) <= self.deadline

    def record(self, result: PassResult) -> None:
        if result.status in ("ok", "timeout"):
            self.durations[result.case.mode].append(result.seconds)
            key = (result.case.mode, result.case.variation.name, result.case.role)
            self.case_seconds[key] = max(self.case_seconds.get(key, 0.0), result.seconds)
        elif result.status == "error" and result.case.role == "resident" and result.case.repeat == 0:
            fallback = fallback_for(result.case.variation)
            marker = (result.case.mode, result.case.variation.name)
            if fallback is not None and marker not in self.scheduled_fallbacks:
                self.scheduled_fallbacks.add(marker)
                if fallback not in self.all_variations:
                    self.all_variations.append(fallback)
                # Drop the failed variation's remaining cases in this mode; its fallback replaces them.
                self.queue = collections.deque(
                    case for case in self.queue
                    if not (case.mode is result.case.mode and case.variation == result.case.variation))
                self.queue.extendleft(reversed(variation_cases(
                    result.case.mode, fallback, resident_repeat=result.case.mode is PrefixMode.CHUNKED)))

    def schedule_confirmation(self, chunked_summary: dict[str, Any] | None) -> None:
        """Tokenwise order, most informative first (the budget usually stops partway):

        resident twice (e: the nondeterminism floor on the control's exact path), the best
        chunked-restoring variation's resident+offload pair (confirms the fix), the baseline
        offload pass (re-measures the round-1 divergence), then the remaining restoring
        variations and finally the rest."""
        if PrefixMode.TOKENWISE not in self.modes or PrefixMode.CHUNKED not in self.modes:
            return
        order: list[str] = []
        if chunked_summary is not None:
            entries = chunked_summary["variations"]

            def rank(name: str) -> tuple[float, float]:
                # Closest match first; among equals, the cheapest offload pass (a
                # deterministic offload pass ran ~6x slower than a plain one in round 2).
                cost = self.case_seconds.get((PrefixMode.CHUNKED, name, "offload"), float("inf"))
                return entries[name]["resident_vs_offload"]["max_abs"], cost

            order = sorted((name for name, entry in entries.items() if entry["restores_equivalence"]),
                           key=rank)
        by_name = {variation.name: variation for variation in self.all_variations}
        # A variation that could not run chunked is represented by its fallback.
        runnable = [variation for variation in self.all_variations
                    if (PrefixMode.CHUNKED, variation.name) not in self.scheduled_fallbacks]
        ranked = [by_name[name] for name in order] + [
            variation for variation in runnable if variation.name not in order]
        self.queue.extend([Case(PrefixMode.TOKENWISE, BASELINE, "resident"),
                           Case(PrefixMode.TOKENWISE, BASELINE, "resident", 1)])
        for index, variation in enumerate(ranked):
            self.queue.extend(variation_cases(PrefixMode.TOKENWISE, variation, resident_repeat=False))
            if index == 0:
                self.queue.append(Case(PrefixMode.TOKENWISE, BASELINE, "offload"))
        if not ranked:
            self.queue.append(Case(PrefixMode.TOKENWISE, BASELINE, "offload"))


# --------------------------------------------------------------------- runner
def model_summary(engine: Any) -> dict[str, Any]:
    config = engine.model.config.get_text_config()
    keys = ("num_hidden_layers", "num_kv_shared_layers", "sliding_window", "head_dim", "global_head_dim",
            "num_attention_heads", "num_key_value_heads", "num_global_key_value_heads", "enable_moe_block",
            "num_experts", "hidden_size", "vocab_size", "_attn_implementation")
    summary = {key: getattr(config, key, None) for key in keys}
    summary["layer_types"] = dict(collections.Counter(getattr(config, "layer_types", []) or []))
    summary["experts_implementation"] = getattr(config, "_experts_implementation", None)
    return summary


def run_diagnostic(engine: Any, output: Path | str, *, positions: Sequence[int] = DEFAULT_POSITIONS,
                   modes: Sequence[PrefixMode] = (PrefixMode.CHUNKED, PrefixMode.TOKENWISE),
                   variations: Sequence[Variation] = VARIATIONS, tail_tokens: int = DEFAULT_TAIL_TOKENS,
                   budget_seconds: float = DEFAULT_BUDGET_SECONDS,
                   clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Run the sweep; the report is rewritten after every case so a timeout keeps evidence."""
    import torch
    import transformers
    import lilith_training_engine as engine_module
    from lilith_numerics import observed_state
    from lilith_training_cache_control import _seed_nonzero_adapter

    output = Path(output)
    positions = tuple(int(position) for position in positions)
    if not positions or tuple(sorted(set(positions))) != positions or positions[0] < 1:
        raise DiagnosticError("positions must be unique positive integers in ascending order")
    if max(positions) > engine.context_tokens:
        raise DiagnosticError("probe positions exceed the engine context bound")
    if tail_tokens < 0:
        raise DiagnosticError("--tail-tokens must be >= 0")
    started = clock()
    planner = Planner(modes, variations, budget_seconds, clock)
    first_parameter = next(engine.model.parameters())
    initial_policy = engine.policy_identity()
    report: dict[str, Any] = {
        "schema": SCHEMA, "status": "running", "tolerance": TOLERANCE, "positions": list(positions),
        "modes": [mode.value for mode in modes], "tail_tokens": tail_tokens,
        "chunked_prefix_tokens": chunked_prefix_length(positions, tail_tokens),
        "budget_seconds": budget_seconds,
        "engine_sha256": hashlib.sha256(Path(engine_module.__file__).read_bytes()).hexdigest(),
        "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
                     "cuda": torch.version.cuda, "hip": torch.version.hip,
                     "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None},
        "runtime": {"device": str(first_parameter.device), "dtype": str(first_parameter.dtype),
                    "device_name": (torch.cuda.get_device_name(first_parameter.device)
                                    if first_parameter.device.type == "cuda" else None),
                    "cublas_workspace_config_preset": _CUBLAS_PRESET},
        "numerics_at_start": observed_state(),
        "model": model_summary(engine),
        "initial_policy_identity": {**asdict(initial_policy), "digest": initial_policy.digest},
        "variations": {variation.name: {"letter": variation.letter, "description": variation.description,
                                        "numerics": variation.numerics.requested(),
                                        "configuration_only": variation.configuration_only}
                       for variation in (BASELINE, *variations)},
        "passes": {},
    }
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")

    def persist() -> None:
        report["elapsed_seconds"] = clock() - started
        output.write_text(json.dumps(report, indent=2) + "\n")

    results: dict[str, PassResult] = {}
    snapshot = engine.snapshot_adapter()
    original_offloading = engine.cache_offloading
    confirmation_scheduled = False
    try:
        report["trainable_names"] = _seed_nonzero_adapter(engine)
        while True:
            if not planner.queue and not confirmation_scheduled:
                confirmation_scheduled = True
                chunked = (summarize_mode(PrefixMode.CHUNKED, results, planner.all_variations)
                           if PrefixMode.CHUNKED in modes else None)
                planner.schedule_confirmation(chunked)
                continue
            if not planner.queue:
                break
            case = planner.queue.popleft()
            if not planner.fits(case):
                results[case.key] = PassResult(case=case, status="skipped_budget")
            else:
                print(f"diagnostic: {case.key} (remaining {planner.deadline - clock():.0f}s)",
                      file=sys.stderr, flush=True)
                results[case.key] = run_pass(engine, case, positions, tail_tokens,
                                             deadline=planner.deadline, clock=clock)
                planner.record(results[case.key])
            report["passes"][case.key] = results[case.key].summary()
            report["modes_summary"] = {mode.value: summarize_mode(mode, results, planner.all_variations)
                                       for mode in modes}
            # Provisional after every case: a killed run still leaves a verdict.
            report["verdict"] = {**verdict(report["modes_summary"], planner.all_variations), "final": False}
            persist()
        report["modes_summary"] = {mode.value: summarize_mode(mode, results, planner.all_variations)
                                   for mode in modes}
        report["verdict"] = {**verdict(report["modes_summary"], planner.all_variations), "final": True}
        report["status"] = "finished"
        return report
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        engine.restore_adapter(snapshot)
        engine.cache_offloading = original_offloading
        restored = engine.policy_identity()
        report["restored_policy_identity"] = {**asdict(restored), "digest": restored.digest}
        report["initial_policy_identity_restored"] = restored == initial_policy
        if restored != initial_policy:
            report.update(status="error", error="PolicyIdentity differs after adapter restoration")
        persist()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-report", type=Path,
                        help="recompute the verdict from a saved report (no GPU, no model) and print it")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--positions", type=int, nargs="+", default=list(DEFAULT_POSITIONS))
    parser.add_argument("--modes", default="chunked,tokenwise",
                        help="comma list of chunked,tokenwise (order is fixed: chunked sweep first)")
    parser.add_argument("--tail-tokens", type=int, default=DEFAULT_TAIL_TOKENS,
                        help="tokenwise steps before the first probe in chunked mode")
    parser.add_argument("--budget-seconds", type=float, default=DEFAULT_BUDGET_SECONDS,
                        help="wall budget for the passes (model load excluded); unrun cases are recorded")
    parser.add_argument("--only", default="",
                        help="comma list of variation names to run (default: all)")
    args = parser.parse_args(argv)
    if args.from_report is not None:
        try:
            print(json.dumps(verdict_from_report(json.loads(args.from_report.read_text())), indent=2))
        except (OSError, ValueError, KeyError) as error:
            print(f"diagnostic: error: cannot read a verdict from {args.from_report}: "
                  f"{type(error).__name__}: {error}", file=sys.stderr)
            return 1
        return 0
    if args.parent is None or args.output is None:
        parser.error("--parent and --output are required unless --from-report is given")
    try:
        modes = [PrefixMode(value.strip()) for value in args.modes.split(",") if value.strip()]
    except ValueError as error:
        parser.error(f"--modes: {error}")
    selected = [name.strip() for name in args.only.split(",") if name.strip()]
    unknown = sorted(set(selected) - {variation.name for variation in VARIATIONS})
    if unknown:
        parser.error("unknown variation(s): " + ", ".join(unknown) +
                     "; choose from " + ", ".join(variation.name for variation in VARIATIONS))
    variations = [variation for variation in VARIATIONS if not selected or variation.name in selected]

    from lilith_adapter_mechanics import PARENT_CONFIG_SHA256, PARENT_SHA256, TARGET
    from lilith_training_cache_control import PINNED_TOKENIZER_SHA256
    from lilith_training_engine import GemmaTrainingEngine

    engine = GemmaTrainingEngine.load(
        args.parent, expected_parent_sha256=PARENT_SHA256, expected_config_sha256=PARENT_CONFIG_SHA256,
        expected_tokenizer_sha256=PINNED_TOKENIZER_SHA256, target_module=TARGET)
    try:
        report = run_diagnostic(engine, args.output, positions=args.positions, modes=modes,
                                variations=variations, tail_tokens=args.tail_tokens,
                                budget_seconds=args.budget_seconds)
    except Exception as error:
        print(f"diagnostic: error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "verdict": report["verdict"]}, sort_keys=True))
    return 0 if report["status"] == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
