#!/usr/bin/env python3
"""Explicit, recorded CUDA numerics configuration for the Lilith training engine.

The training engine (``lilith_training_engine.py``) is qualified by its sha256 and
its arithmetic is never edited without a new qualification. The knobs here are
process-global PyTorch settings applied *around* the engine: SDPA backend
selection, TF32, bf16 reduced-precision GEMM reductions, and deterministic
algorithms. A bare ``NumericsConfig()`` is ``default`` (leave PyTorch untouched),
which is the behaviour the first C6 smoke ran with; every run records the
requested and the observed settings in its manifest (``NumericsConfig.manifest``).

The *recorded default* (``platform_default()``, what every caller resolves to when
no ``--numerics-*`` flag or ``LILITH_NUMERICS_*`` variable is given) is
EFFICIENT SDPA with MATH fallback on CUDA (not ROCm), and ``default`` (untouched)
everywhere else. C6 round 2 (2026-09-28) found the default CUDA SDPA dispatch
(cuDNN attention on H100, torch 2.13.0+cu130) run-to-run nondeterministic: a
resident-cache control run repeated twice differed as much as resident vs.
offloaded did. C6 round 3 confirmed EFFICIENT+MATH restores bit-exact
resident-vs-offload equivalence (logits 0.0/0.0/0.0 at 1023/1024/1025, LoRA grad
diffs 1.4e-8/1.5e-7, tolerance 1e-4). ROCm and CPU are unmeasured on this finding
and keep PyTorch's own default; see ``platform_default`` and ``cuda_default``.

Choosing a value other than the recorded default is an operator decision informed
by ``lilith_cache_path_diagnostic.py``; see its verdict for which setting restored
resident-versus-offloaded equivalence on the measured stack.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import enum
import os
from typing import Any, Iterator, Mapping

SCHEMA = "lilith/numerics@1"
CUBLAS_DETERMINISTIC_WORKSPACE = ":4096:8"


class SdpaBackend(str, enum.Enum):
    """Scaled-dot-product-attention backend selection (``torch.nn.attention.SDPBackend``)."""

    DEFAULT = "default"           # PyTorch picks per call (the qualified behaviour)
    MATH = "math"
    EFFICIENT = "efficient"       # EFFICIENT_ATTENTION
    FLASH = "flash"               # FLASH_ATTENTION (head_dim <= 256; Gemma global layers use 512)
    CUDNN = "cudnn"               # CUDNN_ATTENTION

    def torch_backend(self) -> Any:
        from torch.nn.attention import SDPBackend

        return {
            SdpaBackend.MATH: SDPBackend.MATH,
            SdpaBackend.EFFICIENT: SDPBackend.EFFICIENT_ATTENTION,
            SdpaBackend.FLASH: SDPBackend.FLASH_ATTENTION,
            SdpaBackend.CUDNN: SDPBackend.CUDNN_ATTENTION,
        }[self]


class Toggle(str, enum.Enum):
    """``default`` leaves the PyTorch global untouched; ``on``/``off`` force it."""

    DEFAULT = "default"
    ON = "on"
    OFF = "off"


class NumericsError(ValueError):
    """A numerics configuration that cannot be applied as requested.

    Callers can (a) fall back to ``NumericsConfig()`` (the qualified default),
    (b) drop the offending knob, or (c) fail the run; the diagnostic records the
    error and continues with the next variation.
    """


@dataclass(frozen=True)
class NumericsConfig:
    sdpa_backend: SdpaBackend = SdpaBackend.DEFAULT
    # When the forced backend has no kernel for a call (flash on head_dim 512),
    # allow MATH as the second choice instead of raising.
    sdpa_math_fallback: bool = False
    tf32: Toggle = Toggle.DEFAULT
    bf16_reduced_precision_reduction: Toggle = Toggle.DEFAULT
    deterministic: bool = False
    # Only meaningful with deterministic=True: record nondeterministic ops as
    # warnings instead of raising at the first one.
    deterministic_warn_only: bool = False

    @property
    def is_default(self) -> bool:
        return self == NumericsConfig()

    def label(self) -> str:
        if self.is_default:
            return "default"
        parts = []
        if self.sdpa_backend is not SdpaBackend.DEFAULT:
            parts.append("sdpa=" + self.sdpa_backend.value + ("+math" if self.sdpa_math_fallback else ""))
        if self.tf32 is not Toggle.DEFAULT:
            parts.append("tf32=" + self.tf32.value)
        if self.bf16_reduced_precision_reduction is not Toggle.DEFAULT:
            parts.append("bf16_rr=" + self.bf16_reduced_precision_reduction.value)
        if self.deterministic:
            parts.append("deterministic" + ("(warn_only)" if self.deterministic_warn_only else ""))
        return ",".join(parts)

    def validate(self) -> None:
        if self.sdpa_math_fallback and self.sdpa_backend in (SdpaBackend.DEFAULT, SdpaBackend.MATH):
            raise NumericsError("sdpa_math_fallback needs a forced non-math SDPA backend")
        if self.deterministic_warn_only and not self.deterministic:
            raise NumericsError("deterministic_warn_only needs deterministic=True")

    def requested(self) -> dict[str, Any]:
        return {
            "sdpa_backend": self.sdpa_backend.value,
            "sdpa_math_fallback": self.sdpa_math_fallback,
            "tf32": self.tf32.value,
            "bf16_reduced_precision_reduction": self.bf16_reduced_precision_reduction.value,
            "deterministic": self.deterministic,
            "deterministic_warn_only": self.deterministic_warn_only,
        }

    def manifest(self) -> dict[str, Any]:
        """Requested knobs plus the observed process state, for run manifests."""
        return {"schema": SCHEMA, "label": self.label(), "is_default": self.is_default,
                "requested": self.requested(), "observed": observed_state()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NumericsConfig":
        try:
            config = cls(
                sdpa_backend=SdpaBackend(value.get("sdpa_backend", "default")),
                sdpa_math_fallback=bool(value.get("sdpa_math_fallback", False)),
                tf32=Toggle(value.get("tf32", "default")),
                bf16_reduced_precision_reduction=Toggle(
                    value.get("bf16_reduced_precision_reduction", "default")),
                deterministic=bool(value.get("deterministic", False)),
                deterministic_warn_only=bool(value.get("deterministic_warn_only", False)),
            )
        except ValueError as error:
            raise NumericsError(f"invalid numerics configuration: {error}") from error
        config.validate()
        return config

    @classmethod
    def from_args(cls, args: argparse.Namespace, *, cuda: bool | None = None) -> "NumericsConfig":
        """Explicit ``--numerics-*`` flags always win; with none given, ``platform_default()``.

        ``cuda`` is forwarded to ``platform_default`` unchanged: leave it ``None`` when
        this process runs on the host being measured (it detects locally), pass it
        explicitly when authoring a config for a *different* host.
        """
        explicit = any(value not in (None, False) for value in (
            args.numerics_sdpa_backend, args.numerics_sdpa_math_fallback, args.numerics_tf32,
            args.numerics_bf16_reduced_precision_reduction, args.numerics_deterministic))
        if not explicit:
            return platform_default(cuda=cuda)
        return cls.from_mapping({
            "sdpa_backend": args.numerics_sdpa_backend or SdpaBackend.DEFAULT.value,
            "sdpa_math_fallback": bool(args.numerics_sdpa_math_fallback),
            "tf32": args.numerics_tf32 or Toggle.DEFAULT.value,
            "bf16_reduced_precision_reduction": args.numerics_bf16_reduced_precision_reduction or Toggle.DEFAULT.value,
            "deterministic": bool(args.numerics_deterministic),
            "deterministic_warn_only": False,
        })

    @contextlib.contextmanager
    def applied(self) -> Iterator["NumericsConfig"]:
        """Apply the knobs for the duration of the block and restore the previous state."""
        import torch

        self.validate()
        with contextlib.ExitStack() as stack:
            if self.tf32 is not Toggle.DEFAULT:
                enabled = self.tf32 is Toggle.ON
                previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)

                def restore_tf32(previous=previous):
                    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous

                stack.callback(restore_tf32)
                torch.backends.cuda.matmul.allow_tf32 = enabled
                torch.backends.cudnn.allow_tf32 = enabled
            if self.bf16_reduced_precision_reduction is not Toggle.DEFAULT:
                matmul = torch.backends.cuda.matmul
                previous_rr = matmul.allow_bf16_reduced_precision_reduction
                stack.callback(setattr, matmul, "allow_bf16_reduced_precision_reduction", previous_rr)
                matmul.allow_bf16_reduced_precision_reduction = (
                    self.bf16_reduced_precision_reduction is Toggle.ON)
            if self.deterministic:
                if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
                    raise NumericsError(
                        "deterministic cuBLAS needs CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) "
                        "set before CUDA initialises; export it in the launching shell")
                previous_mode = torch.are_deterministic_algorithms_enabled()
                previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
                previous_cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)

                def restore_deterministic():
                    torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn)
                    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous_cudnn

                stack.callback(restore_deterministic)
                torch.use_deterministic_algorithms(True, warn_only=self.deterministic_warn_only)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            if self.sdpa_backend is not SdpaBackend.DEFAULT:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                backends = [self.sdpa_backend.torch_backend()]
                if self.sdpa_math_fallback:
                    backends.append(SDPBackend.MATH)
                stack.enter_context(sdpa_kernel(backends))
            yield self


def cuda_default() -> "NumericsConfig":
    """The recorded default on CUDA (not ROCm): EFFICIENT SDPA with MATH fallback.

    See the module docstring for the C6 round 2/round 3 finding this encodes.
    """
    return NumericsConfig(sdpa_backend=SdpaBackend.EFFICIENT, sdpa_math_fallback=True)


def detect_cuda() -> bool:
    """True for a real (non-ROCm) CUDA build with a device visible to this process.

    Never raises: an absent torch, or a CUDA-less build, both mean False. Correct
    only for the process's own host; a caller authoring a config for a *different*
    host must not rely on this (see ``platform_default``).
    """
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available()) and torch.version.hip is None


def platform_default(*, cuda: bool | None = None) -> "NumericsConfig":
    """The recorded default for this platform: ``cuda_default()`` on CUDA, else ``NumericsConfig()``.

    ``cuda=None`` detects the local process's own CUDA visibility via ``detect_cuda()``
    (correct for code that runs on the GPU host itself, e.g. the smoke harness or a
    runtime worker). A caller preparing a *pinned* config for a different host (e.g.
    an operator machine authoring a frozen worker config for a remote H100 pod) must
    pass ``cuda`` explicitly instead: local detection would silently record the
    wrong platform's default.
    """
    is_cuda = detect_cuda() if cuda is None else cuda
    return cuda_default() if is_cuda else NumericsConfig()


def observed_state() -> dict[str, Any]:
    """The process-global numerics state as PyTorch reports it right now."""
    state: dict[str, Any] = {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
    try:
        import torch
    except ImportError:
        state["torch"] = None
        return state
    cuda = torch.backends.cuda
    state.update({
        "torch": torch.__version__,
        "matmul_allow_tf32": cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_bf16_reduced_precision_reduction": cuda.matmul.allow_bf16_reduced_precision_reduction,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "sdpa_enabled": {
            "flash": cuda.flash_sdp_enabled(),
            "efficient": cuda.mem_efficient_sdp_enabled(),
            "math": cuda.math_sdp_enabled(),
            "cudnn": cuda.cudnn_sdp_enabled(),
        },
    })
    fill = getattr(getattr(torch.utils, "deterministic", None), "fill_uninitialized_memory", None)
    if fill is not None:
        state["deterministic_fill_uninitialized_memory"] = bool(fill)
    return state


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """CLI flags; with none given, ``NumericsConfig.from_args`` resolves ``platform_default()``
    (EFFICIENT SDPA + MATH fallback on CUDA, untouched on ROCm/CPU)."""
    group = parser.add_argument_group(
        "numerics (recorded in the run manifest; unset = platform_default(), see lilith_numerics.py)")
    group.add_argument("--numerics-sdpa-backend", choices=[item.value for item in SdpaBackend], default=None,
                       help="unset = platform_default()'s choice (efficient on CUDA, default elsewhere)")
    group.add_argument("--numerics-sdpa-math-fallback", action="store_true", default=None,
                       help="allow MATH where the forced SDPA backend has no kernel")
    group.add_argument("--numerics-tf32", choices=[item.value for item in Toggle], default=None)
    group.add_argument("--numerics-bf16-reduced-precision-reduction", choices=[item.value for item in Toggle],
                       default=None)
    group.add_argument("--numerics-deterministic", action="store_true", default=None,
                       help="torch.use_deterministic_algorithms(True); needs CUBLAS_WORKSPACE_CONFIG=:4096:8")
