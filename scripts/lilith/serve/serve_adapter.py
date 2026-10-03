#!/usr/bin/env python3
"""Decide what the Lilith serve pod hands to vLLM: the frozen parent or the certified adapter.

Anima WI 01a0e270-314e (Lilith C2, Stage 0); ADR 116 Decision 7; C4b certification
(`scripts/lilith/exam/certify.py`).

    serve_adapter.py select --registry /workspace/lilith/adapters/active [--mode auto|parent]

Registry layout (an `AdapterRegistry` root, written by C4b `accept`):

    <registry>/active.json                               the one active manifest
    <registry>/acceptances/<candidate sha256>.json       the operator acceptance record
    <registry>/weights/<adapter_sha256>/adapter_model.safetensors
    <registry>/weights/<adapter_sha256>/adapter_config.json

Modes:

* ``auto`` (default): no ``active.json`` -> the frozen parent. An ``active.json`` is
  served only when `AdapterRegistry.load_active` accepts it (operator acceptance embedded
  and on record, adapter bytes hash to the pinned ``adapter_sha256``). A present but
  invalid ``active.json`` is a refusal, never a silent fallback to the parent
  (certify.py: serving "never falls back to a candidate or the parent").
* ``parent``: the frozen parent, even when an adapter is active. Recorded as a forced
  choice in the selection JSON so the session record says why.

Prints one ``lilith/serve-selection@1`` JSON object on stdout. Exit 0 on a selection,
2 on a refusal (the diagnostic goes to stderr as ``[code] message`` + ``help:``).
"""
from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "lilith/serve-selection@1"
# vLLM only accepts these values for --max-lora-rank.
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)
_LILITH = Path(__file__).resolve().parents[1]  # scripts/lilith (payload-relative)


class Mode(enum.StrEnum):
    AUTO = "auto"
    PARENT = "parent"


class SelectionError(Exception):
    code = "lilith::serve::adapter"
    help = ""

    def __init__(self, message: str, *, code: str | None = None, help: str | None = None):
        super().__init__(message)
        if code:
            self.code = code
        if help:
            self.help = help

    def render(self) -> str:
        text = f"[{self.code}] {self}"
        return text + (f"\n  help: {self.help}" if self.help else "")


@dataclasses.dataclass(frozen=True)
class Selection:
    mode: str  # "parent" | "adapter"
    reason: str
    adapter_dir: str | None = None
    adapter_sha256: str | None = None
    manifest_sha256: str | None = None
    run_id: str | None = None
    checkpoint_step: int | None = None
    accepted_by: str | None = None
    lora_rank: int | None = None
    max_lora_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA} | dataclasses.asdict(self)


def vllm_max_lora_rank(rank: int) -> int:
    for allowed in VLLM_LORA_RANKS:
        if rank <= allowed:
            return allowed
    raise SelectionError(f"LoRA rank {rank} exceeds vLLM's largest --max-lora-rank {VLLM_LORA_RANKS[-1]}",
                         code="lilith::serve::adapter_rank",
                         help="serve the parent (--mode parent) or activate an adapter with a supported rank")


def select(registry: Path, mode: Mode) -> Selection:
    active = registry / "active.json"
    if mode is Mode.PARENT:
        why = "forced by LILITH_SERVE_ADAPTER=parent"
        if active.exists():
            why += f" (an active adapter exists at {active} and is NOT served)"
        return Selection(mode="parent", reason=why)
    if not active.exists():
        return Selection(mode="parent", reason=f"no {active}: serving the frozen parent (Stage 0 default)")

    if str(_LILITH) not in sys.path:
        sys.path.insert(0, str(_LILITH))
    try:
        from exam.certify import AdapterRegistry  # noqa: PLC0415  (payload-relative import)
        from exam.errors import CertificationError, NoActiveAdapterError  # noqa: PLC0415
    except ImportError as error:
        raise SelectionError(f"cannot import the C4b certification module: {error}",
                             code="lilith::serve::adapter_payload",
                             help="the payload lacks scripts/lilith/exam or scripts/lilith/dojo; rebuild it with "
                                  "make_payload.sh from origin/main, or serve the parent with "
                                  "LILITH_SERVE_ADAPTER=parent") from error

    reg = AdapterRegistry(registry)
    try:
        # First pass without bytes: learn which adapter the manifest pins.
        manifest = reg.load_active()
        weights = registry / "weights" / manifest.adapter_sha256
        adapter_file = weights / "adapter_model.safetensors"
        if not adapter_file.is_file():
            raise NoActiveAdapterError(f"{adapter_file} is missing")
        manifest = reg.load_active(adapter_file)
    except (CertificationError, NoActiveAdapterError) as error:
        raise SelectionError(f"active.json is present but not servable: {error}",
                             code="lilith::serve::adapter_uncertified",
                             help=f"options: (1) restore the adapter bytes under {registry}/weights/<adapter_sha256>/; "
                                  f"(2) move {active} aside to serve the frozen parent; (3) start with "
                                  "LILITH_SERVE_ADAPTER=parent for this session") from error

    config_path = weights / "adapter_config.json"
    try:
        rank = int(json.loads(config_path.read_text())["r"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SelectionError(f"{config_path} is unreadable or has no integer 'r': {error}",
                             code="lilith::serve::adapter_config",
                             help="copy the PEFT adapter_config.json next to the adapter weights") from error
    acceptance = manifest.acceptance
    return Selection(mode="adapter", reason="certified active adapter (C4b operator acceptance on record)",
                     adapter_dir=str(weights), adapter_sha256=manifest.adapter_sha256,
                     manifest_sha256=manifest.manifest_sha256, run_id=manifest.run_id,
                     checkpoint_step=manifest.checkpoint_step,
                     accepted_by=None if acceptance is None else acceptance.operator,
                     lora_rank=rank, max_lora_rank=vllm_max_lora_rank(rank))


CANDIDATE_SCHEMA = "lilith/serve-candidate@1"


def select_candidate(selection_file: Path, root: Path) -> Selection:
    """Selection for an UNCERTIFIED training-program stage adapter (on-policy rollouts only).

    The file is written by ``scripts/lilith/program/serve_candidate.sh``; it must name an adapter
    directory under ``<root>/program/``, pin ``adapter_model.safetensors`` by sha256 and carry a
    ``max_lora_rank`` that vLLM accepts for the adapter's rank. Nothing here activates anything.
    """
    import hashlib  # noqa: PLC0415

    try:
        data = json.loads(Path(selection_file).read_text())
    except (OSError, ValueError) as error:
        raise SelectionError(f"candidate selection {selection_file} is unreadable: {error}",
                             code="lilith::serve::candidate_file", help="write it with serve_candidate.sh start") from error
    if not isinstance(data, dict) or data.get("schema") != CANDIDATE_SCHEMA or data.get("served_model_name") != "lilith":
        raise SelectionError(f"{selection_file} is not a {CANDIDATE_SCHEMA} for served model 'lilith'",
                             code="lilith::serve::candidate_schema", help="write it with serve_candidate.sh start")
    adapter_dir = Path(str(data.get("adapter_dir", ""))).resolve()
    allowed = (Path(root) / "program").resolve()
    if allowed not in adapter_dir.parents:
        raise SelectionError(f"candidate adapter {adapter_dir} is not under {allowed}",
                             code="lilith::serve::candidate_path",
                             help="only training-program stage adapters are served as candidates; accepted adapters go through C4b")
    weights = adapter_dir / "adapter_model.safetensors"
    pin = data.get("adapter_model_sha256")
    if not weights.is_file() or not isinstance(pin, str):
        raise SelectionError(f"{weights} is missing or the selection pins no sha256",
                             code="lilith::serve::candidate_weights", help="pin the stage adapter's adapter_model.safetensors")
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != pin:
        raise SelectionError(f"{weights.name} sha256 {digest.hexdigest()} != pinned {pin}",
                             code="lilith::serve::candidate_hash", help="never serve an adapter that drifted from the trainer's receipt")
    try:
        rank = int(json.loads((adapter_dir / "adapter_config.json").read_text())["r"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SelectionError(f"{adapter_dir}/adapter_config.json is unreadable or has no integer 'r': {error}",
                             code="lilith::serve::adapter_config", help="the stage adapter directory must be a PEFT adapter") from error
    max_rank = vllm_max_lora_rank(rank)
    if data.get("max_lora_rank") != max_rank:
        raise SelectionError(f"selection max_lora_rank {data.get('max_lora_rank')!r} != {max_rank} required for rank {rank}",
                             code="lilith::serve::adapter_rank", help="serve_candidate.sh derives it from adapter_config.json")
    return Selection(mode="adapter",
                     reason="UNCERTIFIED training-program candidate for on-policy rollouts (no C4b acceptance; never active)",
                     adapter_dir=str(adapter_dir), adapter_sha256=pin, run_id=data.get("run_id"), lora_rank=rank, max_lora_rank=max_rank)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sel = commands.add_parser("select")
    sel.add_argument("--registry", type=Path, required=True)
    sel.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.AUTO.value)
    cand = commands.add_parser("candidate")
    cand.add_argument("--selection", type=Path, required=True)
    cand.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "candidate":
            selection = select_candidate(args.selection, args.root)
        else:
            selection = select(args.registry, Mode(args.mode))
    except SelectionError as error:
        print(error.render(), file=sys.stderr)
        return 2
    print(json.dumps(selection.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
