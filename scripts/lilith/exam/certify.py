"""Lilith candidate / active manifests (workflow §1 "Candidate vs active", ADR 102 §5).

Training writes a **candidate** manifest: the pinned parent (weights, config, tokenizer,
chat template), the adapter hash, the `ToolContractSnapshot` hash, the data manifest
hashes (the C5 trajectories v3 corpus among them), the training-config hash and the main
SHA of the code, the reward policy (`cos-v1`), the frozen exam manifest hash and the
numerics block.

A candidate becomes **active** only through `AdapterRegistry.accept`, which requires:

- a gate report for the same exam manifest with every family PASS and no blocker;
- an explicit `OperatorAcceptance` naming the candidate manifest hash and the gate report
  hash, signed by a human operator identity (Lilith and the supervising agents cannot
  accept: the supervisor proposes, the operator accepts).

Serving calls `AdapterRegistry.load_active`, which reads `active.json` only, checks the
embedded acceptance and the adapter bytes, and never falls back to a candidate or the parent.
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from dojo.reward import COS_V1
from dojo.snapshot import canonical_json, sha256_hex
from exam.errors import CertificationError, NoActiveAdapterError
from exam.manifest import is_sha256

MODEL_MANIFEST_SCHEMA = "lilith/model-manifest@1"
GATE_REPORT_SCHEMA = "lilith/gate-report@1"
ACCEPTANCE_SCHEMA = "lilith/operator-acceptance@1"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")

# Identities that may never accept a candidate (ADR 116: Lilith never activates her own
# weights; the supervisor and the judges propose, the operator accepts).
AGENT_IDENTITIES = frozenset({"lilith", "supervisor", "fable", "opus", "astra", "claude", "codex", "tecton"})


@dataclasses.dataclass(frozen=True)
class ParentPins:
    """google/gemma-4-12B-it @ 5926caa4 (training/lilith/adapter-readiness-20260921/lineage.json)."""

    repo: str = "google/gemma-4-12B-it"
    revision: str = "5926caa4ec0cac5cbfadaf4077420520de1d5205"
    weights_sha256: str = "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d"
    config_sha256: str = "478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9"
    tokenizer_sha256: str = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
    template_sha256: str = "36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


GEMMA4_12B_PARENT = ParentPins()


@dataclasses.dataclass(frozen=True)
class DataManifestPin:
    name: str
    sha256: str
    role: str = "train"  # train | dev | siblings | teacher

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# `dataset_sha256` of ~/lilith-private/c5/cos-trajectories-v3.manifest.json (92 records, 78/14).
C5_TRAJECTORIES_V3 = DataManifestPin("c5-cos-trajectories-v3",
                                     "a89e98ea42eaaaa47f2eba43b2ad2231c7420a8040ec0cdb166946a57c31f98d")


@dataclasses.dataclass(frozen=True)
class Numerics:
    """The C6 round-3 numerics that made resident-vs-offload bit-exact on H100."""

    dtype: str = "bf16"
    attention: str = "sdpa"
    sdpa_backend: str = "efficient"
    sdpa_math_fallback: bool = True
    matmul_tf32: bool = False
    deterministic_algorithms: bool = False
    extra: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self) | {"extra": dict(sorted(self.extra.items()))}


class State(enum.StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"


@dataclasses.dataclass(frozen=True)
class OperatorAcceptance:
    operator: str
    accepted_at: str
    candidate_manifest_sha256: str
    gate_report_sha256: str
    reason: str

    def to_dict(self) -> dict:
        return {"schema": ACCEPTANCE_SCHEMA} | dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperatorAcceptance":
        return cls(data["operator"], data["accepted_at"], data["candidate_manifest_sha256"],
                   data["gate_report_sha256"], data["reason"])


@dataclasses.dataclass(frozen=True)
class ModelManifest:
    run_id: str
    checkpoint_step: int
    adapter_sha256: str
    tool_contract_sha256: str
    data_manifests: Sequence[DataManifestPin]
    training_config_sha256: str
    code_main_sha: str
    exam_manifest_sha256: str
    created_at: str
    parent: ParentPins = GEMMA4_12B_PARENT
    reward_policy: str = COS_V1.name
    reward_policy_sha256: str = COS_V1.policy_sha256
    numerics: Numerics = Numerics()
    state: State = State.CANDIDATE
    # set on activation only; they are part of the active manifest's hash
    candidate_manifest_sha256: str | None = None
    gate_report_sha256: str | None = None
    acceptance: OperatorAcceptance | None = None

    def problems(self) -> list[str]:
        out: list[str] = []
        for name in ("adapter_sha256", "tool_contract_sha256", "training_config_sha256", "exam_manifest_sha256",
                     "reward_policy_sha256"):
            if not is_sha256(getattr(self, name)):
                out.append(f"{name} is not a sha256")
        for name, value in self.parent.to_dict().items():
            if name.endswith("_sha256") and not is_sha256(value):
                out.append(f"parent.{name} is not a sha256")
        if not _GIT_SHA.match(self.code_main_sha or ""):
            out.append("code_main_sha is not a 40-hex git commit")
        if not self.data_manifests:
            out.append("no data manifests")
        for pin in self.data_manifests:
            if not is_sha256(pin.sha256):
                out.append(f"data manifest {pin.name} is not a sha256")
        if self.reward_policy != COS_V1.name or self.reward_policy_sha256 != COS_V1.policy_sha256:
            out.append(f"reward policy {self.reward_policy}/{self.reward_policy_sha256[:12]} is not the pinned cos-v1")
        if self.state is State.ACTIVE and (self.acceptance is None or self.candidate_manifest_sha256 is None
                                           or self.gate_report_sha256 is None):
            out.append("an active manifest must embed its candidate hash, gate report hash and operator acceptance")
        if self.state is State.CANDIDATE and (self.acceptance is not None or self.gate_report_sha256 is not None):
            out.append("a candidate manifest carries no acceptance")
        return out

    def validate(self) -> None:
        problems = self.problems()
        if problems:
            raise CertificationError("model manifest invalid: " + "; ".join(problems))

    def body(self) -> dict:
        return {
            "schema": MODEL_MANIFEST_SCHEMA, "state": self.state.value, "run_id": self.run_id,
            "checkpoint_step": self.checkpoint_step, "created_at": self.created_at, "parent": self.parent.to_dict(),
            "adapter_sha256": self.adapter_sha256, "tool_contract_sha256": self.tool_contract_sha256,
            "data_manifests": [p.to_dict() for p in sorted(self.data_manifests, key=lambda p: p.name)],
            "training_config_sha256": self.training_config_sha256, "code_main_sha": self.code_main_sha,
            "reward_policy": {"name": self.reward_policy, "sha256": self.reward_policy_sha256},
            "exam_manifest_sha256": self.exam_manifest_sha256, "numerics": self.numerics.to_dict(),
            "candidate_manifest_sha256": self.candidate_manifest_sha256,
            "gate_report_sha256": self.gate_report_sha256,
            "acceptance": None if self.acceptance is None else self.acceptance.to_dict(),
        }

    @property
    def manifest_sha256(self) -> str:
        return sha256_hex(canonical_json(self.body()))

    def to_dict(self) -> dict:
        return self.body() | {"manifest_sha256": self.manifest_sha256}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelManifest":
        if data.get("schema") != MODEL_MANIFEST_SCHEMA:
            raise CertificationError(f"schema {data.get('schema')!r} is not {MODEL_MANIFEST_SCHEMA}")
        n = data["numerics"]
        m = cls(
            run_id=data["run_id"], checkpoint_step=int(data["checkpoint_step"]), adapter_sha256=data["adapter_sha256"],
            tool_contract_sha256=data["tool_contract_sha256"],
            data_manifests=tuple(DataManifestPin(**p) for p in data["data_manifests"]),
            training_config_sha256=data["training_config_sha256"], code_main_sha=data["code_main_sha"],
            exam_manifest_sha256=data["exam_manifest_sha256"], created_at=data["created_at"],
            parent=ParentPins(**data["parent"]), reward_policy=data["reward_policy"]["name"],
            reward_policy_sha256=data["reward_policy"]["sha256"],
            numerics=Numerics(n["dtype"], n["attention"], n["sdpa_backend"], n["sdpa_math_fallback"], n["matmul_tf32"],
                              n["deterministic_algorithms"], n.get("extra", {})),
            state=State(data["state"]), candidate_manifest_sha256=data.get("candidate_manifest_sha256"),
            gate_report_sha256=data.get("gate_report_sha256"),
            acceptance=None if data.get("acceptance") is None else OperatorAcceptance.from_dict(data["acceptance"]),
        )
        if data.get("manifest_sha256") != m.manifest_sha256:
            raise CertificationError(f"stored manifest_sha256 {data.get('manifest_sha256')} != recomputed {m.manifest_sha256}")
        return m


def gate_report(exam_manifest_sha256: str, matrices: Sequence[Mapping[str, Any]],
                panel: Mapping[str, Any] | None = None, extra_blockers: Sequence[str] = ()) -> dict:
    """Bundle the sealed family matrices (`FamilyMatrix.to_dict()`) and the panel report into
    the document an operator accepts. Release-ready only when every matrix is, the panel (if
    any) has no failing gated dimension, and nothing else blocks."""
    blockers: list[str] = list(extra_blockers)
    for matrix in matrices:
        for family, row in matrix["families"].items():
            if row["verdict"] != "pass":
                reasons = row["blockers"] or row["missing"]
                blockers.append(f"{matrix['arm']}/{family}: {row['verdict']} ({'; '.join(reasons)})")
    if panel is not None:
        blockers.extend(f"panel: {b}" for b in panel.get("blockers", []))
    body = {"schema": GATE_REPORT_SCHEMA, "exam_manifest_sha256": exam_manifest_sha256,
            "matrices": list(matrices), "panel": panel, "blockers": blockers,
            "release_ready": bool(matrices) and not blockers}
    return body | {"gate_report_sha256": sha256_hex(canonical_json(body))}


def verify_on_main(sha: str, repo: Path, remote_ref: str = "origin/main") -> bool:
    """`git merge-base --is-ancestor <sha> origin/main` (the deploy-from-main rule). The
    caller fetches first; this never fetches."""
    result = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", sha, remote_ref],
                            capture_output=True, text=True, check=False)
    return result.returncode == 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


class AdapterRegistry:
    """On-disk layout under `root`:

        candidates/<candidate sha256>.json   write-once candidate manifests
        acceptances/<candidate sha256>.json  write-once operator acceptance records
        gate-reports/<report sha256>.json    the accepted gate reports
        active.json                          the one active manifest (serving reads only this)
        history/<active sha256>.json         previously active manifests
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    def write_candidate(self, manifest: ModelManifest) -> Path:
        if manifest.state is not State.CANDIDATE:
            raise CertificationError("only candidate manifests are written by training")
        manifest.validate()
        path = self.root / "candidates" / f"{manifest.manifest_sha256}.json"
        if path.exists():
            raise CertificationError(f"{path} exists; candidate manifests are write-once",
                                     help="a new checkpoint or adapter is a new candidate with its own hash")
        _atomic_write(path, manifest.to_dict())
        return path

    def candidate(self, sha: str) -> ModelManifest:
        path = self.root / "candidates" / f"{sha}.json"
        if not path.exists():
            raise CertificationError(f"no candidate {sha} under {self.root}")
        return ModelManifest.from_dict(json.loads(path.read_text()))

    def accept(self, candidate_sha: str, report: Mapping[str, Any], acceptance: OperatorAcceptance) -> ModelManifest:
        candidate = self.candidate(candidate_sha)
        problems: list[str] = []
        operator = acceptance.operator.strip().lower()
        if not operator or operator in AGENT_IDENTITIES:
            problems.append(f"acceptance by {acceptance.operator!r}: only the human operator accepts a candidate")
        if not acceptance.reason.strip():
            problems.append("acceptance has no reason")
        if acceptance.candidate_manifest_sha256 != candidate.manifest_sha256:
            problems.append("acceptance names a different candidate manifest")
        recomputed = sha256_hex(canonical_json({k: v for k, v in report.items() if k != "gate_report_sha256"}))
        if report.get("gate_report_sha256") != recomputed:
            problems.append("gate report hash does not match its content")
        if acceptance.gate_report_sha256 != recomputed:
            problems.append("acceptance names a different gate report")
        if report.get("exam_manifest_sha256") != candidate.exam_manifest_sha256:
            problems.append("gate report was scored against a different exam manifest")
        if not report.get("release_ready") or report.get("blockers"):
            problems.append(f"gates not passed: {report.get('blockers') or 'release_ready is false'}")
        if problems:
            raise CertificationError("activation refused: " + "; ".join(problems))
        active = dataclasses.replace(candidate, state=State.ACTIVE, candidate_manifest_sha256=candidate.manifest_sha256,
                                     gate_report_sha256=recomputed, acceptance=acceptance)
        active.validate()
        acceptance_path = self.root / "acceptances" / f"{candidate.manifest_sha256}.json"
        if acceptance_path.exists():
            raise CertificationError(f"{acceptance_path} exists; this candidate was already accepted")
        _atomic_write(self.root / "gate-reports" / f"{recomputed}.json", dict(report))
        _atomic_write(acceptance_path, acceptance.to_dict())
        current = self.root / "active.json"
        if current.exists():
            previous = json.loads(current.read_text())
            _atomic_write(self.root / "history" / f"{previous.get('manifest_sha256', 'unknown')}.json", previous)
        _atomic_write(current, active.to_dict())
        return active

    def load_active(self, adapter_file: Path | None = None) -> ModelManifest:
        """What serving calls. No active manifest is an error, never a silent fallback."""
        path = self.root / "active.json"
        if not path.exists():
            raise NoActiveAdapterError(f"no active adapter under {self.root}")
        manifest = ModelManifest.from_dict(json.loads(path.read_text()))
        if manifest.state is not State.ACTIVE:
            raise NoActiveAdapterError(f"{path} is a {manifest.state} manifest")
        manifest.validate()
        acceptance = self.root / "acceptances" / f"{manifest.candidate_manifest_sha256}.json"
        if not acceptance.exists() or OperatorAcceptance.from_dict(json.loads(acceptance.read_text())) != manifest.acceptance:
            raise NoActiveAdapterError(f"{path} has no matching acceptance record under {acceptance.parent}")
        if adapter_file is not None:
            actual = file_sha256(adapter_file)
            if actual != manifest.adapter_sha256:
                raise NoActiveAdapterError(f"{adapter_file} hashes to {actual}; active manifest pins {manifest.adapter_sha256}")
        return manifest
