"""The episode interface between the dojo and the akhrtd-owned episode runner.

The runner (a separate step: Lilith driven through the session tool broker)
receives an `EpisodeSpec` and a replay server, lets Lilith act, and hands back
an `Episode`: the receipts the *server* issued (the runner never writes a
receipt itself) plus one `StepRecord` per action, in the ADR 110 step-record
shape. The verifiers in `families` score an `Episode` against its spec.
"""
from __future__ import annotations

import dataclasses
import enum
import json
from typing import Any, Mapping, Protocol, Union

from dojo.snapshot import Provenance, canonical_json, sha256_hex


class Family(enum.StrEnum):
    BRIEFING = "briefing"
    PARK_TRIAGE = "park-triage"
    LANDED_CLAIM = "landed-claim"
    ACTION_ITEM = "action-item"
    COMMITMENT = "commitment"
    CHASE = "chase"
    ILLUMOS_GUIDANCE = "illumos-guidance"
    PM_PO_STEERING = "pm-po-steering"
    OPERATOR_ACTION = "operator-action"
    CAPTAIN_SELECTION = "captain-selection"
    CORRECTION = "correction"
    SCOPE = "scope"  # the prompt names a project; resolve it to its id (C2 round 1, decision 2)
    READ_FAILURE = "read-failure"  # a park triage whose detail read is unavailable (decision 4)
    MEMORY_EPISTEMICS = "memory-epistemics"  # her own recalled answers are beliefs to re-verify (operator decision 7)


class Voice(enum.StrEnum):
    """The crew of eight (ADR 110 Decision 1; English names from dojo manual §2a)."""

    MAKER = "maker"  # 1 Courage, Optimism, Urgency
    TRUTHKEEPER = "truthkeeper"  # 2 Rigor, Transparency
    SEEKER = "seeker"  # 3 Curiosity, Versatility
    CHALLENGER = "challenger"  # 4 Candor
    GUARDIAN = "guardian"  # 5 Empathy, Responsibility
    SACRED_CLOWN = "sacred-clown"  # 6 Humor
    STEWARD = "steward"  # 7 Resilience, Thriftiness
    WEAVER = "weaver"  # 8 Teamwork, Diversity

    @property
    def slot(self) -> int:
        return list(Voice).index(self) + 1


class ReceiptSource(enum.StrEnum):
    SNAPSHOT = "snapshot"  # answered from recorded bytes
    LEDGER = "ledger"  # a write, recorded in the dojo tenancy ledger
    ORACLE = "oracle"  # computed deterministically (ancestry check against recorded refs)
    ERROR = "error"  # the server answered with an error (unknown tool, unrecorded read)
    MEMORY = "memory"  # her own episodic memory answered: a retelling, never evidence of the tracker's state


class RecipientClass(enum.StrEnum):
    EXECUTOR = "executor"  # the akh/agent executing the blocking WI
    IMPLEMENTER = "implementer"
    PRODUCT_MANAGER = "product-manager"
    PRODUCT_OWNER = "product-owner"
    OPERATOR = "operator"


@dataclasses.dataclass(frozen=True)
class ToolReceipt:
    """Issued by the replay server for every tools/call. `call_id` is what a document
    cites as a receipt `reference`; `response_sha256` is its `digest`."""

    call_id: str
    server: str
    tool: str
    arguments: Mapping[str, Any]
    response: Any
    response_sha256: str
    source: ReceiptSource
    snapshot_id: str
    observed_at: int
    # Replay-fidelity audit trail. `arguments` is always what the caller asked; when the
    # server answered from a recorded call whose arguments differ but are provably
    # equivalent (schema-driven, see `replay_match`), `matched_arguments` is the recorded
    # call and `match_rules` names each normalisation applied. `error_reason` is the
    # machine-readable cause of an `ERROR` receipt.
    matched_arguments: Mapping[str, Any] | None = None
    match_rules: tuple[str, ...] = ()
    error_reason: str | None = None

    @property
    def is_error(self) -> bool:
        return self.source is ReceiptSource.ERROR

    @property
    def is_read(self) -> bool:
        return self.source in (ReceiptSource.SNAPSHOT, ReceiptSource.ORACLE)

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "server": self.server,
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "response_sha256": self.response_sha256,
            "source": self.source.value,
            "snapshot_id": self.snapshot_id,
            "observed_at": self.observed_at,
            **({"matched_arguments": dict(self.matched_arguments), "match_rules": list(self.match_rules)}
               if self.match_rules else {}),
            **({"error_reason": self.error_reason} if self.error_reason else {}),
        }


@dataclasses.dataclass(frozen=True)
class ToolCall:
    tool: str
    arguments: Mapping[str, Any]
    receipt_id: str  # the ToolReceipt.call_id the server issued for this call
    server: str = "anima"


@dataclasses.dataclass(frozen=True)
class EmitDocument:
    """A contract document (`lilith/cos-*@1` or a `lilith/dojo-*@1` answer)."""

    document: Mapping[str, Any]

    @property
    def contract(self) -> str | None:
        value = self.document.get("contract")
        return value if isinstance(value, str) else None


@dataclasses.dataclass(frozen=True)
class SendMessage:
    """A board / thread post (`coordination/request@v1`, ...)."""

    recipient: str
    recipient_class: RecipientClass
    protocol: str
    payload: Mapping[str, Any]
    sent_at: int
    follow_up_at: int | None = None


Action = Union[ToolCall, EmitDocument, SendMessage]


@dataclasses.dataclass(frozen=True)
class StepBudget:
    tokens: int = 0
    tool_calls: int = 0
    wall_ms: int = 0


@dataclasses.dataclass(frozen=True)
class BudgetCaps:
    tokens: int = 32_000
    tool_calls: int = 24
    wall_ms: int = 300_000


@dataclasses.dataclass(frozen=True)
class StepRecord:
    episode_id: str
    step: int
    action: Action
    budget: StepBudget = StepBudget()
    voice_shown: Voice | None = None
    observation_ref: str | None = None
    why_this_move: str | None = None
    follows_correction: bool = False  # the step immediately after an operator correction

    @property
    def signature(self) -> str:
        """Structural identity of the move, used by the correction family."""
        action = self.action
        if isinstance(action, ToolCall):
            body: Any = {"tool": action.tool, "arguments": dict(action.arguments)}
        elif isinstance(action, EmitDocument):
            body = {"emit": dict(action.document)}
        else:
            body = {"send": action.protocol, "to": action.recipient, "payload": dict(action.payload)}
        return sha256_hex(canonical_json(body))

    def to_dict(self) -> dict:
        action = self.action
        if isinstance(action, ToolCall):
            rendered: dict = {"kind": "tool_call", "tool": action.tool, "server": action.server,
                              "arguments": dict(action.arguments), "receipt": action.receipt_id}
        elif isinstance(action, EmitDocument):
            rendered = {"kind": "emit", "document": dict(action.document)}
        else:
            rendered = {"kind": "send", "recipient": action.recipient,
                        "recipient_class": action.recipient_class.value, "protocol": action.protocol,
                        "payload": dict(action.payload), "sent_at": action.sent_at,
                        "follow_up_at": action.follow_up_at}
        return {
            "episode_id": self.episode_id,
            "step": self.step,
            "observation_ref": self.observation_ref,
            "voice_shown": self.voice_shown.value if self.voice_shown else None,
            "action": rendered,
            "budget": dataclasses.asdict(self.budget),
            "why_this_move": self.why_this_move,
        }


@dataclasses.dataclass(frozen=True)
class RecalledMemory:
    """One episodic memory the dojo's `memory_recall` replay can return: what her akh stored, verbatim.

    `text` is her own earlier answer (private; never serialised into the spec, only its digest is)."""

    label: str  # e.g. `episode:turn:5347`, as `memory_recall` prints it
    text: str  # the stored summary
    tags: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    order: int = 0  # recency: larger = newer

    @property
    def text_sha256(self) -> str:
        return sha256_hex(self.text.encode("utf-8"))

    def to_dict(self) -> dict:
        return {"label": self.label, "sha256": self.text_sha256, "tags": list(self.tags), "facts": list(self.facts),
                "order": self.order}


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    """One situation. `snapshot_id` is served; `label_snapshot_ids` are never served,
    they only feed hindsight labels."""

    family: Family
    snapshot_id: str
    label_snapshot_ids: tuple[str, ...]
    plan_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    situation: Mapping[str, Any]
    label: Mapping[str, Any]
    provenance: Provenance
    language: str = "en"
    voice_called_for: Voice | None = None
    caps: BudgetCaps = BudgetCaps()
    # Tools the replay answers with an error although the snapshot holds their data: the episode is a
    # failed-read drill. Never shown to the model; the receipt says the tool is unavailable.
    replay_faults: tuple[str, ...] = ()
    # Episodic memories the replay's `memory_recall` serves (memory-epistemics). Empty = the `akh` server is not offered.
    memories: tuple[RecalledMemory, ...] = ()

    @property
    def episode_id(self) -> str:
        subject = sha256_hex(canonical_json(sorted(self.subject_ids)))[:12]
        return f"{self.family.value}:{self.snapshot_id}:{subject}"

    @property
    def horizon_end(self) -> str:
        return max((self.snapshot_id, *self.label_snapshot_ids))

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "family": self.family.value,
            "snapshot": self.snapshot_id,
            "label_snapshots": list(self.label_snapshot_ids),
            "plans": list(self.plan_ids),
            "subjects": list(self.subject_ids),
            "situation": self.situation,
            "label": self.label,
            "language": self.language,
            "voice_called_for": self.voice_called_for.value if self.voice_called_for else None,
            "caps": dataclasses.asdict(self.caps),
            "provenance": self.provenance.to_dict(),
            **({"replay_faults": list(self.replay_faults)} if self.replay_faults else {}),
            **({"memories": [m.to_dict() for m in self.memories]} if self.memories else {}),
        }

    @property
    def spec_sha256(self) -> str:
        return sha256_hex(canonical_json(self.to_dict()))

    def prompt_text(self) -> str:
        """Text the decontamination index sees for this spec (situation only, never the label)."""
        return json.dumps(self.situation, sort_keys=True, ensure_ascii=False)


@dataclasses.dataclass(frozen=True)
class Episode:
    spec: EpisodeSpec
    receipts: tuple[ToolReceipt, ...]
    steps: tuple[StepRecord, ...]

    def receipt(self, call_id: str) -> ToolReceipt | None:
        for r in self.receipts:
            if r.call_id == call_id:
                return r
        return None

    def receipt_by_digest(self, digest: str) -> ToolReceipt | None:
        bare = digest.split(":", 1)[-1]
        for r in self.receipts:
            if r.response_sha256 == bare:
                return r
        return None

    @property
    def read_receipts(self) -> tuple[ToolReceipt, ...]:
        return tuple(r for r in self.receipts if r.is_read)


class ReplayTools(Protocol):
    """What the runner talks to: a JSON-RPC request in, a response out."""

    def handle(self, request: Mapping[str, Any]) -> dict | None: ...


class EpisodeRunner(Protocol):
    """Implemented by the akhrtd-owned runner (separate step). Must return the receipts
    the replay servers issued, unmodified."""

    def run(self, spec: EpisodeSpec, tools: Mapping[str, ReplayTools]) -> Episode: ...
