"""Synthetic `lilith/cos-snapshot@1` worlds, written with the recorder's own
storage functions so the format cannot drift. No private data."""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cos_recorder as rec

from dojo.episode import (
    EmitDocument,
    Episode,
    EpisodeSpec,
    RecipientClass,
    SendMessage,
    StepBudget,
    StepRecord,
    ToolCall,
    Voice,
)
from dojo.replay_server import DojoLedger, StaticAncestryOracle, episode_servers
from dojo.snapshot import SnapshotStore

PROJECT = "proj-0001"
PLAN_A = "plan-aaaa"
PLAN_B = "plan-bbbb"
MAIN_1 = "1" * 40
MAIN_2 = "2" * 40

TOOLS = [
    {"name": "list_work_items", "inputSchema": {"type": "object", "properties": {
        "project_id": {"type": "string"}, "page_token": {"type": "string"}, "plan_id": {"type": "string"}}}},
    {"name": "get_work_item_detail", "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}},
                                                     "required": ["id"]}},
    {"name": "get_wi_loop_state", "inputSchema": {"type": "object", "properties": {
        "work_item_id": {"type": "string"}}, "required": ["work_item_id"]}},
    {"name": "get_thread", "inputSchema": {"type": "object", "properties": {"work_item_id": {"type": "string"}}}},
    {"name": "get_plan", "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
    {"name": "list_plans", "inputSchema": {"type": "object", "properties": {"project_id": {"type": "string"}}}},
    {"name": "approve_and_merge", "inputSchema": {"type": "object", "properties": {
        "work_item_id": {"type": "string"}}, "required": ["work_item_id"]}},
    {"name": "post_message", "inputSchema": {"type": "object", "properties": {
        "content": {"type": "string"}, "work_item_id": {"type": "string"}}, "required": ["content"]}},
]


@dataclasses.dataclass
class WI:
    id: str
    phase: str
    plan: str | None = PLAN_A
    status: str = "open"
    parked_reason: str | None = None
    executor: str = "tecton"
    updated_at: datetime | None = None
    pending_deps: tuple[str, ...] = ()
    blockers: tuple[tuple[str, str], ...] = ()  # unsatisfied (id, phase) in the loop state
    ci_green: bool | None = None
    review_required: bool = False
    review_approved: bool = False
    ai_gate: bool | None = None
    human_accepted: bool | None = None
    prs: tuple[tuple[str, str], ...] = ()  # (change_id, head_sha)
    revisions: int = 0

    @property
    def live(self) -> bool:
        return self.phase not in ("done", "cancelled") and self.status != "cancelled"


class World:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def snapshot(self, when: datetime, items: list[WI], main_sha: str = MAIN_1, tools: list[dict] | None = None,
                 extra: tuple[tuple[str, dict, Any, dict], ...] = ()) -> str:
        """`extra`: more recorded reads, `(call, args, payload, manifest fields)`."""
        sid = when.strftime("%Y%m%dT%H%M%SZ")
        responses: list[dict] = []

        def put(call: str, args: dict, payload: Any) -> None:
            digest, _ = rec.store_object(self.root, payload)
            responses.append({"call": call, "args": args, "sha256": digest})

        put("list_plans", {"project_id": PROJECT}, {"plans": [{"id": PLAN_A, "title": "A"}, {"id": PLAN_B, "title": "B"}],
                                                    "next_page_token": None})
        put("list_work_items", {"project_id": PROJECT}, {"work_items": [
            {"id": w.id, "code": f"WI-{i}", "current_phase": w.phase, "status": w.status, "plan_id": w.plan,
             "priority": "high", "title": f"item {i}", "project_id": PROJECT} for i, w in enumerate(items)],
            "next_page_token": None, "total": len(items)})
        for w in items:
            if not w.live:
                continue
            updated = (w.updated_at or when).isoformat()
            put("get_work_item_detail", {"id": w.id}, {
                "work_item": {"id": w.id, "current_phase": w.phase, "status": w.status, "plan_id": w.plan,
                              "parked_reason": w.parked_reason, "executor": w.executor, "updated_at": updated,
                              "depends_on_ids": list(w.pending_deps),
                              "dependency_edges": [{"depends_on_id": d, "state": "pending", "work_item_id": w.id}
                                                   for d in w.pending_deps]},
                "prs": [{"change_id": c, "head_sha": h, "lifecycle_state": "open", "integration_decision": "pending"}
                        for c, h in w.prs],
                "change_history": [{"id": f"ch-{n}"} for n in range(w.revisions)],
            })
            put("get_wi_loop_state", {"work_item_id": w.id}, {
                "work_item_id": w.id, "current_phase": w.phase, "parked_reason": w.parked_reason,
                "unsatisfied_blockers": [{"work_item_id": b, "current_phase": p, "code": "WI-x"} for b, p in w.blockers],
                "exit_bar": {"ai_gate_cleared": w.ai_gate, "human_review_accepted": w.human_accepted},
                "pipeline_gates": {"ci": {"green_for_current_head": bool(w.ci_green),
                                          "current_head": None if w.ci_green is None else "abc"},
                                   "review": {"required": w.review_required,
                                              "approved_for_current_head": w.review_approved}},
            })
            put("get_scheduling_state", {"work_item_id": w.id}, {"work_item_id": w.id, "retry_count": 0})
        put("get_project_ci_status", {"project_id": PROJECT}, {"resolvable": True})
        for call, args, payload, fields in extra:
            put(call, args, payload)
            responses[-1].update(fields)
        contract_sha, _ = rec.store_object(self.root, {"tools": tools if tools is not None else TOOLS})
        refs_payload = {"remote": rec.GIT_REMOTE, "refs": [{"ref": "HEAD", "sha": main_sha},
                                                           {"ref": "refs/heads/main", "sha": main_sha}]}
        refs_sha, _ = rec.store_object(self.root, refs_payload)
        manifest = {
            "schema": rec.SCHEMA,
            "timestamp": when.isoformat(),
            "collectors": {
                "anima": {"status": "ok", "responses": responses, "errors": [], "tool_contract_sha256": contract_sha},
                "git": {"status": "ok", "errors": [],
                        "responses": [{"call": "git_ls_remote", "args": {"remote": rec.GIT_REMOTE}, "sha256": refs_sha}]},
                "board": {"status": "skipped", "responses": [], "errors": []},
                "fleet": {"status": "not_configured", "responses": [], "errors": []},
            },
        }
        rec.atomic_write_json(self.root / sid / "manifest.json", manifest)
        return sid

    def store(self) -> SnapshotStore:
        return SnapshotStore(self.root)


T0 = datetime(2026, 9, 28, 8, 0, tzinfo=timezone.utc)


def hours(n: float) -> timedelta:
    return timedelta(hours=n)


class Driver:
    """Plays an episode through the real replay servers, the way the akhrtd runner will."""

    def __init__(self, spec: EpisodeSpec, store: SnapshotStore, ledger_path: Path, oracle=None):
        self.spec = spec
        self.snapshot = store.get(spec.snapshot_id)
        self.servers, self.log = episode_servers(self.snapshot, DojoLedger(ledger_path), oracle or StaticAncestryOracle({}))
        self.steps: list[StepRecord] = []
        self._rpc = 0

    def call(self, tool: str, arguments: dict | None = None, server: str = "anima", voice: Voice | None = None) -> tuple[Any, str]:
        self._rpc += 1
        response = self.servers[server].handle({"jsonrpc": "2.0", "id": self._rpc, "method": "tools/call",
                                               "params": {"name": tool, "arguments": arguments or {}}})
        result = response["result"]
        receipt_id = result["_meta"]["lilith/receipt"]
        text = result["content"][0]["text"]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = text
        self.steps.append(StepRecord(self.spec.episode_id, len(self.steps), ToolCall(tool, arguments or {}, receipt_id, server),
                                     StepBudget(tokens=100), voice))
        return payload, receipt_id

    def receipt(self, receipt_id: str) -> dict:
        r = next(r for r in self.log.receipts if r.call_id == receipt_id)
        return {"kind": "tool-call", "reference": r.call_id, "observed_at": r.observed_at,
                "digest": f"sha256:{r.response_sha256}"}

    def emit(self, document: dict, voice: Voice | None = None, **kw) -> None:
        self.steps.append(StepRecord(self.spec.episode_id, len(self.steps), EmitDocument(document),
                                     StepBudget(tokens=200), voice, **kw))

    def send(self, recipient: str, cls: RecipientClass, payload: dict, sent_at: int, follow_up_at: int | None,
             protocol: str = "coordination/request@v1") -> None:
        self.steps.append(StepRecord(self.spec.episode_id, len(self.steps),
                                     SendMessage(recipient, cls, protocol, payload, sent_at, follow_up_at),
                                     StepBudget(tokens=150)))

    def episode(self) -> Episode:
        return Episode(self.spec, tuple(self.log.receipts), tuple(self.steps))
