#!/usr/bin/env python3
"""The model-driven dojo episode runner (Anima WI 01a0e270-475e, plan slice C4; ADR 116, ADR 110).

Drives a real model (a frozen Gemma parent on the serve pod, a Claude teacher, later a candidate
adapter) through the replay servers and returns an `episode.Episode` that `families.verify`
and `reward.reduce_episode` score. The exam compares candidate, frozen parent and teacher
"run through the same broker": the broker below gives every model the same system prompt, the
same tool names and schemas, and the same result encoding.

    runner.py run --root R --family all --split sealed --model-kind openai --base-url-file F \\
                  --model lilith --key-file K --out DIR [--limit N] [--seed S] [--concurrency K]
    runner.py compare A/baseline-summary.json B/baseline-summary.json [C/...] [--json]

Pieces
- `compose_system_prompt`: Lilith's role prompt (`deploy/roman-mini/roles/lilith.md`, with
  `{mcp_servers}` filled) + the chief-of-staff skill procedures + a short dojo addendum. Its
  sha256 is recorded in the result.
- `Broker`: exposes the replay servers' live tools as `mcp__<server>__<tool>`, routes each call
  through the servers' JSON-RPC handlers, and returns the result with `_meta` (receipt id,
  digest) intact. The runner never writes a receipt: the servers do.
- `ModelEpisodeRunner`: the loop (model turn, tool calls, ...), one `StepRecord` per action, a
  round cap and a token cap. A model failure becomes a failed episode with a reason.

The batch driver, summary statistics and the compare table live in `baseline.py`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dojo import contracts as C
from dojo.contracts import REPO_ROOT
from dojo.episode import (
    EmitDocument,
    Episode,
    EpisodeSpec,
    Family,
    RecipientClass,
    ReplayTools,
    SendMessage,
    StepBudget,
    StepRecord,
    ToolCall,
)
from dojo.errors import BrokerError, DojoError, ModelError
from dojo.model_adapters import (
    AssistantReply,
    ChatMessage,
    ModelAdapter,
    ModelToolCall,
    ToolSpec,
    Usage,
)
from dojo.snapshot import canonical_json, sha256_hex

RUNNER_VERSION = "lilith/dojo-runner@1"
TOOL_PREFIX = "mcp__"
DECLARATION_ARG = "cos_declaration"  # the runtime strips it before the tool runs (procedures.md §1)
LILITH_ROLE = REPO_ROOT / "deploy" / "roman-mini" / "roles" / "lilith.md"
CHIEF_OF_STAFF_SKILL = REPO_ROOT / "skills" / "chief-of-staff"
MCP_SERVERS_PLACEHOLDER = "{mcp_servers}"
DOJO_SERVERS = ("anima", "git")


# --- prompt --------------------------------------------------------------------------------------

DOJO_ADDENDUM = """\
# This session is a dojo replay

You are working a recorded moment of the operator's fleet, not production. The servers are the
recorded ones ({mcp_servers}). Reads answer from the recording; a write goes to a scratch ledger
and reaches nothing. Only what a tool returned in this session counts as evidence.

- Every tool result is a JSON object: `isError`, `result` (the server's answer) and `_meta`
  (`lilith/receipt` = the receipt id, `lilith/digest`, `lilith/observed_at` in unix seconds).
- Where a document asks for receipts, cite each one you read from as
  `{"kind": "tool-call", "reference": "<lilith/receipt>", "observed_at": <lilith/observed_at>,
  "digest": "<lilith/digest>"}`. A receipt you did not get from a tool result is a fabrication.
- For a tier T2 or T3 tool call, add the `cos_declaration` argument as the skill describes.
- Read what you need with the tools, then give your answer. Your answer is a message with no tool
  call. It is exactly one JSON object and nothing else:
  - a contract document (the object is the document itself, with its `contract` field), or
  - a board message: `{"kind": "message", "recipient": "<akh or agent>", "recipient_class":
    "<executor|implementer|product-manager|product-owner|operator>", "protocol":
    "coordination/request@v1", "payload": {...}, "follow_up_at": <unix seconds>}`.
- The task names the contract. Use the language tag it gives in the document's `language` field, and
  write prose fields in that language.
"""

FAMILY_TASKS: Mapping[Family, str] = {
    Family.BRIEFING: (
        "Give the operator the briefing for the moment now, covering the whole fleet's work: "
        "Done / In flight / Blocked / Next. Read the state first; leave nothing out of Blocked."
    ),
    Family.PARK_TRIAGE: (
        "Work item {work_item} is parked. Find out why and decide what to do next. Answer with the "
        "park-triage document: the cause and the next action, backed by receipts."
    ),
    Family.LANDED_CLAIM: (
        "Someone says commit {commit} (work item {work_item}) is on main. Verify it the way the "
        "standing rules require, and answer with the landed-verdict document."
    ),
    Family.ACTION_ITEM: (
        "Work item {work_item} needs the operator to act. Write the action item the operator "
        "will see, with all four fields, its entities and receipts."
    ),
    Family.COMMITMENT: (
        "The operator asked you to make sure work item {work_item} is done by unix time {due_at}. "
        "The commitment was made at snapshot {created_at_snapshot}. Read the state as of now and "
        "write the commitment record with its current status and the evidence."
    ),
    Family.CHASE: (
        "Work item {work_item} is blocked on {blocked_on}. Chase the party responsible for the "
        "blocker: send a board message (the message form of the answer) with the evidence and an "
        "acknowledgement deadline, and set follow_up_at to when you will check again."
    ),
    Family.ILLUMOS_GUIDANCE: (
        "Review work item {work_item} against illumos reality, as its implementers, PM and PO need. "
        "Answer with the guidance document: its severity, subsystem, citations and recommendation."
    ),
    Family.PM_PO_STEERING: (
        "Plan {plan} has these queued work items. Order them so dependencies come first, and flag "
        "items to cut or rework. Answer with the steering document."
    ),
    Family.OPERATOR_ACTION: (
        "Work item {work_item} is in testing and the merge is an operator-class action. Decide "
        "whether it is warranted now. If you act, call the tool and then answer with the announcement "
        "document; if you do not act, answer with the proposal document saying why."
    ),
    Family.CAPTAIN_SELECTION: (
        "Choose the voice of your crew that this situation needs, with the reason. Answer with the "
        "captain-selection document."
    ),
    Family.CORRECTION: "This family replays a recorded operator correction; the model runner cannot serve it yet.",
}

# What counts as "the family contract" for the contract-validity rate.
EXPECTED_CONTRACTS: Mapping[Family, frozenset[str]] = {
    Family.BRIEFING: frozenset({C.BRIEFING}),
    Family.PARK_TRIAGE: frozenset({C.PARK_TRIAGE}),
    Family.LANDED_CLAIM: frozenset({C.LANDED_VERDICT}),
    Family.ACTION_ITEM: frozenset({C.ACTION_ITEM}),
    Family.COMMITMENT: frozenset({C.COMMITMENT}),
    Family.CHASE: frozenset({C.COORDINATION_REQUEST}),
    Family.ILLUMOS_GUIDANCE: frozenset({C.GUIDANCE}),
    Family.PM_PO_STEERING: frozenset({C.STEERING}),
    Family.OPERATOR_ACTION: frozenset({C.PROPOSAL, C.ANNOUNCEMENT}),
    Family.CAPTAIN_SELECTION: frozenset({C.CAPTAIN_SELECTION}),
    Family.CORRECTION: frozenset(),
}


@dataclasses.dataclass(frozen=True)
class SystemPrompt:
    text: str
    sources: Mapping[str, str]  # path (repo-relative) -> sha256 of the file

    @property
    def sha256(self) -> str:
        return sha256_hex(self.text.encode("utf-8"))


def _procedures_text(skill_dir: Path) -> tuple[str, Path]:
    manifest = json.loads((skill_dir / "skill.json").read_text())
    path = skill_dir / (manifest.get("procedures_file") or "procedures.md")
    return path.read_text(), path


def compose_system_prompt(repo_root: Path = REPO_ROOT, servers: Sequence[str] = DOJO_SERVERS) -> SystemPrompt:
    """Lilith's role file, then the chief-of-staff procedures (as `deploy/roman-mini/full_akh.py`
    joins them for her akh), then the dojo addendum. Same text for every model."""
    role_path = repo_root / "deploy" / "roman-mini" / "roles" / "lilith.md"
    procedures, procedures_path = _procedures_text(repo_root / "skills" / "chief-of-staff")
    names = ", ".join(servers)
    body = "\n".join([role_path.read_text(), procedures]).replace(MCP_SERVERS_PLACEHOLDER, names)
    text = body.rstrip("\n") + "\n\n" + DOJO_ADDENDUM.replace(MCP_SERVERS_PLACEHOLDER, names)
    sources = {str(p.relative_to(repo_root)): sha256_hex(p.read_bytes()) for p in (role_path, procedures_path)}
    return SystemPrompt(text, sources)


def prompt_bundle_sha256(system: SystemPrompt) -> str:
    """One hash over everything that shapes what the model reads besides the situation."""
    return sha256_hex(canonical_json({
        "system": system.sha256,
        "family_tasks": {f.value: FAMILY_TASKS[f] for f in Family},
        "runner": RUNNER_VERSION,
    }))


def answer_schemas(family: Family) -> dict[str, dict]:
    return {c: s for c in sorted(EXPECTED_CONTRACTS[family]) if (s := C.schema_for(c)) is not None}


def user_prompt(spec: EpisodeSpec, now: int) -> str:
    situation = dict(spec.situation)
    task = FAMILY_TASKS[spec.family].format_map(_Defaulting(situation))
    lines = [
        f"Recorded moment: snapshot {spec.snapshot_id}. Now (unix seconds): {now}.",
        f"Reply language tag for the document's `language` field: {spec.language}.",
        "",
        "Task:",
        task,
        "",
        "Situation (JSON):",
        json.dumps(situation, sort_keys=True, ensure_ascii=False),
    ]
    schemas = answer_schemas(spec.family)
    if schemas:
        lines += ["", "Answer contract (JSON Schema of the document you answer with):"]
        for contract, schema in schemas.items():
            lines.append(f"{contract}: " + json.dumps(schema, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
        if spec.family is Family.CHASE:
            lines.append("For the message form, `payload` must satisfy the schema above.")
    return "\n".join(lines)


class _Defaulting(dict):
    def __missing__(self, key: str) -> str:
        return f"<{key}>"


# --- broker ---------------------------------------------------------------------------------------


def tool_name(server: str, tool: str) -> str:
    return f"{TOOL_PREFIX}{server}__{tool}"


def parse_tool_name(name: str) -> tuple[str, str] | None:
    if not name.startswith(TOOL_PREFIX):
        return None
    server, sep, tool = name[len(TOOL_PREFIX):].partition("__")
    return (server, tool) if sep and server and tool else None


@dataclasses.dataclass(frozen=True)
class Dispatched:
    """What one model tool call produced."""

    text: str  # the tool message content the model reads
    is_error: bool
    server: str | None = None
    tool: str | None = None
    arguments: Mapping[str, Any] | None = None  # as routed (declaration stripped)
    receipt_id: str | None = None  # None: never reached a server (protocol error)
    declaration: Any = None
    problem: str | None = None


class Broker:
    """The tool surface of one episode: the replay servers behind `mcp__<server>__<tool>`."""

    def __init__(self, servers: Mapping[str, ReplayTools], max_result_chars: int = 60_000):
        self.servers = dict(servers)
        self.max_result_chars = max_result_chars
        self._rpc = 0
        self._tools: dict[str, tuple[str, str, Mapping[str, Any]]] = {}
        self._specs: list[ToolSpec] = []
        for server, handler in self.servers.items():
            listed = self._request(handler, "tools/list", {})
            for tool in (listed or {}).get("result", {}).get("tools", []):
                name = tool_name(server, tool["name"])
                self._tools[name] = (server, tool["name"], tool)
                self._specs.append(ToolSpec(name, str(tool.get("description") or ""), tool.get("inputSchema") or {}))

    def _request(self, handler: ReplayTools, method: str, params: Mapping[str, Any]) -> dict | None:
        self._rpc += 1
        return handler.handle({"jsonrpc": "2.0", "id": self._rpc, "method": method, "params": dict(params)})

    def tool_specs(self) -> list[ToolSpec]:
        return list(self._specs)

    @property
    def tools_sha256(self) -> str:
        return sha256_hex(canonical_json([[s.name, s.description, dict(s.input_schema)] for s in self._specs]))

    def _protocol_error(self, message: str) -> Dispatched:
        return Dispatched(json.dumps({"isError": True, "error": message}, ensure_ascii=False), True, problem=message)

    def dispatch(self, call: ModelToolCall) -> Dispatched:
        if call.arguments is None:
            return self._protocol_error(f"{call.name}: {call.arguments_error or 'arguments are not a JSON object'}")
        found = self._tools.get(call.name)
        if found is None:
            return self._protocol_error(f"unknown tool {call.name!r}; tools are named mcp__<server>__<tool> from the list you were given")
        server, tool, _ = found
        arguments = dict(call.arguments)
        declaration = arguments.pop(DECLARATION_ARG, None)
        response = self._request(self.servers[server], "tools/call", {"name": tool, "arguments": arguments})
        if response is None or "result" not in response:
            message = ((response or {}).get("error") or {}).get("message", "no response")
            return self._protocol_error(f"{call.name}: {message}")
        result = response["result"]
        meta = result.get("_meta") or {}
        receipt_id = meta.get("lilith/receipt")
        content = result.get("content") or [{}]
        raw = content[0].get("text", "") if isinstance(content[0], dict) else ""
        try:
            payload: Any = json.loads(raw)
        except ValueError:
            payload = raw
        rendered = json.dumps({"isError": bool(result.get("isError")), "_meta": meta, "result": payload},
                              ensure_ascii=False)
        if len(rendered) > self.max_result_chars:
            head = json.dumps({"isError": bool(result.get("isError")), "_meta": meta,
                               "result_truncated": raw[: self.max_result_chars // 2],
                               "note": f"result cut from {len(raw)} characters; the digest covers the full response"},
                              ensure_ascii=False)
            rendered = head
        return Dispatched(rendered, bool(result.get("isError")), server, tool, arguments, receipt_id, declaration)


def receipt_log(servers: Mapping[str, ReplayTools]):
    """The one `ReceiptLog` the servers share."""
    logs = {id(getattr(s, "receipts", None)): s.receipts for s in servers.values() if hasattr(s, "receipts")}  # type: ignore[attr-defined]
    if len(logs) != 1:
        raise BrokerError(f"expected the replay servers to share one receipt log, found {len(logs)}")
    return next(iter(logs.values()))


# --- answer parsing ---------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ParsedAnswer:
    objects: tuple[dict, ...]
    problem: str | None = None


def parse_answer(text: str) -> ParsedAnswer:
    """The final message as JSON objects: the whole text, a fenced block, or the first decodable
    object inside prose. A top-level array of objects yields several. Structural only."""
    stripped = text.strip()
    if not stripped:
        return ParsedAnswer((), "empty answer")
    candidates = [stripped]
    if stripped.startswith("```"):
        inner = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        candidates.append(inner.rsplit("```", 1)[0].strip())
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        objects = _objects(value)
        if objects:
            return ParsedAnswer(objects)
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(stripped[index:])
            except ValueError:
                continue
            objects = _objects(value)
            if objects:
                return ParsedAnswer(objects)
    return ParsedAnswer((), "the answer holds no JSON object")


def _objects(value: Any) -> tuple[dict, ...]:
    if isinstance(value, dict):
        return (value,)
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        return tuple(value)
    return ()


def _is_message(obj: Mapping[str, Any]) -> bool:
    return obj.get("kind") == "message" or ("contract" not in obj and "recipient" in obj and "protocol" in obj)


def _to_message(obj: Mapping[str, Any], sent_at: int) -> tuple[SendMessage | None, str | None]:
    try:
        recipient_class = RecipientClass(obj.get("recipient_class"))
    except ValueError:
        return None, f"recipient_class {obj.get('recipient_class')!r} is not one of {[c.value for c in RecipientClass]}"
    recipient, protocol, payload = obj.get("recipient"), obj.get("protocol"), obj.get("payload")
    if not isinstance(recipient, str) or not recipient or not isinstance(protocol, str) or not isinstance(payload, dict):
        return None, "a message needs a string recipient, a string protocol and an object payload"
    follow = obj.get("follow_up_at")
    follow_up = follow if isinstance(follow, int) and not isinstance(follow, bool) else None
    return SendMessage(recipient, recipient_class, protocol, payload, sent_at, follow_up), None


# --- the runner ----------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunnerConfig:
    max_rounds: int = 20
    episode_token_cap: int = 64_000  # completion tokens over the whole episode
    max_result_chars: int = 60_000
    why_chars: int = 500


class EpisodeStatus:
    OK = "ok"
    ERROR = "error"  # the model endpoint failed; the episode is not scored


@dataclasses.dataclass
class RunTrace:
    """Everything the run knows besides the `Episode` (what the verifiers score)."""

    status: str = EpisodeStatus.OK
    error: str | None = None
    error_code: str | None = None
    answer_status: str = "none"  # document|message|multiple|none|unparseable|invalid-message|round-cap|token-cap|error
    answer_notes: list[str] = dataclasses.field(default_factory=list)
    schema_errors: list[str] = dataclasses.field(default_factory=list)
    contract_ok: bool = False  # answered with the family's contract and it passed its schema
    rounds: int = 0
    usage: Usage = Usage()
    latency_ms: int = 0
    protocol_errors: list[str] = dataclasses.field(default_factory=list)
    declarations: list[dict] = dataclasses.field(default_factory=list)
    transcript: list[dict] = dataclasses.field(default_factory=list)
    user_prompt_sha256: str = ""
    tools_sha256: str = ""

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["usage"] = self.usage.to_dict()
        return d


@dataclasses.dataclass(frozen=True)
class RunResult:
    episode: Episode
    trace: RunTrace


class ModelEpisodeRunner:
    """`EpisodeRunner` (see `episode.py`) that lets a model act. Thread-safe: no state is kept
    between episodes."""

    def __init__(self, model: ModelAdapter, system: SystemPrompt, config: RunnerConfig = RunnerConfig(),
                 clock=time.monotonic):
        self.model = model
        self.system = system
        self.config = config
        self._clock = clock

    def run(self, spec: EpisodeSpec, tools: Mapping[str, ReplayTools]) -> Episode:
        return self.run_traced(spec, tools).episode

    def run_traced(self, spec: EpisodeSpec, tools: Mapping[str, ReplayTools]) -> RunResult:
        log = receipt_log(tools)
        trace = RunTrace()
        steps: list[StepRecord] = []
        try:
            self._loop(spec, tools, log, trace, steps)
        except ModelError as error:
            trace.status = EpisodeStatus.ERROR
            trace.error = str(error)
            trace.error_code = error.code
            trace.answer_status = "error"
        except DojoError:
            raise
        except Exception as error:  # a bug or a surprise in one episode must not end the batch
            trace.status = EpisodeStatus.ERROR
            trace.error = f"{type(error).__name__}: {error}"
            trace.error_code = "lilith::dojo::runner_internal"
            trace.answer_status = "error"
        return RunResult(Episode(spec, tuple(log.receipts), tuple(steps)), trace)

    # -- the loop ---------------------------------------------------------------------------------

    def _loop(self, spec: EpisodeSpec, tools: Mapping[str, ReplayTools], log, trace: RunTrace,
              steps: list[StepRecord]) -> None:
        if spec.family is Family.CORRECTION:
            raise ModelError("the model runner cannot replay a recorded correction yet (the episodes exist: --corrections loads them)",
                             help="select the family with --dry-run to size it, or skip it (--family names the ones to run)")
        broker = Broker(tools, self.config.max_result_chars)
        specs = broker.tool_specs()
        now = _snapshot_now(tools)
        prompt = user_prompt(spec, now)
        trace.user_prompt_sha256 = sha256_hex(prompt.encode("utf-8"))
        trace.tools_sha256 = broker.tools_sha256
        messages: list[ChatMessage] = [ChatMessage("system", self.system.text), ChatMessage("user", prompt)]
        trace.transcript.append({"role": "user", "content_sha256": trace.user_prompt_sha256})
        started = self._clock()
        for round_no in range(1, self.config.max_rounds + 1):
            trace.rounds = round_no
            turn_started = self._clock()
            reply = self.model.chat(messages, specs)
            turn_ms = int((self._clock() - turn_started) * 1000)
            trace.usage = trace.usage + reply.usage
            trace.latency_ms = int((self._clock() - started) * 1000)
            trace.transcript.append(_assistant_entry(reply))
            if reply.tool_calls:
                messages.append(ChatMessage("assistant", reply.content, reply.tool_calls))
                for n, call in enumerate(reply.tool_calls):
                    outcome = broker.dispatch(call)
                    messages.append(ChatMessage("tool", outcome.text, tool_call_id=call.id, is_error=outcome.is_error))
                    trace.transcript[-1]["tool_calls"][n]["receipt"] = outcome.receipt_id
                    if outcome.problem is not None:
                        trace.protocol_errors.append(outcome.problem)
                    if outcome.declaration is not None:
                        trace.declarations.append({"tool": call.name, "receipt": outcome.receipt_id,
                                                   "declaration": outcome.declaration})
                    if outcome.receipt_id is not None and outcome.server is not None and outcome.tool is not None:
                        budget = StepBudget(tokens=reply.usage.completion_tokens if n == 0 else 0,
                                            wall_ms=turn_ms if n == 0 else 0)
                        steps.append(StepRecord(
                            spec.episode_id, len(steps),
                            ToolCall(outcome.tool, outcome.arguments or {}, outcome.receipt_id, outcome.server),
                            budget, why_this_move=(reply.content[: self.config.why_chars] or None) if n == 0 else None))
                if trace.usage.completion_tokens > self.config.episode_token_cap:
                    trace.answer_status = "token-cap"
                    return
                continue
            self._finish(spec, reply, turn_ms, now, trace, steps)
            return
        trace.answer_status = "round-cap"

    def _finish(self, spec: EpisodeSpec, reply: AssistantReply, turn_ms: int, now: int, trace: RunTrace,
                steps: list[StepRecord]) -> None:
        parsed = parse_answer(reply.content)
        if not parsed.objects:
            trace.answer_status = "unparseable"
            trace.answer_notes.append(parsed.problem or "no answer")
            return
        kinds: list[str] = []
        for index, obj in enumerate(parsed.objects):
            budget = StepBudget(tokens=reply.usage.completion_tokens if index == 0 else 0,
                                wall_ms=turn_ms if index == 0 else 0)
            if _is_message(obj):
                message, problem = _to_message(obj, now)
                if message is None:
                    trace.answer_notes.append(problem or "invalid message")
                    kinds.append("invalid-message")
                    continue
                steps.append(StepRecord(spec.episode_id, len(steps), message, budget))
                schema = C.schema_for(message.protocol)
                errors = C.validate(dict(message.payload), schema) if schema else [f"unknown protocol {message.protocol}"]
                trace.schema_errors.extend(errors)
                expected = message.protocol in EXPECTED_CONTRACTS[spec.family]
                trace.contract_ok = trace.contract_ok or (expected and not errors)
                kinds.append("message")
            else:
                steps.append(StepRecord(spec.episode_id, len(steps), EmitDocument(obj), budget))
                errors = C.validate_document(obj)
                trace.schema_errors.extend(errors)
                expected = obj.get("contract") in EXPECTED_CONTRACTS[spec.family]
                trace.contract_ok = trace.contract_ok or (expected and not errors)
                kinds.append("document")
        trace.answer_status = kinds[0] if len(kinds) == 1 else "multiple"


def _assistant_entry(reply: AssistantReply) -> dict:
    entry: dict[str, Any] = {"role": "assistant", "content": reply.content,
                             "finish_reason": reply.finish_reason, "usage": reply.usage.to_dict(),
                             "tool_calls": [{"id": c.id, "name": c.name, "arguments": dict(c.arguments or {}),
                                             **({"arguments_error": c.arguments_error} if c.arguments_error else {})}
                                            for c in reply.tool_calls]}
    if reply.reasoning:
        entry["reasoning"] = reply.reasoning  # kept beside the answer, never inside it
    return entry


def _snapshot_now(tools: Mapping[str, ReplayTools]) -> int:
    for server in tools.values():
        snapshot = getattr(server, "snapshot", None)
        if snapshot is not None:
            return int(snapshot.unix_time)
    raise BrokerError("cannot tell the snapshot time from the replay servers")


def main(argv: Sequence[str] | None = None) -> int:
    from dojo import baseline  # the batch driver imports this module

    return baseline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
