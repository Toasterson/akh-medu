"""The teacher arm through an akh over ACP (Anima WI 01a0e270-475e, plan slice C4; ADR 116, ADR 110).

Operator decision 2026-09-29: the teacher makes no batch or direct model-API calls; it runs as an
akh. Design: docs/ai/plans/2026-09-29-lilith-c4-teacher-via-akh.md.

`AcpEpisodeRunner` is an `EpisodeRunner` (same `run_traced(spec, tools) -> RunResult` as
`runner.ModelEpisodeRunner`) that lets a dedicated **dojo teacher akh** act on each episode:

1. The episode's replay servers stay in this process. `mcp_gateway.ReplayMcpGateway` serves them over
   loopback MCP HTTP behind a per-episode token, through `runner.Broker.call`: same tool names
   (`mcp__anima__...`, `mcp__git__...`), same recorded schemas, same result bytes, same receipts in
   the episode's one `ReceiptLog`, same `cos_declaration` stripping.
2. It speaks ACP (newline-delimited JSON-RPC over akhomed's loopback TCP listener, the protocol
   `akh-acp --akhomed` bridges): `initialize` (bearer in `_meta.authorization`), then the isolation
   preflight, then `session/new` with the episode's gateway servers in `mcpServers` (akhomed bridges
   client-pushed HTTP MCP servers as `mcp__<name>__<tool>`, `src/acp/mcp_bridge.rs`), then one
   `session/prompt`. Where the dojo system prompt goes is `PromptPlacement`: by default it is the
   akh's own `system_prompt` (rendered from the same text, verified before the run) and the prompt
   turn carries only the task, the parent's exact role layout; the legacy `SESSION_TURN` placement
   sends both in the first user turn.
3. Isolation is enforced, not assumed (the `IsolationRefusal` error aborts the whole batch):
   - preflight `_akh/tools`: the akh's own registry must be empty. Any native or static-MCP tool
     (a live tracker, git, shell, web) refuses the run before the model sees a prompt;
   - during the turn every `tool_call` / permission request must name a replay tool; anything else
     cancels the turn and refuses the batch;
   - the akh's memory must not grow: `_akh/health` symbol and triple counts are read before and after
     every episode (episodes run one at a time), so nothing an episode read can reach the next one;
   - the gateway itself only answers the replay tools, only on loopback, only while the episode is open.
4. The `Episode` is assembled from what the gateway routed (one `StepRecord` per call, receipts from
   the servers unmodified) plus the final answer, read by the shared `runner.finish_answer`, so
   `families.verify`, `reward.reduce_episode` and `baseline.compare` work unchanged.

stdlib only. The tests drive it against a fake ACP peer (`tests/fake_acp.py`).
"""
from __future__ import annotations

import dataclasses
import enum
import json
import queue
import re
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dojo.episode import Episode, EpisodeSpec, Family, ReplayTools, StepBudget, StepRecord, ToolCall
from dojo.errors import DojoError, ModelError
from dojo.mcp_gateway import EpisodeTools, ReplayMcpGateway
from dojo.model_adapters import Secret, Usage
from dojo.runner import (
    Broker,
    EpisodeStatus,
    RunnerConfig,
    RunResult,
    RunTrace,
    SystemPrompt,
    _snapshot_now,
    finish_answer,
    parse_answer,
    receipt_log,
    user_prompt,
)
from dojo.snapshot import sha256_hex

ACP_PROTOCOL_VERSION = 1
KIND = "acp"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
# `ToolExecutionTracker::generate_tool_id` = `<tool name>_<counter>` (src/agent/tool_execution_tracker.rs).
_TOOL_ID = re.compile(r"^(?P<name>.+)_(?P<n>\d+)$")
# How akhomed renders `AkhMessage::Error` as a message chunk (src/acp/mod.rs `to_session_update`).
_AKH_ERROR = re.compile(r"^\s*error\[[^\]]+\]:")

WARM_UP_PROMPT = "Reply with the single word: ready. Do not call any tool."
# Symbols listed per warm-up report: labels only, so a one-time init is named without dumping memory.
WARM_UP_LABELS_SHOWN = 20

PROMPT_PREAMBLE = """\
The two sections below are your complete instructions for this session. The first is your system
prompt, the second is the task. Use only the tools listed for this session.
"""


class PromptPlacement(enum.StrEnum):
    """Where the teacher receives the dojo system prompt (the in-process parent gets it as the system message)."""

    AKH_SYSTEM = "akh-system-prompt"  # the akh's rendered `system_prompt`; the prompt turn is the task alone
    SESSION_TURN = "acp-session-prompt"  # inside the first user turn, under the akh's own system prompt


class PromptParityRefusal(DojoError):
    """The teacher's active system prompt is not the run's dojo prompt."""

    code = "lilith::dojo::teacher_prompt_parity"
    help = (
        "render the dojo prompt into the teacher akh: leave `prompt` unset (or `\"dojo\"`) in "
        "config/dojo-teacher.json, run `python3 deploy/roman-mini/configure_core.py --plan` then the render "
        "from a clean origin/main checkout, restart the core so akhomed reloads the workspace, and pass the "
        "rendered core-config/akh-medu/workspaces/dojo-teacher.toml as --akh-config; or, to send the prompt "
        "in the first user turn instead (a recorded placement difference), pass `--teacher-prompt session`"
    )


class IsolationRefusal(DojoError):
    """The teacher akh could reach something besides the episode's replay tools. The batch stops."""

    code = "lilith::dojo::teacher_isolation"
    help = (
        "the teacher must be a sealed dojo akh: no native tools, no [[mcp.servers]], exec none, "
        "no memory writes. Render it with `deploy/roman-mini/configure_core.py --dojo-teacher` "
        "(see docs/ai/plans/2026-09-29-lilith-c4-teacher-via-akh.md) and run this command again; "
        "never point the teacher arm at a working akh"
    )


class AcpTransportError(ModelError):
    """The ACP peer failed (connect, EOF, JSON-RPC error, timeout). The episode is not scored and is
    retried on resume, like a model endpoint failure."""

    code = "lilith::dojo::acp_transport"
    help = (
        "check that the Mac core's akhomed listens on the ACP endpoint, that the principal token file "
        "belongs to a principal granted the teacher akh, and that the akh is awake "
        "(`akh -w <akh> acp health`); re-run the same command to retry only the failed episodes"
    )


# --- line transport ----------------------------------------------------------------------------------


class LineChannel(Protocol):
    def send(self, message: Mapping[str, Any]) -> None: ...

    def recv(self, timeout: float) -> dict | None:
        """The next message; None at EOF. Raises `queue.Empty` on timeout."""

    def close(self) -> None: ...


class SocketLineChannel:
    """Newline-delimited JSON over a connected socket, read on a background thread."""

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._inbox: "queue.Queue[dict | None]" = queue.Queue()
        self._write = threading.Lock()
        self._reader = threading.Thread(target=self._read, name="acp-reader", daemon=True)
        self._reader.start()

    def _read(self) -> None:
        buffer = b""
        try:
            while True:
                chunk = self._sock.recv(65536)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        try:
                            decoded = json.loads(line)
                        except ValueError:
                            continue  # not JSON-RPC; the peer's problem, never ours to execute
                        if isinstance(decoded, dict):
                            self._inbox.put(decoded)
        except OSError:
            pass
        self._inbox.put(None)

    def send(self, message: Mapping[str, Any]) -> None:
        raw = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        with self._write:
            self._sock.sendall(raw)

    def recv(self, timeout: float) -> dict | None:
        return self._inbox.get(timeout=timeout)

    def close(self) -> None:
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()


def tcp_connector(endpoint: str, connect_timeout: float = 10.0) -> Callable[[], LineChannel]:
    """`tcp://127.0.0.1:18444` (akhomed's ACP listener). Loopback only: the bearer crosses the
    connection in plain text, exactly as `akh-acp --akhomed` refuses a remote endpoint."""
    url = urlparse(endpoint)
    if url.scheme != "tcp" or not url.hostname or url.port is None:
        raise DojoError(f"ACP endpoint {endpoint!r} is not tcp://host:port",
                        help="akhomed's ACP listener, e.g. tcp://127.0.0.1:18444")
    if url.hostname not in LOOPBACK_HOSTS:
        raise DojoError(f"ACP endpoint {endpoint!r} is not loopback",
                        help="akhomed's ACP TCP is plaintext and the replay gateway binds loopback; run the "
                             "teacher arm on the host that runs the teacher akh")
    host, port = url.hostname, url.port

    def connect() -> LineChannel:
        try:
            sock = socket.create_connection((host, port), timeout=connect_timeout)
        except OSError as error:
            raise AcpTransportError(f"cannot connect to the ACP endpoint {host}:{port}: {error.strerror or error}") from None
        sock.settimeout(None)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return SocketLineChannel(sock)

    return connect


# --- JSON-RPC client ----------------------------------------------------------------------------------


Handler = Callable[[str, Mapping[str, Any]], Any]


class AcpRpcError(AcpTransportError):
    def __init__(self, method: str, error: Mapping[str, Any]):
        data = error.get("data")
        detail = f" ({json.dumps(data, ensure_ascii=False)[:300]})" if data is not None else ""
        super().__init__(f"{method}: JSON-RPC error {error.get('code')}: {error.get('message')}{detail}")
        self.rpc_code = error.get("code")


class AcpClient:
    """Client side of one ACP connection. Requests from the agent (permission, fs, terminal) go to
    `on_request`; notifications to `on_notification`. Single-threaded: call from one thread."""

    def __init__(self, channel: LineChannel, secret: Secret, timeout_s: float):
        self.channel = channel
        self.secret = secret
        self.timeout_s = timeout_s
        self._next = 0

    def request(self, method: str, params: Mapping[str, Any], *, on_notification: Handler | None = None,
                on_request: Handler | None = None, timeout_s: float | None = None) -> Any:
        self._next += 1
        request_id = self._next
        self.channel.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params)})
        deadline = time.monotonic() + (timeout_s or self.timeout_s)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AcpTransportError(f"{method}: no answer within {timeout_s or self.timeout_s:.0f}s")
            try:
                message = self.channel.recv(remaining)
            except queue.Empty:
                continue
            if message is None:
                raise AcpTransportError(f"{method}: the ACP peer closed the connection")
            if "method" in message:
                self._incoming(message, on_notification, on_request)
                continue
            if message.get("id") != request_id:
                continue  # an answer to nothing we asked; ignore
            if "error" in message:
                raise AcpRpcError(method, message.get("error") or {})
            return message.get("result")

    def notify(self, method: str, params: Mapping[str, Any]) -> None:
        self.channel.send({"jsonrpc": "2.0", "method": method, "params": dict(params)})

    def _incoming(self, message: Mapping[str, Any], on_notification: Handler | None, on_request: Handler | None) -> None:
        method = str(message.get("method"))
        params = message.get("params") or {}
        if "id" not in message:
            if on_notification is not None:
                on_notification(method, params)
            return
        result: Any = None
        error: dict | None = None
        if on_request is None:
            error = {"code": -32601, "message": f"{method} is not offered by the dojo client"}
        else:
            try:
                result = on_request(method, params)
            except _Refuse as refusal:
                error = {"code": refusal.code, "message": str(refusal)}
        reply: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
        if error is not None:
            reply["error"] = error
        else:
            reply["result"] = result
        self.channel.send(reply)

    def scrub(self, text: str) -> str:
        return self.secret.scrub(text)


class _Refuse(Exception):
    def __init__(self, message: str, code: int = -32601):
        super().__init__(message)
        self.code = code


# --- the runner ---------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AcpTeacherConfig:
    akh: str  # the dojo teacher akh (akhomed workspace name)
    endpoint_host: str = "127.0.0.1"
    turn_timeout_s: float = 1800.0
    rpc_timeout_s: float = 60.0
    runner: RunnerConfig = RunnerConfig()
    # The library default is the self-contained placement, which needs nothing from the akh's
    # config; the CLI defaults to AKH_SYSTEM and verifies the akh first (`verify_prompt_parity`).
    placement: PromptPlacement = PromptPlacement.SESSION_TURN
    # Two throwaway tool-less prompts before the memory baseline (`AcpEpisodeRunner.warm_up`). The
    # CLI turns it on (`--no-warm-up` off); the library default leaves the akh untouched.
    warm_up: bool = False


# The echo akhomed sends after every tool call: `AkhMessage::ToolResult` becomes an agent thought
# chunk `[<tool>:ok|failed] <output>` (src/acp/mod.rs `to_session_update`). It is tool output, not
# model output, so it never marks the start of a new model request.
_TOOL_ECHO = re.compile(r"^\[[^\]\s:]+:(?:ok|failed)\] ")
_TERMINAL_STATUSES = frozenset({"completed", "failed"})


@dataclasses.dataclass
class RoundCounter:
    """Model turns that asked for tools, derived from the ACP stream (D-E in
    docs/ai/research/2026-09-30-glm-kimi-run-evaluation.md).

    The in-process arm's cap counts model turns. ACP has no round boundary of its own, and
    `_akh/usage` is not a live source (it answers `stale` with no usage while the agent is busy, and read 0
    in 7 of 9 smoke episodes: run evaluation D-H), so the live count is derived from the stream. akhomed runs one model request, then
    executes that round's tool calls one after the other (`src/agent/agentic_chat.rs`), so:

    - the calls of one round arrive back to back, with only tool output between them (their
      completion updates and the `[tool:ok]` echo thought);
    - a later model request streams thinking or answer text first. Any chunk that is not a tool echo,
      after the previous call started, therefore opens a new round.

    A model that emits no thinking and no text before its calls would merge two rounds: the live count
    is a lower bound, so it never cancels early. The exact figure is the akh's own `turns` counter
    (`_akh/usage`), read after the turn; the episode verdict uses that when the akh reports it.
    """

    limit: int
    rounds: int = 0
    _open: set[str] = dataclasses.field(default_factory=set)
    _fresh: bool = True  # model output arrived since the last call started

    def output(self, text: str, *, thought: bool) -> bool:
        """A message or thought chunk. Returns True when the akh has begun a model request beyond
        the cap (the previous round's calls all finished and the model is speaking again)."""
        if not text or (thought and _TOOL_ECHO.match(text)):
            return False
        self._fresh = True
        return self.rounds >= self.limit and not self._open

    def call_started(self, call_id: str) -> bool:
        """A `tool_call` update. Returns True when it belongs to a round beyond the cap."""
        if self._fresh or self.rounds == 0:
            self.rounds += 1
            self._fresh = False
        self._open.add(call_id)
        return self.rounds > self.limit

    def call_updated(self, call_id: str, status: object) -> None:
        if status in _TERMINAL_STATUSES:
            self._open.discard(call_id)


@dataclasses.dataclass
class _TurnState:
    """What arrived on the ACP stream during one `session/prompt`, in stream order."""

    segments: list[list[str]] = dataclasses.field(default_factory=lambda: [[]])  # message text between tool calls
    reasoning: list[str] = dataclasses.field(default_factory=list)
    tool_calls: list[dict] = dataclasses.field(default_factory=list)  # {"id", "name", "title"}
    violations: list[str] = dataclasses.field(default_factory=list)
    permissions: list[dict] = dataclasses.field(default_factory=list)
    capped: bool = False  # the runner cancelled the turn at its round cap
    counter: RoundCounter = dataclasses.field(default_factory=lambda: RoundCounter(limit=0))

    @property
    def answer(self) -> str:
        """Message text after the last tool call: the final message, which carries no tool call."""
        return "".join(self.segments[-1])


def tool_name_of(tool_call_id: str, known: frozenset[str] = frozenset()) -> str:
    """The tool a `toolCallId` names. akhomed's tracker ids are `<tool>_<counter>`; its permission
    requests use the bare tool name."""
    if tool_call_id in known:
        return tool_call_id
    match = _TOOL_ID.match(tool_call_id or "")
    return match.group("name") if match else (tool_call_id or "")


class AcpEpisodeRunner:
    """`EpisodeRunner` whose model is a sealed dojo teacher akh reached over ACP."""

    kind = KIND

    def __init__(self, connect: Callable[[], LineChannel], secret: Secret, system: SystemPrompt,
                 gateway: ReplayMcpGateway, config: AcpTeacherConfig, clock=time.monotonic):
        self.connect = connect
        self.secret = secret
        self.system = system
        self.gateway = gateway
        self.config = config
        self._clock = clock
        self.model_id = f"akh:{config.akh}"
        self.endpoint_host = config.endpoint_host
        self._memory_baseline: tuple[int, int] | None = None
        self.warm_up_report: dict | None = None  # what `probe` measured; provenance records it beside the teacher

    # -- EpisodeRunner -----------------------------------------------------------------------------

    def run(self, spec: EpisodeSpec, tools: Mapping[str, ReplayTools]) -> Episode:
        return self.run_traced(spec, tools).episode

    def run_traced(self, spec: EpisodeSpec, tools: Mapping[str, ReplayTools]) -> RunResult:
        log = receipt_log(tools)
        trace = RunTrace()
        steps: list[StepRecord] = []
        try:
            self._episode(spec, tools, trace, steps)
        except IsolationRefusal:
            raise
        except ModelError as error:
            trace.status = EpisodeStatus.ERROR
            trace.error = self.secret.scrub(str(error))
            trace.error_code = error.code
            trace.answer_status = "error"
        except DojoError:
            raise
        except Exception as error:  # one broken episode must not end the batch
            trace.status = EpisodeStatus.ERROR
            trace.error = self.secret.scrub(f"{type(error).__name__}: {error}")
            trace.error_code = "lilith::dojo::runner_internal"
            trace.answer_status = "error"
        return RunResult(Episode(spec, tuple(log.receipts), tuple(steps)), trace)

    # -- one episode --------------------------------------------------------------------------------

    def _episode(self, spec: EpisodeSpec, tools: Mapping[str, ReplayTools], trace: RunTrace,
                 steps: list[StepRecord]) -> None:
        if spec.family is Family.CORRECTION:
            raise ModelError("the model runner cannot replay a recorded correction yet (the episodes exist: --corrections loads them)",
                             help="select the family with --dry-run to size it, or skip it (--family names the ones to run)")
        broker = Broker(tools, self.config.runner.max_result_chars)
        allowed = frozenset(s.name for s in broker.tool_specs())
        now = _snapshot_now(tools)
        prompt = user_prompt(spec, now)
        trace.user_prompt_sha256 = sha256_hex(prompt.encode("utf-8"))
        trace.tools_sha256 = broker.tools_sha256
        trace.transcript.append({"role": "user", "content_sha256": trace.user_prompt_sha256,
                                 "delivery": "acp-session-prompt"})
        channel = self.connect()
        try:
            client = AcpClient(channel, self.secret, self.config.rpc_timeout_s)
            self._initialize(client)
            before = self._preflight(client)
            usage_before, turns_before = self._usage(client)
            with self.gateway.open(broker) as episode_tools:
                turn, started, turn_ms, stop = self._turn(client, episode_tools.acp_mcp_servers(),
                                                          self._prompt_blocks(prompt), allowed)
                usage_after, turns_after = self._usage(client)
                self._postflight(client, before)
                reported = (turns_after - turns_before) if turns_before is not None and turns_after is not None else None
                used = Usage(max(0, usage_after.prompt_tokens - usage_before.prompt_tokens),
                             max(0, usage_after.completion_tokens - usage_before.completion_tokens), 0)
                self._assemble(spec, episode_tools, turn, started, turn_ms, stop, used, reported, now, trace, steps)
        finally:
            channel.close()

    def probe(self) -> dict:
        """Check the teacher before a batch writes anything: ACP reachable, client-pushed HTTP MCP
        supported, no tool of its own, memory size. The result goes into the run's provenance (a
        resume against a teacher whose memory or daemon changed is refused)."""
        channel = self.connect()
        try:
            client = AcpClient(channel, self.secret, self.config.rpc_timeout_s)
            agent_info = self._initialize(client)
            if self.config.warm_up and self.warm_up_report is None:
                self.warm_up_report = self._warm_up(client)
            symbols, triples = self._preflight(client)
        finally:
            channel.close()
        return {"akh": self.config.akh, "agent_info": agent_info, "own_tools": 0,
                "memory": {"symbols": symbols, "triples": triples},
                "prompt_delivery": self.config.placement.value, "tool_transport": "client-pushed-http-mcp"}

    def _initialize(self, client: AcpClient) -> dict:
        result = client.request("initialize", {
            "protocolVersion": ACP_PROTOCOL_VERSION,
            # The dojo client offers the akh nothing of its own: no filesystem, no terminal.
            "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False},
            "clientInfo": {"name": "lilith-dojo-teacher", "version": "1"},
            "_meta": {"authorization": f"Bearer {self.secret.reveal()}"},
        }) or {}
        http = (((result.get("agentCapabilities") or {}).get("mcpCapabilities") or {}).get("http"))
        if http is not True:
            raise IsolationRefusal(
                "the ACP peer does not accept client-pushed HTTP MCP servers (agentCapabilities.mcpCapabilities.http "
                f"is {http!r}); the episode's replay tools cannot be attached, so the teacher could only use its own tools")
        info = result.get("agentInfo")
        return dict(info) if isinstance(info, Mapping) else {}

    # `_akh/tools` answers stale=True while the previous episode's turn is still settling
    # (cancellation, post-turn bookkeeping). Wait for the akh to go idle before verifying its
    # surface; any other answer (stale missing, or still stale at the deadline) is refused.
    IDLE_WAIT_S = 900.0   # a cancelled turn cannot interrupt an in-flight model request; GLM turns take minutes
    IDLE_POLL_S = 2.0

    def _list_own_tools(self, client: AcpClient) -> dict:
        deadline = time.monotonic() + self.IDLE_WAIT_S
        while True:
            listed = client.request("_akh/tools", {"workspace": self.config.akh}) or {}
            if listed.get("stale") is not True or time.monotonic() >= deadline:
                return listed
            time.sleep(self.IDLE_POLL_S)

    def _preflight(self, client: AcpClient) -> tuple[int, int]:
        listed = self._list_own_tools(client)
        if listed.get("stale") is not False:
            raise IsolationRefusal(
                f"akh {self.config.akh!r} could not list its tools (stale={listed.get('stale')!r}); a turn is in "
                "flight or the akh is not resident, so its tool surface cannot be verified")
        own = sorted(str(t.get("name")) for t in listed.get("tools") or [])
        if own:
            shown = ", ".join(own[:12]) + (f" (+{len(own) - 12} more)" if len(own) > 12 else "")
            raise IsolationRefusal(
                f"akh {self.config.akh!r} exposes {len(own)} tool(s) of its own besides the replay set: {shown}. "
                "A teacher that can reach anything but the recorded snapshot could read the future")
        counts = self._memory(client)
        if self._memory_baseline is None:
            self._memory_baseline = counts
        elif counts != self._memory_baseline:
            raise IsolationRefusal(
                f"akh {self.config.akh!r} memory changed between episodes (symbols, triples) "
                f"{self._memory_baseline} -> {counts}; an earlier episode's evidence could reach this one")
        return counts

    def _warm_up(self, client: AcpClient) -> dict:
        """Absorb a one-time initialisation, then prove it was one-time.

        The first prompt an akhomed process serves for a workspace can intern a few symbols (the
        2026-09-30 abort: 515 -> 517 symbols, 473 triples, on the first episode after a core restart;
        nine episodes on a long-lived process moved nothing). Two identical tool-less prompts run
        before the baseline is taken:

        - triples must not change on either, ever;
        - symbols may grow on the first, and only the first: the second must leave the memory exactly
          as the first left it, so anything that grew is a one-time init and not a per-turn write.

        The isolation check between and across episodes stays exact afterwards. The report names the
        interned symbols by label (`_akh/symbols/list`, labels only) so the init is evidence, not a
        guess."""
        before = self._memory(client)
        known = self._symbol_labels(client)
        counts = [before]
        for n in (1, 2):
            turn, _, _, stop = self._turn(client, [], [{"type": "text", "text": WARM_UP_PROMPT}], frozenset())
            if stop.get("stopReason") != "end_turn" or _AKH_ERROR.match(turn.answer):
                raise AcpTransportError(
                    f"the warm-up prompt {n} did not end normally (stopReason {stop.get('stopReason')!r}"
                    + (f": {turn.answer[:200]}" if turn.answer else "") + ")")
            counts.append(self._memory(client))
            if counts[-1][1] != counts[-2][1]:
                raise IsolationRefusal(
                    f"akh {self.config.akh!r} wrote a triple during warm-up prompt {n}: triples "
                    f"{counts[-2][1]} -> {counts[-1][1]}; a teacher may never write to its memory")
        if counts[2] != counts[1]:
            raise IsolationRefusal(
                f"akh {self.config.akh!r} memory kept changing across two identical warm-up prompts (symbols, "
                f"triples) {counts[1]} -> {counts[2]}: that is a per-turn write, not a one-time init, so an "
                "episode's evidence could reach the next one")
        labels = self._symbol_labels(client)
        interned = sorted(labels - known) if known is not None and labels is not None else None
        return {"prompt": WARM_UP_PROMPT, "prompts": 2, "before": {"symbols": before[0], "triples": before[1]},
                "after": {"symbols": counts[2][0], "triples": counts[2][1]},
                "interned_symbols": counts[1][0] - before[0],
                "interned_labels": None if interned is None else interned[:WARM_UP_LABELS_SHOWN]}

    def _symbol_labels(self, client: AcpClient) -> set[str] | None:
        """Labels of every symbol in the akh (`_akh/symbols/list`), or None when it is not offered."""
        try:
            total = self._memory(client)[0]
            listed = client.request("_akh/symbols/list", {"workspace": self.config.akh, "limit": total + 64}) or {}
        except (AcpRpcError, IsolationRefusal):
            return None
        symbols = listed.get("symbols")
        if not isinstance(symbols, list):
            return None
        return {str(s.get("label")) for s in symbols if isinstance(s, Mapping)}

    def _usage(self, client: AcpClient) -> tuple[Usage, int | None]:
        """The akh's cumulative model usage (`_akh/usage`; the prompt response carries none) and its
        cumulative model-turn count (`turns`: one per model request; None when the akh reports
        none). Episodes run one at a time, so the difference across one turn is that turn's."""
        listed = client.request("_akh/usage", {"workspace": self.config.akh}) or {}
        usage = listed.get("usage") or {}
        turns = usage.get("turns")
        return (Usage(_int(usage.get("prompt_tokens")), _int(usage.get("completion_tokens")), 0),
                turns if isinstance(turns, int) and not isinstance(turns, bool) else None)

    def _postflight(self, client: AcpClient, before: tuple[int, int]) -> None:
        after = self._memory(client)
        if after != before:
            raise IsolationRefusal(
                f"akh {self.config.akh!r} wrote to its memory during an episode (symbols, triples) {before} -> {after}; "
                "the next episode could recall it. Disable transcript capture and autonomy on the teacher akh")

    def _memory(self, client: AcpClient) -> tuple[int, int]:
        health = client.request("_akh/health", {"workspace": self.config.akh}) or {}
        symbols, triples = health.get("symbols"), health.get("triples")
        if not isinstance(symbols, int) or not isinstance(triples, int):
            raise IsolationRefusal(f"akh {self.config.akh!r} did not report its memory size (`_akh/health`)")
        return symbols, triples

    def _prompt_blocks(self, prompt: str) -> list[dict]:
        """The `session/prompt` content for one episode, by placement."""
        if self.config.placement is PromptPlacement.AKH_SYSTEM:
            return [{"type": "text", "text": prompt}]  # exactly the parent's user message
        return [{"type": "text", "text": PROMPT_PREAMBLE},
                {"type": "text", "text": "# System prompt\n\n" + self.system.text},
                {"type": "text", "text": "# Task\n\n" + prompt}]

    def _turn(self, client: AcpClient, mcp_servers: list, blocks: list[dict],
              allowed: frozenset[str]) -> tuple[_TurnState, float, int, Mapping[str, Any]]:
        created = client.request("session/new", {
            "cwd": f"akh-ws:{self.config.akh}",
            "mcpServers": mcp_servers,
            "_meta": {"akh/workspace": self.config.akh},
        }) or {}
        session_id = created.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise AcpTransportError("session/new returned no sessionId")
        state = _TurnState(counter=RoundCounter(limit=self.config.runner.max_rounds))

        def cap_turn() -> None:
            # The in-process loop's round cap, in model turns: the akh made (or is making) a
            # request past `max_rounds`, so the parent could not have read this far.
            if state.capped:
                return
            state.capped = True
            try:
                client.notify("session/cancel", {"sessionId": session_id})
            except OSError:
                pass

        def cancel(reason: str) -> None:
            state.violations.append(reason)
            try:
                client.notify("session/cancel", {"sessionId": session_id})
            except OSError:
                pass

        def on_notification(method: str, params: Mapping[str, Any]) -> None:
            if method != "session/update" or params.get("sessionId") not in (None, session_id):
                return
            update = params.get("update") or {}
            kind = update.get("sessionUpdate")
            if kind == "agent_message_chunk":
                text = _text(update.get("content"))
                state.segments[-1].append(text)
                if state.counter.output(text, thought=False):
                    cap_turn()
            elif kind == "agent_thought_chunk":
                text = _text(update.get("content"))
                state.reasoning.append(text)
                if state.counter.output(text, thought=True):
                    cap_turn()
            elif kind == "tool_call_update":
                state.counter.call_updated(str(update.get("toolCallId") or ""), update.get("status"))
            elif kind == "tool_call":
                call_id = str(update.get("toolCallId") or "")
                name = tool_name_of(call_id, allowed)
                state.tool_calls.append({"id": call_id, "name": name, "title": update.get("title")})
                state.segments.append([])
                over = state.counter.call_started(call_id)
                if name not in allowed:
                    cancel(f"the akh started tool {name!r}, which is not a replay tool")
                elif over:
                    cap_turn()

        def on_request(method: str, params: Mapping[str, Any]) -> Any:
            if method == "session/request_permission":
                call = params.get("toolCall") or {}
                name = tool_name_of(str(call.get("toolCallId") or ""), allowed)
                state.permissions.append({"tool": name, "title": call.get("title")})
                if name in allowed:
                    return {"outcome": {"outcome": "selected", "optionId": "allow"}}
                cancel(f"the akh asked to run {name!r}, which is not a replay tool")
                return {"outcome": {"outcome": "cancelled"}}
            raise _Refuse(f"{method} is not offered by the dojo client")

        started = self._clock()
        try:
            response = client.request("session/prompt", {"sessionId": session_id, "prompt": blocks},
                                      on_notification=on_notification, on_request=on_request,
                                      timeout_s=self.config.turn_timeout_s) or {}
        except AcpTransportError:
            try:  # stop the akh's turn; its tool calls already fail (the gateway closes with the episode)
                client.notify("session/cancel", {"sessionId": session_id})
            except OSError:
                pass
            raise
        turn_ms = int((self._clock() - started) * 1000)
        if state.violations:
            raise IsolationRefusal("; ".join(state.violations))
        return state, started, turn_ms, response

    def _assemble(self, spec: EpisodeSpec, episode_tools: EpisodeTools, turn: _TurnState, started: float,
                  turn_ms: int, stop: Mapping[str, Any], used: Usage, reported_turns: int | None, now: int,
                  trace: RunTrace, steps: list[StepRecord]) -> None:
        completion = used.completion_tokens
        trace.rounds, trace.rounds_source = model_turns(turn, reported_turns)
        trace.latency_ms = turn_ms
        trace.usage = used
        previous = started
        for n, call in enumerate(episode_tools.calls):
            dispatched = call.dispatched
            wall = max(0, int((call.at - previous) * 1000))
            previous = call.at
            if dispatched.problem is not None:
                trace.protocol_errors.append(dispatched.problem)
            if dispatched.declaration is not None:
                trace.declarations.append({"tool": f"mcp__{call.server}__{call.tool}", "receipt": dispatched.receipt_id,
                                           "declaration": dispatched.declaration})
            if dispatched.receipt_id is None or dispatched.server is None or dispatched.tool is None:
                continue
            why = "".join(turn.segments[n]) if n < len(turn.segments) else ""
            steps.append(StepRecord(
                spec.episode_id, len(steps),
                ToolCall(dispatched.tool, dispatched.arguments or {}, dispatched.receipt_id, dispatched.server),
                StepBudget(tokens=0, wall_ms=wall),
                why_this_move=(why[: self.config.runner.why_chars] or None)))
        trace.transcript.append({
            "role": "assistant", "content": turn.answer, "finish_reason": stop.get("stopReason"),
            "usage": trace.usage.to_dict(),
            "tool_calls": [{"id": c["id"], "name": c["name"]} for c in turn.tool_calls],
            "gateway_calls": [{"name": f"mcp__{c.server}__{c.tool}", "receipt": c.dispatched.receipt_id,
                               "arguments": dict(c.dispatched.arguments or {})} for c in episode_tools.calls],
            "segments": ["".join(s) for s in turn.segments],
            **({"reasoning": "".join(turn.reasoning)} if turn.reasoning else {}),
            **({"_meta": stop.get("_meta")} if stop.get("_meta") else {}),
        })
        reason = stop.get("stopReason")
        cap = self.config.runner.max_rounds
        if turn.capped:
            trace.answer_status = "round-cap"
            trace.answer_notes.append(f"cancelled beyond {cap} model turns (the round cap; {trace.rounds_source} count)")
            return
        if reason == "end_turn" and trace.rounds > cap:
            # The in-process loop gives the model `cap` turns including the answer. This akh needed
            # more (its own count, read after the turn), so the parent would have been cut off.
            trace.answer_status = "round-cap"
            trace.answer_notes.append(f"answered on model turn {trace.rounds} (the round cap is {cap}; {trace.rounds_source} count)")
            return
        if reason == "end_turn" and _AKH_ERROR.match(turn.answer) and not parse_answer(turn.answer).objects:
            # The akh reports a provider failure as an `error[<code>]: ...` message chunk and still
            # ends the turn: an endpoint failure, not an answer (not scored, retried on resume).
            raise AcpTransportError(f"the teacher akh reported an error instead of an answer: {turn.answer[:300]}")
        if reason == "end_turn":
            wall = max(0, int((self._clock() - previous) * 1000)) if episode_tools.calls else turn_ms
            # All completion tokens sit on the answer step: the budget score is cumulative, so the
            # answer step (whose weight is the episode weight) sees the same total as in-process.
            finish_answer(spec, turn.answer, completion, wall, now, trace, steps)
            return
        if reason == "max_turn_requests":
            trace.answer_status = "round-cap"
            return
        raise AcpTransportError(f"the teacher turn ended with stopReason {reason!r}"
                                + (f" ({json.dumps(stop.get('_meta'))[:200]})" if stop.get("_meta") else "")
                                + (f": {turn.answer[:300]}" if turn.answer else ""))


def model_turns(turn: _TurnState, reported: int | None) -> tuple[int, str]:
    """Model turns this ACP prompt took, and where the figure came from.

    `reported` is the difference of the akh's own `turns` counter across the prompt (exact: one per
    model request). It is used when present and not below what the stream showed; otherwise the
    figure is derived: the rounds that asked for tools, plus the answer turn unless the runner cut the
    turn off first."""
    observed = turn.counter.rounds
    if reported is not None and reported >= observed:
        return reported, "reported"
    return observed + (0 if turn.capped else 1), "derived"


def _text(content: Any) -> str:
    if isinstance(content, Mapping) and content.get("type") in (None, "text"):
        return str(content.get("text") or "")
    return ""


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def endpoint_host(endpoint: str) -> str:
    return urlparse(endpoint).hostname or endpoint


def read_teacher_config(path: Path, akh: str) -> dict:
    """The teacher's rendered workspace TOML (read-only): it must be the sealed teacher. Returns what
    the provenance records: the file hash and the backing model the akh pins."""
    import tomllib

    try:
        raw = Path(path).read_bytes()
        doc = tomllib.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as error:
        raise IsolationRefusal(f"cannot read the teacher config {path}: {error}") from None
    problems = []
    if doc.get("name") != akh:
        problems.append(f"name is {doc.get('name')!r}, not {akh!r}")
    if doc.get("tool_surface") != "session-only":
        problems.append(f"tool_surface is {doc.get('tool_surface')!r}, not 'session-only'")
    if doc.get("capture_transcripts") is not False:
        problems.append("capture_transcripts is not false")
    if (doc.get("mcp") or {}).get("servers"):
        problems.append("[[mcp.servers]] is set")
    if problems:
        raise IsolationRefusal(f"{path} is not a sealed dojo teacher: " + "; ".join(problems))
    prompt = doc.get("system_prompt")
    return {"path": str(path), "sha256": sha256_hex(raw), "backend": doc.get("backend"), "model": doc.get("model"),
            "tool_surface": doc.get("tool_surface"),
            "system_prompt_sha256": sha256_hex(prompt.encode("utf-8")) if isinstance(prompt, str) else None}


def verify_prompt_parity(declared: Mapping[str, Any] | None, system: SystemPrompt) -> None:
    """Refuse a run whose teacher akh would not receive the run's dojo prompt as its system prompt.

    Checked on the rendered workspace TOML (`read_teacher_config`): its `system_prompt` must hash to
    the run's `system_prompt_sha256`. What this cannot see is whether akhomed has reloaded that file
    (no `_akh/*` read exposes the active prompt), so the refusal help says to restart the core after
    a render. akhomed still composes the text as the "Operator Instructions" section of its own
    system message (psyche identity before it; tool-operation and convention sections after, not
    suppressible per akh): recorded in the plan, not in the hash."""
    if declared is None:
        raise PromptParityRefusal("the teacher's rendered workspace TOML was not given (--akh-config), so its "
                                  "system prompt cannot be compared with the run's")
    actual = declared.get("system_prompt_sha256")
    if actual != system.sha256:
        raise PromptParityRefusal(
            f"the teacher akh's system_prompt is {'not set' if actual is None else 'sha256 ' + actual[:16]}, "
            f"the run's dojo system prompt is sha256 {system.sha256[:16]} ({declared.get('path')})")
