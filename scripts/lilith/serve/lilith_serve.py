#!/usr/bin/env python3
"""Mac-side supervisor for Lilith's Runpod serve pod (Anima WI 01a0e270-314e, Lilith C2 Stage 0).

ADR 116 Decision 7 and the local GPU qualification plan §1 make Runpod serving a *test*
surface with hard limits: at most 2 h per session, stop after 20 min without requests,
at most 6 GPU-hours per week, and at most 25% of the remaining credit per month (training
has priority on credit). This CLI owns those limits.

    lilith_serve.py create --yes [--key-source env|runpod-secret] [--kv-cache-dtype auto]
    lilith_serve.py install-payload <lilith-payload-main-<sha12>.tar.gz> [--launch]
    lilith_serve.py start [--wait-seconds 1800] [--watch]
    lilith_serve.py stop [--reason manual]
    lilith_serve.py status [--json]
    lilith_serve.py idle-watch [--interval 60] [--daemon] [--once]
    lilith_serve.py ledger [--json] [--offline]
    lilith_serve.py recreate --yes --confirm-terminate <old pod id>
    lilith_serve.py sync-endpoint

The pod ``akh-medu-lilith-serve`` is created once (EU-FR-1, 1x H100 80GB, network volume
``c6kdciozex``) and afterwards only stopped and started: a stopped pod keeps its id, so
``https://<podId>-8000.proxy.runpod.net`` stays the same (docs.runpod.io/pods/configuration/
expose-ports). A start can come up with zero GPUs when the pod's machine is full
(docs.runpod.io/pods/troubleshooting/zero-gpus); ``start`` detects that, stops the pod again
and names the options. Only ``recreate`` changes the pod id and therefore the URL; it
updates the 1Password endpoint item and says so loudly.

Secrets
    The Runpod API key is read with ``op-service-account read`` at call time and lives in
    memory only. The serve API key is generated once by ``create`` and stored in 1Password
    item ``lilith-serve-api-key`` (vault Homelab, field ``credential``); it reaches the op
    CLI on stdin and the pod through the Runpod create body, never argv, disk or logs. Every
    log line passes a redactor that knows each secret this process has seen.

Local state (non-secret) lives in ``~/lilith-private/serve/`` (``LILITH_SERVE_STATE_DIR``):
``state.json`` (pod id, base URL, the open session), ``ledger.jsonl`` (session start/stop
events: the GPU-hour and spend ledger) and ``logs/supervisor.log``.

Readiness (``start`` and ``wait-ready``) is /health plus one OpenAI chat request carrying a
trivial ``get_time`` tool: a well-formed ``tool_calls`` entry records ``tool_calling: verified``;
a plain-text answer records ``unverified`` with the raw finish reason; a refused request, a
malformed call or native call syntax left unparsed in ``content`` is a failure (exit 6; under
``start`` the pod is stopped). The record lands in ``state.json`` and ``readiness/<session>.json``.

Exit codes: 0 ok, 1 error, 2 usage/confirmation, 3 budget refusal, 4 zero GPUs on start,
5 vLLM not healthy in time, 6 tool calling broken.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import enum
import json
import os
import re
import secrets as secrets_module
import shlex
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

# ----------------------------------------------------------------- constants
POD_NAME = "akh-medu-lilith-serve"
DATACENTER = "EU-FR-1"
NETWORK_VOLUME_ID = "c6kdciozex"
GPU_TYPE = "NVIDIA H100 80GB HBM3"
IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
ALLOWED_CUDA = ("13.0",)
SERVE_PORT = 8000
SERVED_MODEL = "lilith"
DEFAULT_RATE_USD_PER_HR = 3.49  # 1x H100 80GB, EU-FR-1, 2026-09-27; the pod's own costPerHr wins
VAULT = "Homelab"
RUNPOD_KEY_REF = "op://Homelab/Runpod API Key/credential"
API_KEY_ITEM = "lilith-serve-api-key"
SERVE_KEY_REF = f"op://{VAULT}/{API_KEY_ITEM}/credential"
ENDPOINT_ITEM = "lilith-serve-endpoint"
RUNPOD_SECRET_NAME = "lilith_serve_api_key"
USER_AGENT = "curl/8.7.1"  # Cloudflare blocks Python's default User-Agent
POD_ROOT = "/workspace/lilith"
SERVE_ENTRY = f"{POD_ROOT}/serve/current/scripts/lilith/serve/serve_pod.sh"
SERVE_LOG = f"{POD_ROOT}/serve/logs/serve_pod.log"
# The pod's start command: launch serve_pod.sh from the installed payload (if any), then hand
# the container to the image's own /start.sh (sshd), so ssh keeps working for payload installs.
START_COMMAND = (
    "bash", "-c",
    f"mkdir -p {POD_ROOT}/serve/logs; S={SERVE_ENTRY}; "
    f'if [ -f "$S" ]; then nohup bash "$S" >> {SERVE_LOG} 2>&1 < /dev/null & fi; '
    "if [ -x /start.sh ]; then exec /start.sh; else exec sleep infinity; fi",
)
KV_CACHE_DTYPES = ("auto", "fp8", "fp8_e4m3", "fp8_e5m2")
# Prometheus series whose change means "vLLM served something" (any one suffices).
ACTIVITY_COUNTERS = ("vllm:request_success_total", "vllm:e2e_request_latency_seconds_count",
                     "vllm:prompt_tokens_total", "vllm:generation_tokens_total")
IN_FLIGHT_GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")


@dataclasses.dataclass(frozen=True)
class Policy:
    """Serving policy (local GPU qualification plan §1; operator decision 2026-09-27)."""

    session_cap_s: int = 2 * 3600
    idle_s: int = 20 * 60
    week_gpu_hours: float = 6.0
    month_credit_share: float = 0.25
    reserve_usd: float = 5.0  # never start into the last few dollars: training has priority
    min_session_s: int = 10 * 60  # refuse a start that could not last this long
    interval_s: int = 60
    pod_check_s: int = 300  # REST pod-status check cadence inside a session
    idle_pod_check_s: int = 900  # daemon cadence without a session (adopts a console start)


# ----------------------------------------------------------------- errors
class ServeError(Exception):
    """Python counterpart of the repo's miette pattern: stable code, message, help options."""

    code = "lilith::serve"
    help = ""
    exit_code = 1

    def __init__(self, message: str, *, help: str | None = None):
        super().__init__(message)
        if help is not None:
            self.help = help

    def render(self) -> str:
        text = f"error[{self.code}]: {self}"
        return text + (f"\n  help: {self.help}" if self.help else "")


class ConfirmationRequired(ServeError):
    code = "lilith::serve::confirm"
    exit_code = 2


class UsageError(ServeError):
    code = "lilith::serve::usage"
    exit_code = 2


class BudgetRefused(ServeError):
    code = "lilith::serve::budget"
    exit_code = 3


class ZeroGpu(ServeError):
    code = "lilith::serve::zero_gpu"
    exit_code = 4


class HealthTimeout(ServeError):
    code = "lilith::serve::health_timeout"
    exit_code = 5


class NotCreated(ServeError):
    code = "lilith::serve::not_created"
    help = "run `lilith_serve.py create --yes` once (operator decision: it creates a billable pod)"


class AlreadyCreated(ServeError):
    code = "lilith::serve::already_created"


class SecretStoreError(ServeError):
    code = "lilith::serve::secret_store"


class RunpodApiError(ServeError):
    code = "lilith::serve::runpod_api"

    def __init__(self, message: str, *, status: int | None = None, help: str | None = None):
        super().__init__(message, help=help)
        self.status = status


class ProbeError(ServeError):
    code = "lilith::serve::probe"


class ProvenanceError(ServeError):
    code = "lilith::serve::provenance"


class ToolCallingBroken(ServeError):
    code = "lilith::serve::tool_calling"
    exit_code = 6
    help = ("check session.json `tool_calling` on the pod (parser, chat template) and the vLLM log; "
            "LILITH_SERVE_TOOL_PARSER / LILITH_SERVE_CHAT_TEMPLATE in the pod env select another pair")


# ----------------------------------------------------------------- redaction + logging
class Redactor:
    """Every secret this process has touched; logs and errors pass through `scrub`."""

    def __init__(self) -> None:
        self._secrets: set[str] = set()

    def add(self, value: str) -> str:
        if value and len(value) >= 8:
            self._secrets.add(value)
        return value

    def scrub(self, text: str) -> str:
        for value in sorted(self._secrets, key=len, reverse=True):
            text = text.replace(value, "[redacted]")
        return text


REDACTOR = Redactor()


class Log:
    def __init__(self, directory: Path | None, echo: bool = True):
        self.path = None if directory is None else directory / "logs" / "supervisor.log"
        self.echo = echo

    def __call__(self, message: str) -> None:
        line = f"{utc_now().strftime('%Y-%m-%dT%H:%M:%SZ')} lilith-serve: {REDACTOR.scrub(message)}"
        if self.echo:
            print(line, file=sys.stderr, flush=True)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a") as handle:
                handle.write(line + "\n")


# ----------------------------------------------------------------- time
def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso(moment: dt.datetime) -> str:
    return moment.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso(text: str) -> dt.datetime:
    moment = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=dt.timezone.utc)


class Clock(Protocol):
    def now(self) -> dt.datetime: ...
    def sleep(self, seconds: float) -> None: ...


class RealClock:
    def now(self) -> dt.datetime:
        return utc_now()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def week_start(moment: dt.datetime) -> dt.datetime:
    day = moment.astimezone(dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return day - dt.timedelta(days=day.weekday())


def month_start(moment: dt.datetime) -> dt.datetime:
    return moment.astimezone(dt.timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def next_month(moment: dt.datetime) -> dt.datetime:
    start = month_start(moment)
    return (start + dt.timedelta(days=32)).replace(day=1)


# ----------------------------------------------------------------- 1Password
class SecretStore(Protocol):
    def read(self, reference: str) -> str: ...
    def item_fields(self, title: str) -> dict[str, str] | None: ...
    def create_api_credential(self, title: str, value: str) -> None: ...
    def upsert_endpoint(self, pod_id: str, base_url: str) -> None: ...


class OnePassword:
    """`op-service-account` (the op CLI with the machine-local service account).

    Secret values travel on stdout (read) or stdin (create) only, never argv."""

    def __init__(self, binary: str):
        self.binary = binary

    def _run(self, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run([self.binary, *args], input=stdin, capture_output=True, text=True,
                                  check=False, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SecretStoreError(f"cannot run {self.binary}: {type(error).__name__}",
                                   help="check ~/.local/bin/op-service-account and its service-account token") from error

    def read(self, reference: str) -> str:
        result = self._run(["read", "--no-newline", reference])
        value = result.stdout.strip()
        if result.returncode != 0 or not value:
            raise SecretStoreError(f"1Password read of {reference} failed (exit {result.returncode})",
                                   help="check the item exists in vault Homelab and the service account can read it")
        return REDACTOR.add(value)

    def item_fields(self, title: str) -> dict[str, str] | None:
        result = self._run(["item", "get", title, "--vault", VAULT, "--format", "json"])
        if result.returncode != 0:
            return None
        try:
            item = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise SecretStoreError(f"1Password item {title} returned invalid JSON") from error
        fields: dict[str, str] = {}
        for field in item.get("fields", []):
            if field.get("type") == "CONCEALED":
                continue  # never pull a secret into a metadata read
            label = field.get("label") or field.get("id")
            if label and field.get("value") is not None:
                fields[str(label)] = str(field["value"])
        return fields

    def create_api_credential(self, title: str, value: str) -> None:
        template = {"title": title, "category": "API_CREDENTIAL",
                    "fields": [{"id": "credential", "label": "credential", "type": "CONCEALED", "value": value},
                               {"id": "notesPlain", "label": "notesPlain", "type": "STRING", "purpose": "NOTES",
                                "value": "vLLM API key of the Runpod pod akh-medu-lilith-serve "
                                         "(lilith_serve.py create; Anima WI 01a0e270-314e)"}]}
        result = self._run(["item", "create", "--vault", VAULT, "--format", "json"], stdin=json.dumps(template))
        if result.returncode != 0:
            raise SecretStoreError(f"1Password item create {title} failed (exit {result.returncode}): "
                                   f"{REDACTOR.scrub(result.stderr.strip())[:300]}",
                                   help=f"create an API Credential item '{title}' in vault {VAULT} by hand with a "
                                        "random 64-character credential; create reuses an existing item")

    def upsert_endpoint(self, pod_id: str, base_url: str) -> None:
        assignments = [f"pod_id[text]={pod_id}", f"base_url[text]={base_url}"]
        if self.item_fields(ENDPOINT_ITEM) is None:
            args = ["item", "create", "--vault", VAULT, "--category", "Secure Note", "--title", ENDPOINT_ITEM,
                    *assignments]
        else:
            args = ["item", "edit", ENDPOINT_ITEM, "--vault", VAULT, *assignments]
        result = self._run(args)
        if result.returncode != 0:
            raise SecretStoreError(f"1Password endpoint update failed (exit {result.returncode})",
                                   help="rerun `lilith_serve.py sync-endpoint` once 1Password is reachable; "
                                        f"local state already records pod {pod_id}")


# ----------------------------------------------------------------- HTTP
def http(method: str, url: str, *, headers: Mapping[str, str] | None = None, body: Any = None,
         timeout: float = 30) -> tuple[int, bytes]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("User-Agent", USER_AGENT)
    request.add_header("Accept", "application/json")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        with error:
            return error.code, error.read()


class KeyProvider:
    """Reads the Runpod key per use (daemon) or once per process (CLI), memory only."""

    def __init__(self, store: SecretStore, reference: str, cache: bool):
        self.store, self.reference, self.cache = store, reference, cache
        self._value: str | None = None

    def __call__(self) -> str:
        if self._value is not None:
            return self._value
        value = self.store.read(self.reference)
        if self.cache:
            self._value = value
        return value


@dataclasses.dataclass(frozen=True)
class PodInfo:
    id: str
    desired_status: str
    gpu_count: int | None
    rate_usd_per_hr: float | None
    public_ip: str | None
    port_mappings: Mapping[str, int]
    last_started_at: str | None

    @classmethod
    def from_api(cls, data: Mapping[str, Any]) -> "PodInfo":
        gpu = data.get("gpu") if isinstance(data.get("gpu"), Mapping) else None
        count = data.get("gpuCount")
        if gpu is not None and gpu.get("count") is not None:
            count = gpu.get("count")
        rate = data.get("adjustedCostPerHr") or data.get("costPerHr")
        return cls(id=str(data["id"]), desired_status=str(data.get("desiredStatus", "UNKNOWN")),
                   gpu_count=None if count is None else int(count),
                   rate_usd_per_hr=None if rate in (None, "", 0) else float(rate),
                   public_ip=data.get("publicIp") or None,
                   port_mappings={str(k): int(v) for k, v in (data.get("portMappings") or {}).items()},
                   last_started_at=data.get("lastStartedAt"))


class Runpod:
    def __init__(self, key: Callable[[], str], rest: str, graphql: str):
        self.key, self.rest, self.graphql = key, rest.rstrip("/"), graphql

    def _call(self, method: str, url: str, body: Any = None) -> Any:
        status, raw = http(method, url, headers={"Authorization": f"Bearer {self.key()}"}, body=body, timeout=60)
        if not 200 <= status < 300:
            detail = REDACTOR.scrub(raw.decode(errors="replace"))[:400]
            raise RunpodApiError(f"{method} {url} -> HTTP {status}: {detail}", status=status,
                                 help="check the Runpod console; 401 = key, 400/404 = pod id, 5xx = retry later")
        return json.loads(raw) if raw.strip() else None

    def get_pod(self, pod_id: str) -> PodInfo:
        return PodInfo.from_api(self._call("GET", f"{self.rest}/pods/{pod_id}"))

    def create_pod(self, body: Mapping[str, Any]) -> PodInfo:
        return PodInfo.from_api(self._call("POST", f"{self.rest}/pods", body))

    def start_pod(self, pod_id: str) -> None:
        self._call("POST", f"{self.rest}/pods/{pod_id}/start")

    def stop_pod(self, pod_id: str) -> None:
        self._call("POST", f"{self.rest}/pods/{pod_id}/stop")

    def delete_pod(self, pod_id: str) -> None:
        self._call("DELETE", f"{self.rest}/pods/{pod_id}")

    def balance_usd(self) -> float:
        query = {"query": "query { myself { clientBalance currentSpendPerHr spendLimit } }"}
        data = self._call("POST", self.graphql, query)
        try:
            return float(data["data"]["myself"]["clientBalance"])
        except (TypeError, KeyError, ValueError) as error:
            raise RunpodApiError("Runpod GraphQL returned no clientBalance",
                                 help="the budget check fails closed without a balance; retry, or read it "
                                      "in the Runpod console and decide by hand") from error


# ----------------------------------------------------------------- vLLM probe
@dataclasses.dataclass(frozen=True)
class Metrics:
    activity: tuple[float, ...]
    in_flight: float


def parse_metrics(text: str) -> Metrics:
    totals: dict[str, float] = {}
    seen = False
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)", line)
        if not match:
            continue
        name = match.group(1)
        if name in ACTIVITY_COUNTERS or name in IN_FLIGHT_GAUGES:
            seen = True
            try:
                totals[name] = totals.get(name, 0.0) + float(match.group(3))
            except ValueError:
                continue
    if not seen:
        raise ProbeError("no vLLM request metrics in /metrics",
                         help="the proxy may be serving something else, or vLLM renamed its series")
    return Metrics(activity=tuple(totals.get(name, 0.0) for name in ACTIVITY_COUNTERS),
                   in_flight=sum(totals.get(name, 0.0) for name in IN_FLIGHT_GAUGES))


class Vllm:
    def __init__(self, base_url: str, api_key: Callable[[], str | None] = lambda: None):
        self.base_url, self.api_key = base_url.rstrip("/"), api_key

    def _headers(self) -> dict[str, str]:
        key = self.api_key()
        return {} if not key else {"Authorization": f"Bearer {key}"}

    def healthy(self) -> bool:
        try:
            status, _ = http("GET", f"{self.base_url}/health", timeout=15)
        except (urllib.error.URLError, OSError, TimeoutError):
            return False
        return status == 200

    def metrics(self) -> Metrics:
        try:
            status, raw = http("GET", f"{self.base_url}/metrics", headers=self._headers(), timeout=20)
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            raise ProbeError(f"/metrics unreachable: {type(error).__name__}") from error
        if status != 200:
            raise ProbeError(f"/metrics -> HTTP {status}")
        return parse_metrics(raw.decode(errors="replace"))

    def chat(self, request: Mapping[str, Any]) -> tuple[int, Any]:
        try:
            status, raw = http("POST", f"{self.base_url}/v1/chat/completions", headers=self._headers(),
                               body=request, timeout=180)
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            raise ToolCallingBroken(f"/v1/chat/completions unreachable: {type(error).__name__}") from error
        try:
            return status, json.loads(raw)
        except json.JSONDecodeError:
            return status, REDACTOR.scrub(raw.decode(errors="replace"))[:400]


# ----------------------------------------------------------------- tool-calling readiness
PROBE_TOOL_NAME = "get_time"
PROBE_TOOL = {"type": "function", "function": {
    "name": PROBE_TOOL_NAME, "description": "Return the current time as an ISO-8601 string.",
    "parameters": {"type": "object", "properties": {"timezone": {"type": "string",
                                                                 "description": "IANA timezone, e.g. UTC"}}}}}
# Gemma 4's native call syntax; seeing it in `content` means vLLM did not parse the call.
NATIVE_CALL_MARKERS = ("<|tool_call>", "<tool_call|>")


@dataclasses.dataclass(frozen=True)
class ToolProbe:
    """Readiness evidence: `verified` (a well-formed tool call came back) or `unverified`
    (the model answered in plain text; the raw finish reason is kept)."""

    status: str
    finish_reason: str | None
    tool_calls: int
    detail: str
    checked_at: str

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "lilith/serve-tool-probe@1"} | dataclasses.asdict(self)


def probe_request() -> dict[str, Any]:
    return {"model": SERVED_MODEL, "tools": [PROBE_TOOL], "tool_choice": "auto", "temperature": 0,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": f"Call the {PROBE_TOOL_NAME} tool for UTC."}]}


def check_tool_calls(status: int, response: Any, now: dt.datetime) -> ToolProbe:
    """Pure check of one chat completion. Raises ToolCallingBroken when the server refuses the
    tools request or returns a malformed / unparsed call; plain text is `unverified`."""
    if status != 200 or not isinstance(response, Mapping):
        raise ToolCallingBroken(f"tools request -> HTTP {status}: {str(response)[:300]}")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ToolCallingBroken("tools request returned no choices")
    choice = choices[0]
    finish = choice.get("finish_reason")
    message = choice.get("message") if isinstance(choice.get("message"), Mapping) else {}
    calls = message.get("tool_calls") or []
    content = message.get("content") or ""
    if not calls:
        if any(marker in content for marker in NATIVE_CALL_MARKERS):
            raise ToolCallingBroken("the model emitted a native tool call that vLLM left unparsed in `content` "
                                    "(tool-call parser and chat template do not match)")
        return ToolProbe("unverified", finish, 0, "the model answered in plain text; no tool_calls", iso(now))
    problems = []
    for index, call in enumerate(calls if isinstance(calls, list) else [calls]):
        function = call.get("function") if isinstance(call, Mapping) else None
        if not isinstance(call, Mapping) or not call.get("id") or call.get("type") != "function":
            problems.append(f"call {index}: missing id or type != function")
        if not isinstance(function, Mapping) or function.get("name") != PROBE_TOOL_NAME:
            problems.append(f"call {index}: function name is not {PROBE_TOOL_NAME}")
            continue
        try:
            arguments = json.loads(function.get("arguments") or "")
        except (TypeError, json.JSONDecodeError):
            arguments = None
        if not isinstance(arguments, dict):
            problems.append(f"call {index}: arguments are not a JSON object")
    if problems:
        raise ToolCallingBroken("malformed tool_calls: " + "; ".join(problems))
    return ToolProbe("verified", finish, len(calls), f"{len(calls)} well-formed {PROBE_TOOL_NAME} call(s)", iso(now))


# ----------------------------------------------------------------- ledger + budget
@dataclasses.dataclass(frozen=True)
class Span:
    session: str
    pod_id: str
    start: dt.datetime
    stop: dt.datetime | None
    rate: float
    reason: str | None

    def overlap_seconds(self, window_start: dt.datetime, now: dt.datetime) -> float:
        end = self.stop or now
        return max(0.0, (min(end, now) - max(self.start, window_start)).total_seconds())


@dataclasses.dataclass(frozen=True)
class Usage:
    week_hours: float
    month_hours: float
    month_spend_usd: float
    total_hours: float
    total_spend_usd: float
    open_session: str | None

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in dataclasses.asdict(self).items()}


class Ledger:
    """Append-only JSONL of session start/stop events (serving only; training is separate)."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, event: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as handle:
            handle.write(json.dumps(dict(event), sort_keys=True) + "\n")

    def start(self, session: str, pod_id: str, at: dt.datetime, rate: float, cap_s: int, source: str) -> None:
        self.append({"event": "session_start", "session": session, "pod_id": pod_id, "at": iso(at),
                     "rate_usd_per_hr": rate, "cap_seconds": cap_s, "source": source})

    def stop(self, session: str, pod_id: str, at: dt.datetime, reason: str) -> None:
        self.append({"event": "session_stop", "session": session, "pod_id": pod_id, "at": iso(at),
                     "reason": reason})

    def spans(self) -> list[Span]:
        if not self.path.exists():
            return []
        starts: dict[str, dict[str, Any]] = {}
        stops: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "session_start":
                starts[event["session"]] = event
                order.append(event["session"])
            elif event.get("event") == "session_stop":
                stops.setdefault(event["session"], event)
        spans = []
        for session in order:
            begin, end = starts[session], stops.get(session)
            spans.append(Span(session=session, pod_id=begin["pod_id"], start=parse_iso(begin["at"]),
                              stop=None if end is None else parse_iso(end["at"]),
                              rate=float(begin.get("rate_usd_per_hr") or DEFAULT_RATE_USD_PER_HR),
                              reason=None if end is None else end.get("reason")))
        return spans

    def usage(self, now: dt.datetime) -> Usage:
        spans = self.spans()
        epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
        week, month = week_start(now), month_start(now)
        return Usage(
            week_hours=sum(s.overlap_seconds(week, now) for s in spans) / 3600,
            month_hours=sum(s.overlap_seconds(month, now) for s in spans) / 3600,
            month_spend_usd=sum(s.overlap_seconds(month, now) * s.rate for s in spans) / 3600,
            total_hours=sum(s.overlap_seconds(epoch, now) for s in spans) / 3600,
            total_spend_usd=sum(s.overlap_seconds(epoch, now) * s.rate for s in spans) / 3600,
            open_session=next((s.session for s in reversed(spans) if s.stop is None), None))


@dataclasses.dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    cap_seconds: int
    limits_seconds: Mapping[str, int]
    binding: str
    balance_usd: float
    rate_usd_per_hr: float
    month_allowance_usd: float
    refusal: str | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {"limits_seconds": dict(self.limits_seconds)}


def evaluate_budget(policy: Policy, usage: Usage, balance_usd: float, rate: float, now: dt.datetime) -> BudgetDecision:
    """The session cap is the tightest of four limits; below `min_session_s` the start is refused.

    Monthly credit share: serving may spend at most `month_credit_share` of the credit that was
    available to it this month, i.e. of (current balance + serving spend so far this month).
    Top-ups raise the allowance; training spend lowers it (training has priority on credit)."""
    allowance = policy.month_credit_share * (balance_usd + usage.month_spend_usd)
    per_second = rate / 3600
    limits = {
        "session_cap": policy.session_cap_s,
        "weekly_gpu_hours": int((policy.week_gpu_hours - usage.week_hours) * 3600),
        "monthly_credit_share": int((allowance - usage.month_spend_usd) / per_second),
        "credit_reserve": int((balance_usd - policy.reserve_usd) / per_second),
    }
    binding = min(limits, key=lambda name: limits[name])
    cap = max(0, min(limits.values()))
    refusal = None
    if cap < policy.min_session_s:
        resets = {
            "weekly_gpu_hours": f"the weekly allowance resets {iso(week_start(now) + dt.timedelta(days=7))}",
            "monthly_credit_share": f"the monthly allowance resets {iso(next_month(now))}, or after a credit top-up",
            "credit_reserve": "the operator tops up the Runpod credit (the supervisor never adds funds)",
            "session_cap": "lower min_session_s",
        }[binding]
        refusal = (f"serving budget exhausted: {binding} leaves {max(0, limits[binding]) // 60} min "
                   f"(< {policy.min_session_s // 60} min minimum). Used this week {usage.week_hours:.2f} of "
                   f"{policy.week_gpu_hours:g} GPU-h; this month ${usage.month_spend_usd:.2f} of "
                   f"${allowance:.2f} ({policy.month_credit_share:.0%} of credit); balance ${balance_usd:.2f}. "
                   f"Next: {resets}.")
    return BudgetDecision(allowed=refusal is None, cap_seconds=cap, limits_seconds=limits, binding=binding,
                          balance_usd=balance_usd, rate_usd_per_hr=rate, month_allowance_usd=round(allowance, 4),
                          refusal=refusal)


# ----------------------------------------------------------------- state
@dataclasses.dataclass
class Session:
    id: str
    started_at: str
    cap_seconds: int
    rate_usd_per_hr: float
    source: str
    ready_at: str | None = None
    last_activity_at: str | None = None
    last_activity: list[float] | None = None
    metrics_misses: int = 0
    last_pod_check_at: str | None = None
    tool_calling: dict[str, Any] | None = None


@dataclasses.dataclass
class State:
    pod_id: str | None = None
    base_url: str | None = None
    created_at: str | None = None
    session: Session | None = None
    last_idle_pod_check_at: str | None = None

    @classmethod
    def load(cls, path: Path) -> "State":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        session = data.get("session")
        return cls(pod_id=data.get("pod_id"), base_url=data.get("base_url"), created_at=data.get("created_at"),
                   session=None if session is None else Session(**session),
                   last_idle_pod_check_at=data.get("last_idle_pod_check_at"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = path.with_name(f".{path.name}.{os.getpid()}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(dataclasses.asdict(self), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)


def proxy_url(pod_id: str, port: int = SERVE_PORT, template: str | None = None) -> str:
    """The Runpod HTTP proxy URL: https://<podId>-<port>.proxy.runpod.net (stable across stop/start)."""
    if not re.fullmatch(r"[a-z0-9]+", pod_id):
        raise UsageError(f"pod id {pod_id!r} is not a Runpod pod id")
    return (template or "https://{pod_id}-{port}.proxy.runpod.net").format(pod_id=pod_id, port=port)


# ----------------------------------------------------------------- the supervisor
@dataclasses.dataclass
class Context:
    state_dir: Path
    store: SecretStore
    runpod: Runpod
    policy: Policy
    clock: Clock
    log: Log
    proxy_template: str | None = None
    repo: Path = Path(__file__).resolve().parents[3]

    @property
    def state_path(self) -> Path:
        return self.state_dir / "state.json"

    @property
    def ledger(self) -> Ledger:
        return Ledger(self.state_dir / "ledger.jsonl")

    def state(self) -> State:
        return State.load(self.state_path)

    def serve_key(self) -> str:
        return self.store.read(SERVE_KEY_REF)


def pod_body(api_key_env: str, kv_cache_dtype: str) -> dict[str, Any]:
    return {
        "name": POD_NAME, "imageName": IMAGE, "cloudType": "SECURE", "computeType": "GPU",
        "gpuTypeIds": [GPU_TYPE], "gpuCount": 1, "dataCenterIds": [DATACENTER],
        "networkVolumeId": NETWORK_VOLUME_ID, "volumeMountPath": "/workspace",
        "allowedCudaVersions": list(ALLOWED_CUDA), "containerDiskInGb": 30, "interruptible": False,
        "ports": [f"{SERVE_PORT}/http", "22/tcp"], "supportPublicIp": True,
        "dockerStartCmd": list(START_COMMAND),
        "env": {"LILITH_SERVE_API_KEY": api_key_env, "LILITH_ROOT": POD_ROOT,
                "LILITH_SERVE_KV_CACHE_DTYPE": kv_cache_dtype},
    }


def new_session_id(now: dt.datetime) -> str:
    return "serve-" + now.strftime("%Y%m%dT%H%M%SZ")


def open_session(ctx: Context, state: State, pod_id: str, now: dt.datetime, rate: float, cap: int,
                 source: str, started_at: dt.datetime | None = None) -> Session:
    begin = started_at or now
    session = Session(id=new_session_id(begin), started_at=iso(begin), cap_seconds=cap, rate_usd_per_hr=rate,
                      source=source, last_pod_check_at=iso(now))
    ctx.ledger.start(session.id, pod_id, begin, rate, cap, source)
    state.session = session
    state.save(ctx.state_path)
    ctx.log(f"session {session.id} opened ({source}) on pod {pod_id}: cap {cap // 60} min at ${rate:.2f}/h")
    return session


def close_session(ctx: Context, state: State, reason: str, now: dt.datetime) -> None:
    if state.session is None:
        return
    ctx.ledger.stop(state.session.id, state.pod_id or "?", now, reason)
    started = parse_iso(state.session.started_at)
    hours = (now - started).total_seconds() / 3600
    ctx.log(f"session {state.session.id} closed ({reason}) after {hours:.2f} h "
            f"(~${hours * state.session.rate_usd_per_hr:.2f})")
    state.session = None
    state.save(ctx.state_path)


def stop_pod(ctx: Context, state: State, reason: str) -> None:
    """Stop first, ledger second: a failed stop keeps the session open (it still bills)."""
    if not state.pod_id:
        raise NotCreated("no serve pod recorded")
    ctx.runpod.stop_pod(state.pod_id)
    ctx.log(f"pod {state.pod_id} stopped ({reason})")
    close_session(ctx, state, reason, ctx.clock.now())


def require_pod(ctx: Context) -> State:
    state = ctx.state()
    if state.pod_id:
        return state
    fields = ctx.store.item_fields(ENDPOINT_ITEM)
    if fields and fields.get("pod_id"):
        state.pod_id = fields["pod_id"]
        state.base_url = fields.get("base_url") or proxy_url(state.pod_id, template=ctx.proxy_template)
        state.save(ctx.state_path)
        ctx.log(f"adopted pod {state.pod_id} from 1Password item {ENDPOINT_ITEM}")
        return state
    raise NotCreated("no serve pod recorded locally or in 1Password")


def budget_for(ctx: Context, rate: float) -> BudgetDecision:
    now = ctx.clock.now()
    return evaluate_budget(ctx.policy, ctx.ledger.usage(now), ctx.runpod.balance_usd(), rate, now)


def adopt(ctx: Context, state: State, pod: PodInfo, now: dt.datetime) -> Session:
    """A running pod without a supervised session (console start, lost state): supervise it
    under the budget from now on. A budget of 0 means the next idle-watch tick stops it."""
    rate = pod.rate_usd_per_hr or DEFAULT_RATE_USD_PER_HR
    try:
        cap = budget_for(ctx, rate).cap_seconds
    except ServeError as error:
        ctx.log(f"budget unavailable while adopting ({error}); using the plain session cap")
        cap = ctx.policy.session_cap_s
    started = parse_iso(pod.last_started_at) if pod.last_started_at else now
    return open_session(ctx, state, pod.id, now, rate, cap, "adopted", started)


def cmd_create(ctx: Context, *, yes: bool, key_source: str, kv_cache_dtype: str, source: str = "create") -> dict[str, Any]:
    if not yes:
        raise ConfirmationRequired(f"create makes the billable pod {POD_NAME} (1x H100, ${DEFAULT_RATE_USD_PER_HR}/h)",
                                   help="rerun with --yes once the operator has approved the create")
    state = ctx.state()
    if source == "create":
        existing = state.pod_id or (ctx.store.item_fields(ENDPOINT_ITEM) or {}).get("pod_id")
        if existing:
            raise AlreadyCreated(f"serve pod {existing} already exists",
                                 help="use `start`; after a zero-GPU start use `recreate --yes --confirm-terminate "
                                      f"{existing}`")
    decision = budget_for(ctx, DEFAULT_RATE_USD_PER_HR)
    if not decision.allowed:
        raise BudgetRefused(decision.refusal or "over budget")
    if ctx.store.item_fields(API_KEY_ITEM) is None:
        ctx.store.create_api_credential(API_KEY_ITEM, REDACTOR.add(secrets_module.token_urlsafe(48)))
        ctx.log(f"generated the serve API key into 1Password {VAULT}/{API_KEY_ITEM}")
    key = ctx.serve_key()
    env_value = key if key_source == "env" else "{{ RUNPOD_SECRET_%s }}" % RUNPOD_SECRET_NAME
    pod = ctx.runpod.create_pod(pod_body(env_value, kv_cache_dtype))
    now = ctx.clock.now()
    state.pod_id, state.base_url, state.created_at = pod.id, proxy_url(pod.id, template=ctx.proxy_template), iso(now)
    state.session = None
    state.save(ctx.state_path)
    ctx.log(f"created pod {pod.id} ({POD_NAME}, {DATACENTER}, volume {NETWORK_VOLUME_ID}); base URL {state.base_url}")
    open_session(ctx, state, pod.id, now, pod.rate_usd_per_hr or DEFAULT_RATE_USD_PER_HR, decision.cap_seconds, source)
    ctx.store.upsert_endpoint(pod.id, state.base_url)
    return {"pod_id": pod.id, "base_url": state.base_url, "session_cap_seconds": decision.cap_seconds,
            "next": ["make_payload.sh from a clean origin/main checkout",
                     "lilith_serve.py install-payload <tarball> --launch",
                     "lilith_serve.py wait-ready (or start --watch next time)",
                     "keep lilith_serve.py idle-watch running (LaunchAgent it.akh.lilith-serve-idle)"]}


def wait_ready(ctx: Context, state: State, timeout_s: int, stop_on_failure: bool) -> dict[str, Any]:
    """Health, then one tools request: readiness evidence includes tool calling.

    Returns the tool probe record. A refused or malformed tools response raises
    ToolCallingBroken (and stops the pod when `stop_on_failure`, i.e. under `start`)."""
    probe = Vllm(state.base_url or "", ctx.serve_key)
    deadline = ctx.clock.now() + dt.timedelta(seconds=timeout_s)
    while not probe.healthy():
        if ctx.clock.now() >= deadline:
            if stop_on_failure:
                stop_pod(ctx, state, "health_timeout")
            raise HealthTimeout(f"vLLM at {state.base_url} not healthy after {timeout_s}s"
                                + (" (pod stopped)" if stop_on_failure else ""),
                                help="ssh in and read /workspace/lilith/serve/status.json and serve/logs/; a lock "
                                     "change rebuilds the env (~5 min) before the ~8 min model load")
        ctx.clock.sleep(10)
    ctx.log(f"vLLM healthy at {state.base_url} (model {SERVED_MODEL}); checking tool calling")
    try:
        status, response = probe.chat(probe_request())
        tool_probe = check_tool_calls(status, response, ctx.clock.now())
    except ToolCallingBroken as error:
        if stop_on_failure:
            stop_pod(ctx, ctx.state(), "tool_calling_broken")
            error.args = (f"{error.args[0]} (pod stopped)",)
        raise
    record = tool_probe.to_dict()
    state = ctx.state()
    if state.session is not None:
        if state.session.ready_at is None:
            state.session.ready_at = state.session.last_activity_at = iso(ctx.clock.now())
        state.session.tool_calling = record
        state.save(ctx.state_path)
        evidence = ctx.state_dir / "readiness" / f"{state.session.id}.json"
        evidence.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        evidence.write_text(json.dumps(record | {"session": state.session.id, "pod_id": state.pod_id,
                                                  "base_url": state.base_url}, indent=2) + "\n")
    ctx.log(f"tool calling {tool_probe.status}: {tool_probe.detail} (finish_reason={tool_probe.finish_reason})")
    return record


def cmd_start(ctx: Context, *, wait_seconds: int) -> dict[str, Any]:
    state = require_pod(ctx)
    pod = ctx.runpod.get_pod(state.pod_id)
    now = ctx.clock.now()
    if pod.desired_status == "RUNNING":
        if state.session is None:
            adopt(ctx, state, pod, now)
        ctx.log(f"pod {pod.id} is already running; not starting it again")
        return {"pod_id": pod.id, "base_url": state.base_url, "already_running": True}
    rate = pod.rate_usd_per_hr or DEFAULT_RATE_USD_PER_HR
    decision = budget_for(ctx, rate)
    if not decision.allowed:
        raise BudgetRefused(decision.refusal or "over budget",
                            help="the pod stays stopped; serve locally (C6b) or wait for the reset named above")
    ctx.runpod.start_pod(pod.id)
    open_session(ctx, state, pod.id, ctx.clock.now(), rate, decision.cap_seconds, "start")
    # A resumed pod is tied to its machine; with that machine's GPUs rented out it comes back with 0.
    for _ in range(12):
        pod = ctx.runpod.get_pod(pod.id)
        if pod.gpu_count is not None:
            break
        ctx.clock.sleep(5)
    if pod.gpu_count == 0:
        stop_pod(ctx, ctx.state(), "zero_gpu")
        raise ZeroGpu(f"pod {pod.id} resumed with zero GPUs (its machine's H100 is rented out); stopped it again",
                      help=f"options: (1) retry `start` later; (2) `recreate --yes --confirm-terminate {pod.id}` "
                           "(new pod on the same volume, NEW proxy URL: update the lilith akh's base_url); "
                           "(3) serve locally once a local card qualifies (C6b)")
    tool_calling = wait_ready(ctx, ctx.state(), wait_seconds, stop_on_failure=True)
    state = ctx.state()
    stop_by = parse_iso(state.session.started_at) + dt.timedelta(seconds=state.session.cap_seconds)
    return {"pod_id": pod.id, "base_url": state.base_url, "model": SERVED_MODEL, "tool_calling": tool_calling,
            "session": state.session.id, "session_cap_seconds": state.session.cap_seconds,
            "stop_by": iso(stop_by), "idle_stop_minutes": ctx.policy.idle_s // 60, "budget": decision.to_dict()}


def cmd_stop(ctx: Context, reason: str) -> dict[str, Any]:
    state = require_pod(ctx)
    pod = ctx.runpod.get_pod(state.pod_id)
    if pod.desired_status != "RUNNING":
        close_session(ctx, state, "external", ctx.clock.now())
        return {"pod_id": pod.id, "status": pod.desired_status, "stopped": False}
    stop_pod(ctx, state, reason)
    return {"pod_id": pod.id, "stopped": True, "reason": reason}


class TickOutcome(enum.StrEnum):
    NO_SESSION = "no_session"
    ACTIVE = "active"
    IDLE_STOP = "idle_stop"
    CAP_STOP = "session_cap_stop"
    EXTERNAL_STOP = "external_stop"
    STOP_FAILED = "stop_failed"
    ADOPTED = "adopted"


def watch_tick(ctx: Context, probe: Vllm) -> TickOutcome:
    """One idle-watch step. Metrics come through the public proxy (no Runpod key needed); the
    Runpod API is consulted every `pod_check_s` and for stops only."""
    state = ctx.state()
    now = ctx.clock.now()
    policy = ctx.policy
    session = state.session
    if session is None:
        last = state.last_idle_pod_check_at
        if not state.pod_id or (last and (now - parse_iso(last)).total_seconds() < policy.idle_pod_check_s):
            return TickOutcome.NO_SESSION
        state.last_idle_pod_check_at = iso(now)
        state.save(ctx.state_path)
        pod = ctx.runpod.get_pod(state.pod_id)
        if pod.desired_status != "RUNNING":
            return TickOutcome.NO_SESSION
        adopt(ctx, state, pod, now)
        ctx.log(f"pod {pod.id} was running without a supervised session (console start?); adopted it")
        return TickOutcome.ADOPTED

    def stop(reason: str, outcome: TickOutcome) -> TickOutcome:
        try:
            stop_pod(ctx, state, reason)
        except ServeError as error:
            ctx.log(f"STOP FAILED for pod {state.pod_id} ({reason}): {error.render()}; retrying next tick")
            return TickOutcome.STOP_FAILED
        return outcome

    elapsed = (now - parse_iso(session.started_at)).total_seconds()
    if elapsed >= session.cap_seconds:
        return stop("session_cap", TickOutcome.CAP_STOP)

    if not session.last_pod_check_at or (now - parse_iso(session.last_pod_check_at)).total_seconds() >= policy.pod_check_s:
        session.last_pod_check_at = iso(now)
        try:
            pod = ctx.runpod.get_pod(state.pod_id)
        except ServeError as error:
            ctx.log(f"pod status check failed: {error}")
        else:
            if pod.desired_status != "RUNNING":
                close_session(ctx, state, "external", now)
                return TickOutcome.EXTERNAL_STOP

    try:
        metrics = probe.metrics()
    except ServeError as error:
        session.metrics_misses += 1
        if session.metrics_misses in (1, 5) or session.metrics_misses % 30 == 0:
            ctx.log(f"metrics miss {session.metrics_misses}: {error}")
    else:
        session.metrics_misses = 0
        if session.ready_at is None:
            session.ready_at = session.last_activity_at = iso(now)
            ctx.log("vLLM metrics reachable: idle clock starts")
        activity = list(metrics.activity)
        changed = session.last_activity is not None and activity != session.last_activity
        if changed or metrics.in_flight > 0:  # the first read is only a baseline
            session.last_activity_at = iso(now)
        session.last_activity = activity
    state.save(ctx.state_path)

    # The idle clock runs from readiness: model load time is not idle time (the cap covers it).
    if session.ready_at is not None and session.last_activity_at is not None:
        idle = (now - parse_iso(session.last_activity_at)).total_seconds()
        if idle >= policy.idle_s:
            return stop("idle", TickOutcome.IDLE_STOP)
    return TickOutcome.ACTIVE


def memoize(read: Callable[[], str]) -> Callable[[], str | None]:
    """Read a secret once per process (idle-watch probes /metrics every minute)."""
    cache: list[str] = []

    def get() -> str | None:
        if not cache:
            cache.append(read())
        return cache[0]
    return get


def cmd_idle_watch(ctx: Context, *, daemon: bool, once: bool, serve_key: Callable[[], str | None]) -> TickOutcome:
    serve_key = memoize(serve_key)  # type: ignore[arg-type]
    outcome = TickOutcome.NO_SESSION
    probe: Vllm | None = None
    while True:
        state = ctx.state()
        if state.base_url and (probe is None or probe.base_url != state.base_url.rstrip("/")):
            probe = Vllm(state.base_url, serve_key)
        try:
            outcome = watch_tick(ctx, probe or Vllm("http://invalid.localhost", serve_key))
        except ServeError as error:
            ctx.log(f"idle-watch tick failed: {error.render()}")
            outcome = TickOutcome.STOP_FAILED
        if once:
            return outcome
        if not daemon and outcome in (TickOutcome.NO_SESSION, TickOutcome.IDLE_STOP, TickOutcome.CAP_STOP,
                                      TickOutcome.EXTERNAL_STOP):
            return outcome
        ctx.clock.sleep(ctx.policy.interval_s)


def cmd_status(ctx: Context) -> dict[str, Any]:
    state = require_pod(ctx)
    pod = ctx.runpod.get_pod(state.pod_id)
    now = ctx.clock.now()
    out: dict[str, Any] = {"pod_id": pod.id, "base_url": state.base_url, "desired_status": pod.desired_status,
                           "gpu_count": pod.gpu_count, "rate_usd_per_hr": pod.rate_usd_per_hr,
                           "usage": ctx.ledger.usage(now).to_dict()}
    if state.session is not None:
        s = state.session
        elapsed = (now - parse_iso(s.started_at)).total_seconds()
        out["session"] = {"id": s.id, "elapsed_minutes": round(elapsed / 60, 1),
                          "cap_minutes": s.cap_seconds // 60, "ready": s.ready_at is not None,
                          "idle_minutes": None if not s.last_activity_at else
                          round((now - parse_iso(s.last_activity_at)).total_seconds() / 60, 1)}
    if pod.desired_status == "RUNNING" and state.base_url:
        out["healthy"] = Vllm(state.base_url).healthy()
    return out


def cmd_ledger(ctx: Context, offline: bool) -> dict[str, Any]:
    now = ctx.clock.now()
    usage = ctx.ledger.usage(now)
    out: dict[str, Any] = {"usage": usage.to_dict(), "policy": dataclasses.asdict(ctx.policy),
                           "week_starts": iso(week_start(now)), "month_starts": iso(month_start(now))}
    if not offline:
        state = ctx.state()
        rate = (state.session.rate_usd_per_hr if state.session else None) or DEFAULT_RATE_USD_PER_HR
        out["budget"] = evaluate_budget(ctx.policy, usage, ctx.runpod.balance_usd(), rate, now).to_dict()
    return out


def cmd_recreate(ctx: Context, *, yes: bool, confirm: str | None, key_source: str, kv_cache_dtype: str) -> dict[str, Any]:
    state = require_pod(ctx)
    if not yes or confirm != state.pod_id:
        raise ConfirmationRequired(f"recreate terminates pod {state.pod_id} and creates a new billable pod "
                                   "with a NEW proxy URL",
                                   help=f"rerun with --yes --confirm-terminate {state.pod_id} once the operator approved")
    old, old_url = state.pod_id, state.base_url
    if state.session is not None:
        close_session(ctx, state, "recreate", ctx.clock.now())
    ctx.runpod.delete_pod(old)
    ctx.log(f"terminated pod {old} (the network volume {NETWORK_VOLUME_ID} keeps /workspace)")
    result = cmd_create(ctx, yes=True, key_source=key_source, kv_cache_dtype=kv_cache_dtype, source="recreate")
    ctx.log(f"BASE URL CHANGED: {old_url} -> {result['base_url']}; update the lilith akh's chat base_url")
    return result | {"old_pod_id": old, "old_base_url": old_url, "base_url_changed": True}


def cmd_sync_endpoint(ctx: Context) -> dict[str, Any]:
    state = ctx.state()
    if not state.pod_id or not state.base_url:
        raise NotCreated("no local pod record to sync")
    ctx.store.upsert_endpoint(state.pod_id, state.base_url)
    return {"pod_id": state.pod_id, "base_url": state.base_url, "synced": True}


# ----------------------------------------------------------------- payload install (ssh)
PAYLOAD_NAME = re.compile(r"^lilith-payload-main-([0-9a-f]{12})\.tar\.gz$")


def payload_main_sha(tarball: Path) -> str:
    match = PAYLOAD_NAME.match(tarball.name)
    if not match:
        raise UsageError(f"{tarball.name} is not a make_payload.sh tarball (lilith-payload-main-<sha12>.tar.gz)")
    with tarfile.open(tarball) as archive:
        member = archive.extractfile("source-manifest.json")
        if member is None:
            raise ProvenanceError(f"{tarball} has no source-manifest.json")
        sha = json.load(member)["main_sha"]
    if not sha.startswith(match.group(1)):
        raise ProvenanceError(f"manifest commit {sha} does not match the tarball name")
    return sha


def check_on_main(repo: Path, sha: str) -> None:
    subprocess.run(["git", "-C", str(repo), "fetch", "-q", "origin", "main"], check=False, timeout=120)
    result = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", sha, "origin/main"],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ProvenanceError(f"payload commit {sha} is not an ancestor of origin/main",
                              help="deploy only from main: merge the PR, rebuild the payload from a clean "
                                   "origin/main checkout (make_payload.sh), then install that")


def cmd_install_payload(ctx: Context, tarball: Path, *, launch: bool, identity: str | None) -> dict[str, Any]:
    state = require_pod(ctx)
    sha = payload_main_sha(tarball)
    check_on_main(ctx.repo, sha)
    pod = ctx.runpod.get_pod(state.pod_id)
    if pod.desired_status != "RUNNING" or not pod.public_ip or "22" not in pod.port_mappings:
        raise UsageError(f"pod {pod.id} is {pod.desired_status} without public ssh",
                         help="the pod must be running (create, or start) with 22/tcp exposed")
    short = sha[:12]
    target = f"{POD_ROOT}/serve/payloads/main-{short}"
    remote = (
        f"set -euo pipefail; mkdir -p {POD_ROOT}/serve/payloads {POD_ROOT}/serve/logs; T={target}; "
        'if [ ! -d "$T" ]; then rm -rf "$T.partial"; mkdir -p "$T.partial"; tar -xzf - -C "$T.partial"; '
        'mv "$T.partial" "$T"; else cat > /dev/null; fi; '
        f'python3 "$T/scripts/lilith/runpod/verify_payload.py" verify --root "$T" --manifest "$T/source-manifest.json" '
        f'--expected-sha {sha} --output "$T/provenance-$(date -u +%Y%m%dT%H%M%SZ).json"; '
        f"printf '%s\\n' '{json.dumps({'schema': 'lilith/serve-current@1', 'main_sha': sha, 'payload': target})}' "
        f"> {POD_ROOT}/serve/current.json; "
        f'ln -sfn "$T" {POD_ROOT}/serve/current.new && mv -T {POD_ROOT}/serve/current.new {POD_ROOT}/serve/current'
    )
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new",
           "-o", f"UserKnownHostsFile={ctx.state_dir / 'known_hosts'}", "-p", str(pod.port_mappings["22"])]
    if identity:
        ssh += ["-i", identity, "-o", "IdentitiesOnly=yes"]
    host = f"root@{pod.public_ip}"
    with tarball.open("rb") as payload:
        result = subprocess.run([*ssh, host, remote], stdin=payload, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ProvenanceError(f"payload install failed on the pod (exit {result.returncode}): "
                              f"{REDACTOR.scrub(result.stderr.strip())[-600:]}",
                              help="the pod verifies the payload against its manifest and main SHA; rebuild it")
    ctx.log(f"payload main-{short} installed and verified on pod {pod.id}; current -> {target}")
    if launch:
        command = f"nohup bash {shlex.quote(SERVE_ENTRY)} >> {SERVE_LOG} 2>&1 < /dev/null &"
        subprocess.run([*ssh, host, command], check=True, capture_output=True, timeout=60)
        ctx.log("serve_pod.sh launched; wait with `lilith_serve.py wait-ready`")
    return {"pod_id": pod.id, "main_sha": sha, "payload": target, "launched": launch}


# ----------------------------------------------------------------- CLI
def build_context(policy: Policy, *, cache_key: bool) -> Context:
    state_dir = Path(os.environ.get("LILITH_SERVE_STATE_DIR", Path.home() / "lilith-private" / "serve"))
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    store = OnePassword(os.environ.get("LILITH_SERVE_OP", str(Path.home() / ".local/bin/op-service-account")))
    runpod = Runpod(KeyProvider(store, RUNPOD_KEY_REF, cache=cache_key),
                    os.environ.get("LILITH_SERVE_RUNPOD_REST", "https://rest.runpod.io/v1"),
                    os.environ.get("LILITH_SERVE_RUNPOD_GRAPHQL", "https://api.runpod.io/graphql"))
    return Context(state_dir=state_dir, store=store, runpod=runpod, policy=policy, clock=RealClock(),
                   log=Log(state_dir), proxy_template=os.environ.get("LILITH_SERVE_PROXY_TEMPLATE"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="one-time: create the serve pod (billable; needs --yes)")
    recreate = sub.add_parser("recreate", help="terminate + create (new pod id and URL; needs confirmation)")
    for p in (create, recreate):
        p.add_argument("--yes", action="store_true")
        p.add_argument("--key-source", choices=("env", "runpod-secret"), default="env",
                       help=f"env: the key value in the pod env; runpod-secret: {{{{ RUNPOD_SECRET_{RUNPOD_SECRET_NAME} }}}}")
        p.add_argument("--kv-cache-dtype", choices=KV_CACHE_DTYPES, default="auto")
    recreate.add_argument("--confirm-terminate", metavar="POD_ID")
    install = sub.add_parser("install-payload", help="upload + verify a make_payload.sh tarball over ssh")
    install.add_argument("tarball", type=Path)
    install.add_argument("--launch", action="store_true")
    install.add_argument("--identity", default=None)
    start = sub.add_parser("start", help="resume the pod within budget and wait for vLLM")
    start.add_argument("--wait-seconds", type=int, default=1800)
    start.add_argument("--watch", action="store_true", help="run idle-watch in the foreground afterwards")
    ready = sub.add_parser("wait-ready", help="wait for vLLM health (after create + install-payload --launch)")
    ready.add_argument("--wait-seconds", type=int, default=1800)
    stop = sub.add_parser("stop")
    stop.add_argument("--reason", default="manual")
    status = sub.add_parser("status")
    status.add_argument("--json", action="store_true")
    watch = sub.add_parser("idle-watch", help="stop on 20 min idle or the session cap")
    watch.add_argument("--interval", type=int, default=60)
    watch.add_argument("--daemon", action="store_true", help="keep running between sessions (LaunchAgent)")
    watch.add_argument("--once", action="store_true")
    ledger = sub.add_parser("ledger")
    ledger.add_argument("--json", action="store_true")
    ledger.add_argument("--offline", action="store_true", help="skip the Runpod balance query")
    sub.add_parser("sync-endpoint", help=f"write the local pod id/base URL to 1Password {ENDPOINT_ITEM}")
    args = parser.parse_args(argv)

    policy = Policy(interval_s=getattr(args, "interval", 60) or 60)
    ctx = build_context(policy, cache_key=args.command != "idle-watch")
    try:
        if args.command == "create":
            result: Any = cmd_create(ctx, yes=args.yes, key_source=args.key_source, kv_cache_dtype=args.kv_cache_dtype)
        elif args.command == "recreate":
            result = cmd_recreate(ctx, yes=args.yes, confirm=args.confirm_terminate, key_source=args.key_source,
                                  kv_cache_dtype=args.kv_cache_dtype)
        elif args.command == "install-payload":
            result = cmd_install_payload(ctx, args.tarball, launch=args.launch, identity=args.identity)
        elif args.command == "start":
            result = cmd_start(ctx, wait_seconds=args.wait_seconds)
            if args.watch:
                print(json.dumps(result, indent=2))
                result = {"idle_watch": cmd_idle_watch(ctx, daemon=False, once=False, serve_key=ctx.serve_key).value}
        elif args.command == "wait-ready":
            tool_calling = wait_ready(ctx, require_pod(ctx), args.wait_seconds, stop_on_failure=False)
            result = {"base_url": ctx.state().base_url, "healthy": True, "tool_calling": tool_calling}
        elif args.command == "stop":
            result = cmd_stop(ctx, args.reason)
        elif args.command == "status":
            result = cmd_status(ctx)
        elif args.command == "idle-watch":
            result = {"idle_watch": cmd_idle_watch(ctx, daemon=args.daemon, once=args.once,
                                                   serve_key=ctx.serve_key).value}
        elif args.command == "ledger":
            result = cmd_ledger(ctx, args.offline)
        else:
            result = cmd_sync_endpoint(ctx)
    except ServeError as error:
        ctx.log(error.render())
        return error.exit_code
    except KeyboardInterrupt:
        ctx.log("interrupted; the pod state is unchanged (check `status`)")
        return 130
    print(REDACTOR.scrub(json.dumps(result, indent=2, sort_keys=True, default=str)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
