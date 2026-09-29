"""Model adapters for the dojo episode runner (Anima WI 01a0e270-475e, plan slice C4).

One small protocol, `ModelAdapter.chat(messages, tools) -> AssistantReply`, with two
implementations that translate the SAME neutral messages and tool list, so a frozen
parent (OpenAI-compatible vLLM) and a Claude teacher (Anthropic Messages API) see identical
prompts and tools ("same broker", exam workflow §2):

- `OpenAIChatAdapter`: `POST {base_url}/chat/completions`. Sends `User-Agent: curl/8.7.1`
  (Runpod's proxy answers 403 to Python's default UA). Reads `tool_calls`; keeps
  `reasoning_content` / `reasoning` (and a leading `<think>` block) out of the answer
  content.
- `AnthropicAdapter`: `POST {base_url}/v1/messages` with `x-api-key`, `anthropic-version`,
  `tool_use` / `tool_result` blocks.

Secrets: an API key comes from an environment variable or a 0600 file (`load_secret`). It is
held in a `Secret` whose repr is redacted and it is scrubbed from every error text. It is
never accepted on the command line, logged, or written to disk.

stdlib only (urllib). No paid API is called by the tests: they use local fake servers.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import stat
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dojo.errors import ModelError, SecretSourceError

USER_AGENT = "curl/8.7.1"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
# The newest Opus id the repo names (docs/ai/plans/2026-04-25-phase39-distributed-inference.md).
# The exam plan (§3) names the judge "Opus 5.5" but no API id for it appears in the repo, so the
# operator passes `--model` for the teacher run; this default only keeps the CLI usable.
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"  # Opus 5.5, the exam plan's teacher
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 529})


# --- secrets ------------------------------------------------------------------------------


class Secret:
    """An API key. `repr`/`str` never show it; `reveal()` is called only when building headers."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        value = value.strip()
        if not value:
            raise SecretSourceError("the key source is empty")
        self._value = value

    def reveal(self) -> str:
        return self._value

    def scrub(self, text: str) -> str:
        return text.replace(self._value, "***")

    def __repr__(self) -> str:
        return "Secret(***)"

    __str__ = __repr__


def load_secret(*, env: str | None = None, file: Path | None = None,
                environ: Mapping[str, str] | None = None) -> Secret:
    """Exactly one of `env` (variable name) or `file` (mode 0600, owned by the caller)."""
    if (env is None) == (file is None):
        raise SecretSourceError("give exactly one of --key-env and --key-file")
    if env is not None:
        value = (environ if environ is not None else os.environ).get(env)
        if not value:
            raise SecretSourceError(f"environment variable {env} is not set or empty")
        return Secret(value)
    path = Path(file)  # type: ignore[arg-type]
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise SecretSourceError(f"cannot read key file {path}: {error.strerror}") from None
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise SecretSourceError(f"key file {path} is readable by group or others (mode {stat.S_IMODE(mode):04o})")
    try:
        return Secret(path.read_text())
    except OSError as error:
        raise SecretSourceError(f"cannot read key file {path}: {error.strerror}") from None


# --- neutral message types ----------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any]

    def schema(self) -> dict:
        schema = dict(self.input_schema) if self.input_schema else {}
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        return schema


@dataclasses.dataclass(frozen=True)
class ModelToolCall:
    """A tool call the model requested. `arguments` is None when the model's argument text was
    not a JSON object (`arguments_error` says why): the broker then answers with an error and
    the model can retry."""

    id: str
    name: str
    arguments: Mapping[str, Any] | None
    arguments_error: str | None = None


@dataclasses.dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.prompt_tokens + other.prompt_tokens, self.completion_tokens + other.completion_tokens,
                     self.cached_prompt_tokens + other.cached_prompt_tokens)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ChatMessage:
    """Neutral conversation turn. Roles: system, user, assistant, tool."""

    role: str
    content: str = ""
    tool_calls: tuple[ModelToolCall, ...] = ()
    tool_call_id: str | None = None  # role == "tool"
    is_error: bool = False  # role == "tool"


@dataclasses.dataclass(frozen=True)
class AssistantReply:
    content: str  # the answer; reasoning is never in here
    tool_calls: tuple[ModelToolCall, ...]
    reasoning: str | None
    usage: Usage
    finish_reason: str | None


class ModelAdapter(Protocol):
    kind: str
    model_id: str
    endpoint_host: str

    def chat(self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> AssistantReply: ...


# --- HTTP ------------------------------------------------------------------------------------

Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]


def urllib_transport(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


@dataclasses.dataclass(frozen=True)
class HttpPolicy:
    timeout_s: float = 300.0
    max_attempts: int = 4
    backoff_s: float = 2.0  # attempt n waits backoff_s * 2**(n-1)


def post_json(url: str, headers: Mapping[str, str], payload: Mapping[str, Any], secret: Secret,
              policy: HttpPolicy = HttpPolicy(), transport: Transport = urllib_transport,
              sleep: Callable[[float], None] = time.sleep) -> dict:
    """POST JSON, retry transient failures, return the decoded JSON object.

    Every failure becomes a `ModelError` naming host, status and a scrubbed body excerpt."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    host = urlparse(url).hostname or url
    last = "no attempt made"
    for attempt in range(1, max(1, policy.max_attempts) + 1):
        try:
            status, raw = transport(url, headers, body, policy.timeout_s)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            last = f"transport error from {host}: {secret.scrub(str(getattr(error, 'reason', error)))}"
        else:
            if 200 <= status < 300:
                try:
                    decoded = json.loads(raw)
                except ValueError:
                    raise ModelError(f"{host} answered {status} with a body that is not JSON") from None
                if not isinstance(decoded, dict):
                    raise ModelError(f"{host} answered {status} with JSON that is not an object")
                return decoded
            excerpt = secret.scrub(raw[:300].decode("utf-8", "replace")).replace("\n", " ")
            last = f"{host} answered HTTP {status}: {excerpt}"
            if status not in RETRYABLE_STATUS:
                raise ModelError(last, help="a 4xx is not retried: check the model id, the key and the request size")
        if attempt < policy.max_attempts:
            sleep(policy.backoff_s * 2 ** (attempt - 1))
    raise ModelError(f"{last} (after {policy.max_attempts} attempts)")


_THINK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)


def split_reasoning(content: str) -> tuple[str, str | None]:
    """A leading `<think>...</think>` block is reasoning, not answer (structural tag, not language)."""
    match = _THINK.match(content)
    if match is None:
        return content, None
    return content[match.end():], match.group(1).strip() or None


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in (None, "text"))
    return ""


def _parse_arguments(raw: Any) -> tuple[Mapping[str, Any] | None, str | None]:
    if isinstance(raw, dict):
        return raw, None
    if raw in (None, ""):
        return {}, None
    if not isinstance(raw, str):
        return None, f"arguments have type {type(raw).__name__}, expected a JSON object"
    try:
        decoded = json.loads(raw)
    except ValueError as error:
        return None, f"arguments are not valid JSON ({error})"
    if not isinstance(decoded, dict):
        return None, "arguments are not a JSON object"
    return decoded, None


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


# --- OpenAI-compatible -------------------------------------------------------------------------


class OpenAIChatAdapter:
    kind = "openai"

    def __init__(self, base_url: str, model: str, secret: Secret, *, temperature: float | None = 0.0,
                 max_tokens: int = 4096, policy: HttpPolicy = HttpPolicy(), transport: Transport = urllib_transport,
                 sleep: Callable[[float], None] = time.sleep):
        self.base_url = base_url.rstrip("/")
        self.model_id = model
        self.endpoint_host = urlparse(self.base_url).hostname or self.base_url
        self._secret = secret
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._policy = policy
        self._transport = transport
        self._sleep = sleep

    def request_body(self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> dict:
        wire: list[dict] = []
        for m in messages:
            if m.role == "assistant":
                turn: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    turn["tool_calls"] = [{"id": c.id, "type": "function",
                                           "function": {"name": c.name, "arguments": json.dumps(dict(c.arguments or {}), ensure_ascii=False)}}
                                          for c in m.tool_calls]
                wire.append(turn)
            elif m.role == "tool":
                wire.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            else:
                wire.append({"role": m.role, "content": m.content})
        body: dict[str, Any] = {"model": self.model_id, "messages": wire, "max_tokens": self._max_tokens}
        if self._temperature is not None:
            body["temperature"] = self._temperature
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                               "parameters": t.schema()}} for t in tools]
            body["tool_choice"] = "auto"
        return body

    def chat(self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> AssistantReply:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            "Authorization": f"Bearer {self._secret.reveal()}",
        }
        raw = post_json(f"{self.base_url}/chat/completions", headers, self.request_body(messages, tools),
                        self._secret, self._policy, self._transport, self._sleep)
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ModelError(f"{self.endpoint_host} reply has no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        content, inline_reasoning = split_reasoning(_text_of(message.get("content")))
        reasoning = message.get("reasoning_content") or message.get("reasoning") or inline_reasoning
        calls = []
        for n, call in enumerate(message.get("tool_calls") or []):
            function = (call or {}).get("function") or {}
            arguments, problem = _parse_arguments(function.get("arguments"))
            calls.append(ModelToolCall(str(call.get("id") or f"call_{n}"), str(function.get("name") or ""), arguments, problem))
        usage = raw.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        return AssistantReply(
            content=content,
            tool_calls=tuple(calls),
            reasoning=reasoning if isinstance(reasoning, str) and reasoning else None,
            usage=Usage(_int(usage.get("prompt_tokens")), _int(usage.get("completion_tokens")),
                        _int(details.get("cached_tokens")) if isinstance(details, dict) else 0),
            finish_reason=choice.get("finish_reason"),
        )


# --- Anthropic Messages ----------------------------------------------------------------------------


class AnthropicAdapter:
    kind = "anthropic"

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, secret: Secret | None = None, *,
                 base_url: str = ANTHROPIC_BASE_URL, temperature: float | None = 0.0, max_tokens: int = 4096,
                 cache: bool = True, policy: HttpPolicy = HttpPolicy(), transport: Transport = urllib_transport,
                 sleep: Callable[[float], None] = time.sleep):
        if secret is None:
            raise SecretSourceError("the Anthropic adapter needs a key")
        self.base_url = base_url.rstrip("/")
        self.model_id = model
        self.endpoint_host = urlparse(self.base_url).hostname or self.base_url
        self._secret = secret
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._cache = cache
        self._policy = policy
        self._transport = transport
        self._sleep = sleep

    def request_body(self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> dict:
        system_text = "\n\n".join(m.content for m in messages if m.role == "system")
        wire: list[dict] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
                if m.is_error:
                    block["is_error"] = True
                if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list) \
                        and all(b.get("type") == "tool_result" for b in wire[-1]["content"]):
                    wire[-1]["content"].append(block)  # all results of one turn share one user message
                else:
                    wire.append({"role": "user", "content": [block]})
            elif m.role == "assistant":
                blocks: list[dict] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": dict(c.arguments or {})}
                           for c in m.tool_calls]
                wire.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
            else:
                wire.append({"role": "user", "content": [{"type": "text", "text": m.content}]})
        body: dict[str, Any] = {"model": self.model_id, "max_tokens": self._max_tokens, "messages": wire}
        if system_text:
            block = {"type": "text", "text": system_text}
            if self._cache:
                block["cache_control"] = {"type": "ephemeral"}
            body["system"] = [block]
        if self._temperature is not None:
            body["temperature"] = self._temperature
        if tools:
            listed = [{"name": t.name, "description": t.description, "input_schema": t.schema()} for t in tools]
            if self._cache:
                listed[-1]["cache_control"] = {"type": "ephemeral"}
            body["tools"] = listed
        return body

    def chat(self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> AssistantReply:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            "x-api-key": self._secret.reveal(),
            "anthropic-version": ANTHROPIC_VERSION,
        }
        raw = post_json(f"{self.base_url}/v1/messages", headers, self.request_body(messages, tools),
                        self._secret, self._policy, self._transport, self._sleep)
        blocks = raw.get("content")
        if not isinstance(blocks, list):
            raise ModelError(f"{self.endpoint_host} reply has no content blocks")
        text: list[str] = []
        thinking: list[str] = []
        calls: list[ModelToolCall] = []
        for n, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                text.append(str(block.get("text", "")))
            elif kind in ("thinking", "redacted_thinking"):
                thinking.append(str(block.get("thinking") or ""))
            elif kind == "tool_use":
                arguments, problem = _parse_arguments(block.get("input"))
                calls.append(ModelToolCall(str(block.get("id") or f"toolu_{n}"), str(block.get("name") or ""), arguments, problem))
        usage = raw.get("usage") or {}
        cached = _int(usage.get("cache_read_input_tokens"))
        prompt = _int(usage.get("input_tokens")) + cached + _int(usage.get("cache_creation_input_tokens"))
        reasoning = "\n".join(t for t in thinking if t) or None
        return AssistantReply("".join(text), tuple(calls), reasoning,
                              Usage(prompt, _int(usage.get("output_tokens")), cached), raw.get("stop_reason"))
