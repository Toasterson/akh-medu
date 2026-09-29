#!/usr/bin/env python3
"""Verify Lilith's llama-server (ADR 121). Standard library only, Python 3.10+.

Runs from the Mac (or hephaistos) against the LAN endpoint, with the API key from a file or the
environment, never from argv:

    python3 scripts/lilith/hephaistos/verify.py \\
        --base-url http://192.168.1.114:8080 --key-file "$ROOT/secrets/lilith-openai-key"

Checks, each PASS / WARN / FAIL:

  health           GET /health answers 200 without a key (the only public endpoint)
  auth             /v1/models and /v1/chat/completions refuse a request without the key
  models           GET /v1/models lists the served name (default `lilith`)
  context          /props reports a per-slot context of at least --min-ctx (default 49152, the
                   akh's `context_window`)
  tool_call        the readiness probe of scripts/lilith/serve/lilith_serve.py: a `get_time`
                   tool, tool_choice auto, temperature 0, must return well-formed `tool_calls`
                   (id, type function, name get_time, JSON-object arguments), and no native
                   Gemma 4 call syntax may be left in `content`
  tool_roundtrip   the tool result goes back in a `tool` message and the model answers in plain
                   text (the turn shape the akh's agent loop uses)
  reasoning_leak   `content` never carries the channel/thought/tool markers, with the system
                   message + tools the akh always sends and with thinking switched on; with
                   thinking on, `reasoning_content` is reported (WARN when the model produced no
                   reasoning: the parser cannot be shown to route it)
  throughput_<N>   prefill and decode tokens/s for a prompt of about N tokens (default 8192 and
                   32768; cache_prompt off so the prefill is real), from the server's own
                   `timings`; WARN below --min-decode-tps / --min-prefill-tps when given

Exit status: 0 no FAIL, 1 at least one FAIL, 2 could not run (unreachable, no key).
`--json PATH` writes the full record (no secrets). `--vram` (on hephaistos) adds GPU memory in use
from /sys/class/drm after load and after each throughput probe.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

DEFAULT_MODEL = "lilith"
DEFAULT_SIZES = (8192, 32768)
MIN_CTX = 49152
USER_AGENT = "lilith-verify/1"

# Gemma 4's channel / thought / tool syntax. None of it may reach the akh in `content`.
LEAK_MARKERS = ("<|channel>", "<channel|>", "<|think|>", "<|tool_call>", "<tool_call|>",
                "<|tool_response>", "<tool_response|>", "<|turn>", "<turn|>")
NATIVE_CALL_MARKERS = ("<|tool_call>", "<tool_call|>")

PROBE_TOOL_NAME = "get_time"
PROBE_TOOL = {"type": "function", "function": {
    "name": PROBE_TOOL_NAME, "description": "Return the current time as an ISO-8601 string.",
    "parameters": {"type": "object", "properties": {"timezone": {"type": "string",
                                                                 "description": "IANA timezone, e.g. UTC"}}}}}

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclasses.dataclass
class Check:
    name: str
    status: str
    detail: str
    data: dict[str, Any] = dataclasses.field(default_factory=dict)


class Unreachable(Exception):
    """The server could not be reached at all (exit 2)."""


class Client:
    def __init__(self, base_url: str, key: str | None, model: str, timeout: float = 900.0,
                 opener: Callable[..., Any] = urllib.request.urlopen):
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.model = model
        self.timeout = timeout
        self._opener = opener

    def scrub(self, text: str) -> str:
        return text.replace(self.key, "***") if self.key else text

    def request(self, method: str, path: str, body: Mapping[str, Any] | None = None,
                *, auth: bool = True, timeout: float | None = None) -> tuple[int, Any]:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if auth and self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        request = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers)
        try:
            with self._opener(request, timeout=timeout or self.timeout) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, raw = error.code, error.read()
        except (urllib.error.URLError, OSError, socket.timeout) as error:
            raise Unreachable(self.scrub(f"{method} {self.base_url + path}: {error}")) from error
        try:
            return status, json.loads(raw)
        except json.JSONDecodeError:
            return status, self.scrub(raw.decode(errors="replace"))[:400]

    def chat(self, body: Mapping[str, Any], timeout: float | None = None) -> tuple[int, Any]:
        return self.request("POST", "/v1/chat/completions", {"model": self.model, **body}, timeout=timeout)


# ------------------------------------------------------------------------------------------- pure checks


def message_of(response: Any) -> Mapping[str, Any]:
    choices = response.get("choices") if isinstance(response, Mapping) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
        if isinstance(message, Mapping):
            return message
    return {}


def finish_reason_of(response: Any) -> str | None:
    choices = response.get("choices") if isinstance(response, Mapping) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        return choices[0].get("finish_reason")
    return None


def leaked_markers(text: str | None) -> list[str]:
    return [marker for marker in LEAK_MARKERS if marker in (text or "")]


def judge_tool_call(status: int, response: Any) -> Check:
    """The `lilith_serve.check_tool_calls` rules, strict: plain text instead of a call is a FAIL here
    (the serve script records it as `unverified`, which is not good enough to hand Lilith a server)."""
    if status != 200 or not isinstance(response, Mapping):
        return Check("tool_call", FAIL, f"HTTP {status}: {str(response)[:300]}")
    message = message_of(response)
    if not message:
        return Check("tool_call", FAIL, "no choices in the response")
    calls = message.get("tool_calls") or []
    content = message.get("content") or ""
    if not calls:
        if any(marker in content for marker in NATIVE_CALL_MARKERS):
            return Check("tool_call", FAIL, "the model emitted native tool-call syntax that the server left "
                         "unparsed in `content` (chat template and tool-call parser do not match)")
        return Check("tool_call", FAIL, "the model answered in plain text; no tool_calls "
                     f"(finish_reason={finish_reason_of(response)})", {"content": content[:200]})
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
        return Check("tool_call", FAIL, "malformed tool_calls: " + "; ".join(problems))
    leaks = leaked_markers(content)
    if leaks:
        return Check("tool_call", FAIL, f"tool_calls are fine but `content` leaks {leaks}")
    return Check("tool_call", PASS, f"{len(calls)} well-formed {PROBE_TOOL_NAME} call(s), "
                 f"finish_reason={finish_reason_of(response)}", {"tool_calls": calls})


def probe_request() -> dict[str, Any]:
    return {"tools": [PROBE_TOOL], "tool_choice": "auto", "temperature": 0, "max_tokens": 256,
            "messages": [{"role": "user", "content": f"Call the {PROBE_TOOL_NAME} tool for UTC."}]}


def roundtrip_request(call: Mapping[str, Any]) -> dict[str, Any]:
    return {"tools": [PROBE_TOOL], "tool_choice": "auto", "temperature": 0, "max_tokens": 256,
            "messages": [
                {"role": "user", "content": f"Call the {PROBE_TOOL_NAME} tool for UTC, then tell me the time."},
                {"role": "assistant", "content": None, "tool_calls": [dict(call)]},
                {"role": "tool", "tool_call_id": call.get("id", "call_0"), "name": PROBE_TOOL_NAME,
                 "content": json.dumps({"time": "2026-09-30T08:15:00Z"})}]}


def filler(words: int, seed: int = 1) -> str:
    """Deterministic, non-repeating text of about `words` words (a probe input, not language logic)."""
    vocabulary = ("harbour lantern copper river meadow signal winter orchard granite ferry window "
                  "compass valley ember thistle quarry beacon marble cedar tundra anchor prairie "
                  "lagoon cinder falcon saddle mosaic glacier pepper thunder velvet hollow trellis "
                  "juniper basalt ribbon sparrow monsoon kettle lattice harvest dune sextant bramble").split()
    state = seed
    out = []
    for index in range(words):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(vocabulary[state % len(vocabulary)])
        if index % 17 == 16:
            out[-1] += "."
    return " ".join(out)


def prompt_for(words: int) -> list[dict[str, str]]:
    return [{"role": "system", "content": "You are a throughput probe. Read the document, then follow the last line."},
            {"role": "user", "content": "DOCUMENT\n" + filler(words) + "\n\nNow count from 1 to 100, separated by spaces, "
             "and write nothing else."}]


def calibrate(client: Client, target_tokens: int) -> tuple[list[dict[str, str]], int]:
    """Word count whose prompt tokenizes to within ~3% below `target_tokens` (uses POST /tokenize)."""
    words = max(64, int(target_tokens / 1.6))
    tokens = 0
    for _ in range(6):
        status, response = client.request("POST", "/tokenize", {"content": json.dumps(prompt_for(words))})
        if status != 200 or not isinstance(response, Mapping) or "tokens" not in response:
            return prompt_for(words), 0  # no /tokenize: the server's own usage numbers are reported instead
        tokens = len(response["tokens"])
        if 0.97 * target_tokens <= tokens <= target_tokens:
            break
        words = max(64, int(words * target_tokens / max(tokens, 1) * 0.985))
    return prompt_for(words), tokens


def vram_used_gib() -> float | None:
    """AMD GPU memory in use, from sysfs (only meaningful on hephaistos)."""
    best = None
    for path in sorted(Path("/sys/class/drm").glob("card[0-9]*/device/mem_info_vram_used")):
        try:
            total = int((path.parent / "mem_info_vram_total").read_text())
            used = int(path.read_text())
        except (OSError, ValueError):
            continue
        if best is None or total > best[0]:
            best = (total, used)
    return round(best[1] / 2**30, 2) if best else None


# ------------------------------------------------------------------------------------------- the checks


def run_checks(client: Client, *, min_ctx: int, sizes: tuple[int, ...], min_decode_tps: float | None,
               min_prefill_tps: float | None, vram: bool, log: Callable[[str], None] = lambda _: None) -> list[Check]:
    results: list[Check] = []

    def record(check: Check) -> Check:
        results.append(check)
        log(f"{check.status:4}  {check.name:<18} {check.detail}")
        return check

    def with_vram(check: Check) -> Check:
        if vram:
            used = vram_used_gib()
            if used is not None:
                check.data["vram_used_gib"] = used
                check.detail += f" [vram {used} GiB in use]"
        return check

    status, _ = client.request("GET", "/health", auth=False, timeout=15)
    record(Check("health", PASS if status == 200 else FAIL, f"GET /health -> HTTP {status}"))
    if status != 200:
        return results

    refused = {}
    refused["/v1/models"] = client.request("GET", "/v1/models", auth=False, timeout=15)[0]
    refused["/v1/chat/completions"] = client.request(
        "POST", "/v1/chat/completions", {"model": client.model, "max_tokens": 1,
                                         "messages": [{"role": "user", "content": "x"}]}, auth=False, timeout=15)[0]
    open_paths = [path for path, code in refused.items() if code not in (401, 403)]
    record(Check("auth", FAIL if open_paths else PASS,
                 f"served without a key: {open_paths}" if open_paths else
                 "no key -> " + ", ".join(f"{p} {c}" for p, c in refused.items())))

    status, models = client.request("GET", "/v1/models", timeout=30)
    ids = [m.get("id") for m in models.get("data", [])] if isinstance(models, Mapping) else []
    record(Check("models", PASS if client.model in ids else FAIL,
                 f"/v1/models -> {ids}" if status == 200 else f"/v1/models -> HTTP {status}: {str(models)[:200]}"))

    status, props = client.request("GET", "/props", timeout=30)
    generation = props.get("default_generation_settings", {}) if isinstance(props, Mapping) else {}
    n_ctx = generation.get("n_ctx") if isinstance(generation, Mapping) else None
    slots = props.get("total_slots") if isinstance(props, Mapping) else None
    if isinstance(n_ctx, int):
        record(Check("context", PASS if n_ctx >= min_ctx else FAIL,
                     f"per-slot context {n_ctx} tokens ({slots} slots); need >= {min_ctx}",
                     {"n_ctx": n_ctx, "total_slots": slots,
                      "build_info": props.get("build_info") if isinstance(props, Mapping) else None}))
    else:
        record(Check("context", WARN, f"/props (HTTP {status}) gave no n_ctx; the largest throughput probe "
                     "still proves the capacity it reaches"))

    status, response = client.chat(probe_request(), timeout=180)
    tool = record(with_vram(judge_tool_call(status, response)))
    if tool.status == PASS:
        call = tool.data["tool_calls"][0]
        status, response = client.chat(roundtrip_request(call), timeout=180)
        message = message_of(response)
        content = message.get("content") or ""
        problems = []
        if status != 200:
            problems.append(f"HTTP {status}: {str(response)[:200]}")
        elif message.get("tool_calls"):
            problems.append("the model called the tool again instead of answering")
        elif not content.strip():
            problems.append("empty answer after the tool result")
        problems += [f"leaks {m}" for m in leaked_markers(content)]
        record(Check("tool_roundtrip", FAIL if problems else PASS,
                     "; ".join(problems) if problems else f"answered after the tool result: {content.strip()[:80]!r}"))
    else:
        record(Check("tool_roundtrip", WARN, "skipped: the tool call failed"))

    leaks: list[str] = []
    extra = {}
    for label, kwargs in (("default", {}), ("thinking", {"chat_template_kwargs": {"enable_thinking": True}})):
        status, response = client.chat({"tools": [PROBE_TOOL], "tool_choice": "auto", "temperature": 0,
                                        "max_tokens": 700,
                                        "messages": [{"role": "system", "content": "You are Lilith, a concise assistant."},
                                                     {"role": "user", "content": "In one sentence, what is 17 times 23?"}],
                                        **kwargs}, timeout=300)
        message = message_of(response)
        if status != 200:
            leaks.append(f"{label}: HTTP {status}: {str(response)[:200]}")
            continue
        found = leaked_markers(message.get("content"))
        if found:
            leaks.append(f"{label}: content carries {found}")
        if (message.get("content") or "").lstrip().startswith("thought\n"):
            leaks.append(f"{label}: content starts with the raw `thought` channel")
        extra[label] = {"content_chars": len(message.get("content") or ""),
                        "reasoning_chars": len(message.get("reasoning_content") or message.get("reasoning") or "")}
    if leaks:
        record(Check("reasoning_leak", FAIL, "; ".join(leaks), extra))
    elif extra.get("thinking", {}).get("reasoning_chars", 0) == 0:
        record(Check("reasoning_leak", WARN, "no leak, but thinking mode produced no reasoning_content, so "
                     "routing into reasoning_content is unproven", extra))
    else:
        record(Check("reasoning_leak", PASS, f"no marker in content; thinking mode routed "
                     f"{extra['thinking']['reasoning_chars']} chars to reasoning_content", extra))

    for size in sizes:
        name = f"throughput_{size}"
        messages, estimated = calibrate(client, size)
        started = time.monotonic()
        status, response = client.chat({"messages": messages, "max_tokens": 128, "temperature": 0,
                                        "cache_prompt": False}, timeout=client.timeout)
        wall = time.monotonic() - started
        if status != 200 or not isinstance(response, Mapping):
            record(Check(name, FAIL, f"HTTP {status}: {str(response)[:300]}"))
            continue
        usage = response.get("usage", {})
        timings = response.get("timings", {}) if isinstance(response.get("timings"), Mapping) else {}
        prompt_tokens = usage.get("prompt_tokens") or timings.get("prompt_n")
        completion = usage.get("completion_tokens") or timings.get("predicted_n")
        prefill = timings.get("prompt_per_second")
        decode = timings.get("predicted_per_second")
        detail = (f"{prompt_tokens} prompt tokens: prefill {prefill:.1f} tok/s, " if prefill else
                  f"{prompt_tokens} prompt tokens: ") + \
                 (f"decode {decode:.1f} tok/s" if decode else "decode n/a") + f" ({completion} generated, {wall:.1f}s wall)"
        status_word = PASS
        if min_decode_tps is not None and decode is not None and decode < min_decode_tps:
            status_word, detail = WARN, detail + f"; below the {min_decode_tps} tok/s decode floor"
        if min_prefill_tps is not None and prefill is not None and prefill < min_prefill_tps:
            status_word, detail = WARN, detail + f"; below the {min_prefill_tps} tok/s prefill floor"
        if isinstance(prompt_tokens, int) and prompt_tokens < 0.9 * size:
            status_word, detail = WARN, detail + f"; the prompt is well under the {size}-token target"
        leaked = leaked_markers(message_of(response).get("content"))
        if leaked:
            status_word, detail = FAIL, detail + f"; content leaks {leaked}"
        record(with_vram(Check(name, status_word, detail, {
            "target_tokens": size, "calibrated_tokens": estimated, "prompt_tokens": prompt_tokens,
            "completion_tokens": completion, "prefill_tps": prefill, "decode_tps": decode, "wall_seconds": round(wall, 2)})))
    return results


def read_key(key_file: Path | None) -> str | None:
    if key_file is not None:
        try:
            key = key_file.read_text().strip()
        except OSError as error:
            raise SystemExit(f"verify: error[lilith::heph::key]: cannot read {key_file}: {error}\n"
                             "  help: on hephaistos the key is /etc/lilith-serve/api-key (root/lilith-serve only); "
                             "on the Mac it is $ROOT/secrets/lilith-openai-key")
        return key or None
    return os.environ.get("LILITH_SERVE_API_KEY", "").strip() or None


def main(argv: list[str], *, opener: Callable[..., Any] = urllib.request.urlopen) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("LILITH_SERVE_BASE_URL", "http://192.168.1.114:8080"),
                        help="server root, without /v1 (default $LILITH_SERVE_BASE_URL or http://192.168.1.114:8080)")
    parser.add_argument("--key-file", type=Path, help="file holding the API key (else $LILITH_SERVE_API_KEY)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--min-ctx", type=int, default=MIN_CTX)
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)),
                        help="throughput prompt sizes in tokens, comma-separated; empty skips them")
    parser.add_argument("--min-decode-tps", type=float, default=None)
    parser.add_argument("--min-prefill-tps", type=float, default=None)
    parser.add_argument("--vram", action="store_true", help="record GPU memory in use (run on hephaistos)")
    parser.add_argument("--json", type=Path, help="write the full record here")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args(argv)

    key = read_key(args.key_file)
    if key is None:
        print("verify: error[lilith::heph::key]: no API key\n  help: pass --key-file PATH or set LILITH_SERVE_API_KEY "
              "(never put the key on the command line)", file=sys.stderr)
        return 2
    sizes = tuple(int(part) for part in args.sizes.split(",") if part.strip())
    client = Client(args.base_url, key, args.model, timeout=args.timeout, opener=opener)
    print(f"verifying {args.base_url} (model {args.model})", file=sys.stderr)
    try:
        results = run_checks(client, min_ctx=args.min_ctx, sizes=sizes, min_decode_tps=args.min_decode_tps,
                             min_prefill_tps=args.min_prefill_tps, vram=args.vram,
                             log=lambda line: print(line, file=sys.stderr))
    except Unreachable as error:
        print(f"verify: error[lilith::heph::unreachable]: {error}\n  help: on hephaistos: systemctl status "
              "lilith-serve; journalctl -u lilith-serve -n 50; the server binds only the LAN address in "
              "/etc/lilith-serve/server.env", file=sys.stderr)
        return 2
    record = {"schema": "lilith/hephaistos-verify@1", "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "base_url": args.base_url, "model": args.model, "from_host": socket.gethostname(),
              "checks": [dataclasses.asdict(c) for c in results]}
    text = client.scrub(json.dumps(record, indent=2, sort_keys=True, default=str))
    if args.json:
        args.json.write_text(text + "\n")
    failed = [c.name for c in results if c.status == FAIL]
    print(f"{'FAILED: ' + ', '.join(failed) if failed else 'all checks passed'}"
          f" ({sum(c.status == WARN for c in results)} warning(s))", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
