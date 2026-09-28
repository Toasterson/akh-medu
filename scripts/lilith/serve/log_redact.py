#!/usr/bin/env python3
"""Privacy filter for the Lilith serve pod's vLLM log (Anima WI 01a0e270-314e, Lilith C2).

    vllm serve ... 2>&1 | log_redact.py --dir /workspace/lilith/serve/logs --prefix vllm

Reads the vLLM server's combined output on stdin and appends it to
``<dir>/<prefix>-<UTC date>.log`` with a UTC timestamp per line. The request log that
remains is the uvicorn access line (method, path, status) plus vLLM's periodic
throughput stats; prompt and response bodies never reach the volume:

* vLLM's request logger is off at the source (``serve_pod.sh`` picks the flag the
  installed vLLM supports). This filter is the second line of defence.
* A line carrying a request body marker (vLLM's ``Received request ... prompt: ...``,
  ``prompt_token_ids``, ``Generated response`` and friends) is replaced by a one-line
  ``[redacted: request body]`` stub that keeps the request id when one is present.
* The serve API key (``VLLM_API_KEY`` in this process's environment) is replaced by
  ``[redacted: api key]`` wherever it appears.
* Lines are capped at 2,000 characters.

Also echoes every kept line to stdout so ``serve_pod.log`` shows the same stream.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from pathlib import Path
from typing import Iterable, TextIO

# vLLM request-logger / engine markers that precede request content. These are fixed
# strings of vLLM's own log format, not natural-language matching.
BODY_MARKERS = (
    "Received request",
    "prompt:",
    "prompt_token_ids",
    "prompt_embeds",
    "Generated response",
    "output_token_ids",
    "'messages':",
    '"messages":',
    "multi_modal_data",
)
_REQUEST_ID = re.compile(r"\b((?:chatcmpl|cmpl|embd|req)-[0-9A-Za-z_-]{6,})")
MAX_LINE = 2000


def redact_line(line: str, secrets: Iterable[str] = ()) -> str:
    for secret in secrets:
        if secret:
            line = line.replace(secret, "[redacted: api key]")
    if any(marker in line for marker in BODY_MARKERS):
        match = _REQUEST_ID.search(line)
        return "[redacted: request body]" + (f" id={match.group(1)}" if match else "")
    if len(line) > MAX_LINE:
        return line[:MAX_LINE] + f" [truncated {len(line) - MAX_LINE} chars]"
    return line


def run(stream: TextIO, directory: Path, prefix: str, echo: TextIO | None, secrets: tuple[str, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    handle = None
    day = None
    try:
        for raw in stream:
            now = dt.datetime.now(dt.timezone.utc)
            if now.date() != day:
                if handle is not None:
                    handle.close()
                day = now.date()
                path = directory / f"{prefix}-{day.isoformat()}.log"
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                handle = os.fdopen(fd, "a", buffering=1)
            line = f"{now.strftime('%Y-%m-%dT%H:%M:%SZ')} {redact_line(raw.rstrip(chr(10)), secrets)}"
            handle.write(line + "\n")
            if echo is not None:
                echo.write(line + "\n")
                echo.flush()
    finally:
        if handle is not None:
            handle.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--prefix", default="vllm")
    parser.add_argument("--no-echo", action="store_true")
    args = parser.parse_args(argv)
    secrets = tuple(s for s in (os.environ.get("VLLM_API_KEY", ""),) if len(s) >= 8)
    run(sys.stdin, args.dir, args.prefix, None if args.no_echo else sys.stdout, secrets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
