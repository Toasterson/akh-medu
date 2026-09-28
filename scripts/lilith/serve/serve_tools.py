#!/usr/bin/env python3
"""Resolve vLLM's tool-call parser for the Lilith serve pod (Anima WI 01a0e270-314e, Lilith C2).

Lilith's akh talks to vLLM through the ``openai`` backing with native ``tool_calls``, so the
server must run with ``--enable-auto-tool-choice --tool-call-parser <name>``. Serving without
tools is never a silent fallback.

    serve_tools.py resolve --requested auto|<name> --help-file <vllm serve --help output>
                           --chat-template <path> [--output selection.json]

Which parsers exist is read from the *installed* vLLM, never assumed:

1. ``ToolParserManager`` (``vllm.tool_parsers`` in current vLLM,
   ``vllm.entrypoints.openai.tool_parsers`` in older releases): the eagerly registered
   ``tool_parsers`` plus the ``lazy_parsers`` names.
2. Otherwise the ``--tool-call-parser {a,b,...}`` choices in ``vllm serve --help`` output.

Choice:

* ``--requested <name>`` (``LILITH_SERVE_TOOL_PARSER``): used when that name is registered,
  otherwise a refusal listing the registered names.
* ``--requested auto`` (default): the first of ``PREFERRED`` that is registered. ``gemma4`` is
  vLLM's Gemma 4 parser (``Gemma4EngineToolParser``, native ``<|tool_call>call:...`` syntax,
  which the parent's own ``chat_template.jinja`` emits). Nothing else is auto-selected:
  ``pythonic`` (vLLM's documented choice for Gemma 3) only parses Python-list calls produced by
  a *pythonic* chat template, which the pinned parent template is not, so picking it silently
  would serve a model whose calls never parse. With no preferred parser registered the
  resolution fails and names the options.

The chat template is the parent's pinned ``chat_template.jinja`` (``LILITH_SERVE_CHAT_TEMPLATE``
overrides it; an override must name a template that matches the chosen parser). It must exist;
its sha256 is recorded next to the parser.

Prints one ``lilith/serve-tools@1`` JSON object. Exit 0 on a choice, 2 on a refusal
(``[code] message`` + ``help:`` on stderr).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "lilith/serve-tools@1"
PREFERRED = ("gemma4",)
MANAGER_MODULES = ("vllm.tool_parsers", "vllm.entrypoints.openai.tool_parsers")


class ToolParserError(Exception):
    code = "lilith::serve::tool_parser"

    def __init__(self, message: str, *, help: str, code: str | None = None):
        super().__init__(message)
        self.help = help
        if code:
            self.code = code

    def render(self) -> str:
        return f"[{self.code}] {self}\n  help: {self.help}"


@dataclasses.dataclass(frozen=True)
class Registry:
    names: tuple[str, ...]
    source: str


@dataclasses.dataclass(frozen=True)
class ToolSelection:
    parser: str
    requested: str
    reason: str
    registry_source: str
    available: tuple[str, ...]
    chat_template: str
    chat_template_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA} | dataclasses.asdict(self) | {"available": list(self.available),
                                                                 "enable_auto_tool_choice": True}


def registry_from_manager() -> Registry | None:
    for module_name in MANAGER_MODULES:
        try:
            module = importlib.import_module(module_name)
            manager = getattr(module, "ToolParserManager")
        except Exception:  # noqa: BLE001  (any import failure means "try the next source")
            continue
        names = set(getattr(manager, "tool_parsers", {}) or {}) | set(getattr(manager, "lazy_parsers", {}) or {})
        if names:
            return Registry(tuple(sorted(names)), f"{module_name}.ToolParserManager")
    return None


def registry_from_help(help_text: str) -> Registry | None:
    match = re.search(r"--tool-call-parser\s*(?:TOOL_CALL_PARSER\s*)?\{([^}]*)\}", help_text)
    if not match:
        return None
    names = tuple(sorted({n.strip() for n in match.group(1).split(",") if n.strip()}))
    return Registry(names, "vllm serve --help (--tool-call-parser choices)") if names else None


def choose(requested: str, registry: Registry, preferred: Sequence[str] = PREFERRED) -> tuple[str, str]:
    listing = ", ".join(registry.names)
    if requested != "auto":
        if requested in registry.names:
            return requested, f"requested by LILITH_SERVE_TOOL_PARSER={requested}"
        raise ToolParserError(f"tool parser {requested!r} is not registered in this vLLM (available: {listing})",
                              help="set LILITH_SERVE_TOOL_PARSER to one of the available names, or unset it "
                                   "to auto-select the Gemma 4 parser")
    for name in preferred:
        if name in registry.names:
            return name, f"auto: {name} is the Gemma 4 parser registered in this vLLM"
    raise ToolParserError(
        f"no Gemma 4 tool parser ({', '.join(preferred)}) is registered in this vLLM (available: {listing})",
        help="options: (1) install a payload whose lock pins a vLLM with the gemma4 parser; (2) set "
             "LILITH_SERVE_TOOL_PARSER=<name> together with LILITH_SERVE_CHAT_TEMPLATE=<the template that "
             "parser expects> (pythonic needs a pythonic Gemma template, not the pinned parent template). "
             "The pod never serves without tool calling")


def resolve(requested: str, help_text: str, chat_template: Path) -> ToolSelection:
    registry = registry_from_manager() or registry_from_help(help_text)
    if registry is None:
        raise ToolParserError("cannot list the tool parsers of the installed vLLM (no ToolParserManager, "
                              "no --tool-call-parser choices in `vllm serve --help`)",
                              code="lilith::serve::tool_parser_registry",
                              help="check the env's vLLM install (bootstrap --check); the pod refuses to "
                                   "serve without tool calling")
    parser, reason = choose(requested, registry)
    if not chat_template.is_file():
        raise ToolParserError(f"chat template {chat_template} does not exist",
                              code="lilith::serve::chat_template",
                              help="the pinned parent ships chat_template.jinja; re-verify the parent "
                                   "(bootstrap --check) or point LILITH_SERVE_CHAT_TEMPLATE at the template")
    digest = hashlib.sha256(chat_template.read_bytes()).hexdigest()
    return ToolSelection(parser=parser, requested=requested, reason=reason, registry_source=registry.source,
                         available=registry.names, chat_template=str(chat_template), chat_template_sha256=digest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    res = sub.add_parser("resolve")
    res.add_argument("--requested", default="auto")
    res.add_argument("--help-file", type=Path, required=True)
    res.add_argument("--chat-template", type=Path, required=True)
    args = parser.parse_args(argv)
    help_text = args.help_file.read_text(errors="replace") if args.help_file.is_file() else ""
    try:
        selection = resolve(args.requested.strip() or "auto", help_text, args.chat_template)
    except ToolParserError as error:
        print(error.render(), file=sys.stderr)
        return 2
    print(json.dumps(selection.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
