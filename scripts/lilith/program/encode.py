"""Tokenization with loss masks, on the pinned Gemma tokenizer and chat template.

The code path follows `scripts/lilith/history/prepare.py::encode_lesson`: the context (system, tool
declarations, user turns) is rendered by the pinned template up to its generation boundary
(`enable_thinking=False`, so the boundary includes the empty thought channel) and masked 0. What the
model generates after that boundary is assembled in the model's own generation order and masked 1 where
the record supports it:

    [content] <|tool_call>call:..{..}<tool_call|> <|tool_response>      supervised (stop token included)
    response:name{value:..}<tool_response|>                              observation, masked
    ... [content] <turn|>                                                supervised (end of turn)

The call and observation text come from the template itself (probe diffs), so formatting is never
re-implemented here. A record that does not fit the context limit raises `Unfit`; nothing is truncated.
"""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from program.schema import CONTEXT_LIMIT, ProgramError, canonical, trained

PARENT_REVISION = "5926caa4ec0cac5cbfadaf4077420520de1d5205"
TOKENIZER_PINS = {
    "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    "tokenizer_config.json": "09e6222fd7049dae603d0f61a8b10af5a813f22dca10e6cdc6918f76d7661104",
    "chat_template.jinja": "36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0",
}


class ChatCodec(Protocol):
    end_of_turn: str
    tool_stop: str

    def context(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None) -> str: ...
    def call_text(self, name: str, arguments: Any) -> str: ...
    def response_text(self, name: str, body: str) -> str: ...
    def encode(self, text: str) -> list[int]: ...


@dataclasses.dataclass(frozen=True)
class Encoded:
    input_ids: list[int]
    target_mask: list[int]
    prompt_tokens: int

    @property
    def length(self) -> int:
        return len(self.input_ids)

    @property
    def supervised(self) -> int:
        return sum(self.target_mask)

    def to_dict(self) -> dict:
        # The trainer validates masks as booleans (first token masked, ≥1 supervised).
        return {"input_ids": self.input_ids, "target_mask": [bool(m) for m in self.target_mask]}


class Unfit(ProgramError):
    """The record needs more tokens than the context limit; it is reported, never truncated."""

    def __init__(self, length: int, limit: int, prompt_tokens: int):
        super().__init__(f"{length} tokens exceed the {limit}-token context ({prompt_tokens} in the prompt)",
                         help="shorten the source observations upstream or leave the record out; export never truncates")
        self.length, self.limit, self.prompt_tokens = length, limit, prompt_tokens


def split_block(messages: Sequence[Mapping[str, Any]]) -> int:
    """Index where the final model block (a run of assistant and tool messages) begins."""
    k = len(messages)
    while k > 0 and messages[k - 1].get("role") in ("assistant", "tool"):
        k -= 1
    if k == len(messages):
        raise ProgramError("the messages end in a context turn: nothing for the model to generate",
                           help="end the list with the assistant action to learn")
    if k == 0:
        raise ProgramError("the messages have no prompt before the first assistant action")
    if messages[k].get("role") != "assistant":
        raise ProgramError("a tool message cannot start the model block")
    return k


def _encode_block(codec: ChatCodec, block: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    mask: list[int] = []
    names: dict[str, str] = {}
    pending: list[str] = []  # call ids still waiting for their observation
    for i, m in enumerate(block):
        role = m.get("role")
        if role == "assistant":
            if pending:
                raise ProgramError("an assistant message follows tool calls that have no observation")
            text = (m.get("content") or "").strip()
            calls = m.get("tool_calls") or []
            for c in calls:
                fn = c["function"]
                names[c["id"]] = fn["name"]
                pending.append(c["id"])
                text += codec.call_text(fn["name"], fn["arguments"])
            if calls:
                text += codec.tool_stop
            else:
                if not text:
                    raise ProgramError("an assistant answer is empty")
                if i != len(block) - 1:
                    raise ProgramError("an assistant answer without tool calls must close the block")
                text += codec.end_of_turn
            piece = codec.encode(text)
            ids += piece
            mask += [1 if trained(m) else 0] * len(piece)
        elif role == "tool":
            cid = m.get("tool_call_id")
            if cid not in pending:
                raise ProgramError(f"tool message answers unknown call {cid!r}")
            pending.remove(cid)
            piece = codec.encode(codec.response_text(names[cid], m["content"]))
            ids += piece
            mask += [0] * len(piece)
        else:
            raise ProgramError(f"unexpected {role!r} message inside the model block")
    if pending:
        raise ProgramError("tool calls without observations at the end of the block")
    if not block[-1].get("content") and block[-1].get("tool_calls"):
        raise ProgramError("the block must end with an assistant answer")
    return ids, mask


def encode_sft(codec: ChatCodec, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None = None,
               limit: int = CONTEXT_LIMIT) -> Encoded:
    k = split_block(messages)
    prompt = codec.encode(codec.context(messages[:k], tools))
    ids, mask = _encode_block(codec, messages[k:])
    enc = Encoded(prompt + ids, [0] * len(prompt) + mask, len(prompt))
    if enc.length > limit:
        raise Unfit(enc.length, limit, enc.prompt_tokens)
    if not enc.supervised:
        raise ProgramError("no supervised token after masking")
    return enc


def encode_preference(codec: ChatCodec, chosen: Sequence[Mapping[str, Any]], rejected: Sequence[Mapping[str, Any]],
                      tools: Sequence[Mapping[str, Any]] | None = None, limit: int = CONTEXT_LIMIT) -> tuple[Encoded, Encoded]:
    kc, kr = split_block(chosen), split_block(rejected)
    if kc != kr or canonical(list(chosen[:kc])) != canonical(list(rejected[:kr])):
        raise ProgramError("chosen and rejected must share the same prompt (same situation)")
    return encode_sft(codec, chosen, tools, limit), encode_sft(codec, rejected, tools, limit)


# -- the Gemma codec --------------------------------------------------------------------------


class GemmaCodec:
    end_of_turn = "<turn|>"
    tool_stop = "<|tool_response>"
    _HEADER = "<|turn>model\n"

    def __init__(self, tokenizer: Any, template: str, info: Mapping[str, Any] | None = None):
        self.tokenizer, self.template, self.info = tokenizer, template, dict(info or {})
        self._calls: dict[str, str] = {}
        self._responses: dict[str, str] = {}
        eot, stop = self.encode(self.end_of_turn), self.encode(self.tool_stop)
        if len(eot) != 1 or len(stop) != 1:
            raise ProgramError("the pinned tokenizer does not encode the Gemma control tokens as single ids")
        self.info.setdefault("end_of_turn_id", eot[0])
        self.info.setdefault("tool_stop_id", stop[0])

    def _render(self, messages, tools=None, generation=False) -> str:
        return self.tokenizer.apply_chat_template(
            list(messages), chat_template=self.template, tokenize=False, tools=list(tools) if tools else None,
            add_generation_prompt=generation, enable_thinking=False)

    def context(self, messages, tools) -> str:
        return self._render(messages, tools, generation=True)

    def _probe(self, name: str, arguments: Any, body: str | None) -> tuple[str, str]:
        base = [{"role": "user", "content": "x"}]
        call = {"role": "assistant", "content": "", "tool_calls": [{"id": "p", "type": "function",
                                                                    "function": {"name": name, "arguments": arguments}}]}
        r0 = self._render(base)
        r1 = self._render(base + [call])
        head = r0 + self._HEADER
        if not (r1.startswith(head) and r1.endswith(self.tool_stop)):
            raise ProgramError("the pinned template no longer renders a tool call the way the encoder expects")
        call_text = r1[len(head):-len(self.tool_stop)]
        response = ""
        if body is not None:
            r2 = self._render(base + [call, {"role": "tool", "tool_call_id": "p", "content": body}])
            if not r2.startswith(r1):
                raise ProgramError("the pinned template no longer renders a tool response the way the encoder expects")
            response = r2[len(r1):]
        return call_text, response

    def call_text(self, name: str, arguments: Any) -> str:
        return self._probe(name, arguments, None)[0]

    def response_text(self, name: str, body: str) -> str:
        key = canonical([name, body])
        if key not in self._responses:
            self._responses[key] = self._probe(name, {}, body)[1]
        return self._responses[key]

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_gemma_codec(parent: Path) -> GemmaCodec:
    """The parent's tokenizer, verified against the pins `history/prepare.py` tokenized with."""
    from importlib.metadata import version

    from transformers import AutoTokenizer  # heavy; only the real export needs it

    parent = Path(parent)
    for name, pin in TOKENIZER_PINS.items():
        if sha256_file(parent / name) != pin:
            raise ProgramError(f"parent {name} does not match its pin", help="point --parent at the pinned Gemma 4 12B parent")
    tokenizer = AutoTokenizer.from_pretrained(parent, local_files_only=True)
    template = (parent / "chat_template.jinja").read_text()
    info = {"parent_revision": PARENT_REVISION, "tokenizer_pins": dict(TOKENIZER_PINS), "context_limit": CONTEXT_LIMIT,
            "packages": {n: version(n) for n in ("transformers", "tokenizers", "jinja2")}, "enable_thinking": False}
    return GemmaCodec(tokenizer, template, info)
