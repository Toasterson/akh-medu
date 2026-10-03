import json
import re
import sys
from pathlib import Path

import pytest

LILITH = Path(__file__).resolve().parents[2]
if str(LILITH) not in sys.path:
    sys.path.insert(0, str(LILITH))

from program import schema as S  # noqa: E402

MARK = re.compile(r"<[^<>\n]{1,40}>")


class FakeCodec:
    """A character-level stand-in for the pinned Gemma codec: `<...>` markers are single tokens."""

    end_of_turn = "<EOT>"
    tool_stop = "<STOP>"
    info = {"parent_revision": "fake", "tokenizer_pins": {"tokenizer.json": "0", "tokenizer_config.json": "0",
                                                          "chat_template.jinja": "0"}, "enable_thinking": False}

    def context(self, messages, tools):
        out = ["<bos>"]
        if tools:
            out.append("<tools:%s>" % ",".join(t["function"]["name"] for t in tools))
        for m in messages:
            out.append(f"<{m['role']}>{m.get('content') or ''}")
        return "".join(out) + "<gen>"

    def call_text(self, name, arguments):
        return f"<call:{name}>{json.dumps(arguments, sort_keys=True)}<endcall>"

    def response_text(self, name, body):
        return f"<resp:{name}>{body}<endresp>"

    def encode(self, text):
        ids, pos = [], 0
        for m in MARK.finditer(text):
            ids += [ord(c) for c in text[pos:m.start()]]
            ids.append(100000 + sum(ord(c) * (i + 1) for i, c in enumerate(m.group())) % 50000)
            pos = m.end()
        ids += [ord(c) for c in text[pos:]]
        return ids


@pytest.fixture
def codec():
    return FakeCodec()


def rec(**kw):
    base = dict(source="kata_drill", source_ref="t", stage="white", families=["silence"], language="en",
                messages=[S.system("sys"), S.user("hi"), S.assistant("hello")])
    base.update(kw)
    return S.make_record(**base)


@pytest.fixture
def make():
    return rec
