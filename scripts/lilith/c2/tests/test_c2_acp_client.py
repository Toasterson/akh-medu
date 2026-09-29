import json
import socket
import threading

import pytest

from c2 import acp_client
from c2.errors import AcpUnavailableError, TranscriptsUnavailableError

TOKEN = "t0ken-t0ken-t0ken-t0ken-t0ken"


class FakeCore:
    """A one-connection-at-a-time ACP/TCP core: newline JSON-RPC, bearer check on `initialize`."""

    def __init__(self, turns=None, list_error=None, token=TOKEN):
        self.turns = turns or []
        self.list_error = list_error
        self.token = token
        self.requests = []
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(4)
        self.address = "127.0.0.1:%d" % self.server.getsockname()[1]
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            with conn, conn.makefile("rwb") as stream:
                for raw in stream:
                    request = json.loads(raw)
                    self.requests.append(request)
                    stream.write(json.dumps(self.respond(request)).encode() + b"\n")
                    stream.flush()

    def respond(self, request):
        base = {"jsonrpc": "2.0", "id": request["id"]}
        if request["method"] == "initialize":
            if request["params"]["_meta"]["authorization"] != f"Bearer {self.token}":
                return {**base, "error": {"code": 403, "message": "invalid credentials", "data": {"status": 403}}}
            return {**base, "result": {"protocolVersion": 1}}
        if request["method"] == "_akh/transcripts/list":
            if self.list_error:
                return {**base, "error": self.list_error}
            after = request["params"]["after_seq"]
            page = [t for t in self.turns if t["seq"] > after][: request["params"]["limit"]]
            return {**base, "result": {"turns": page, "count": len(page),
                                       "next_after_seq": page[-1]["seq"] if page else after}}
        return {**base, "error": {"code": -32601, "message": "Method not found", "data": {"message": "unknown"}}}

    def close(self):
        self.server.close()


@pytest.fixture
def core():
    made = []

    def make(**kw):
        made.append(FakeCore(**kw))
        return made[-1]

    yield make
    for c in made:
        c.close()


def test_list_turns_pages_by_watermark(core):
    fake = core(turns=[{"seq": n, "payload": {}} for n in range(1, 8)])
    got = list(acp_client.list_turns(fake.address, TOKEN, "lilith", page_limit=3))
    assert [t["seq"] for t in got] == list(range(1, 8))
    lists = [r for r in fake.requests if r["method"] == "_akh/transcripts/list"]
    assert [r["params"]["after_seq"] for r in lists] == [0, 3, 6, 7]
    assert all(r["params"]["workspace"] == "lilith" for r in lists)


def test_a_wrong_token_is_refused_without_echoing_it(core):
    fake = core(turns=[])
    with pytest.raises(AcpUnavailableError) as raised:
        list(acp_client.list_turns(fake.address, "another-token-another-token-xx", "lilith"))
    assert "another-token" not in str(raised.value) and TOKEN not in str(raised.value)
    assert raised.value.exit_code == 4


def test_an_old_core_says_which_method_is_missing(core):
    fake = core(list_error={"code": -32601, "message": "Method not found",
                            "data": {"message": "unknown _akh data-plane method: _akh/transcripts/list"}})
    with pytest.raises(TranscriptsUnavailableError, match="older than ADR 116 C2"):
        list(acp_client.list_turns(fake.address, TOKEN, "lilith"))


def test_a_core_without_a_turn_store_surfaces_its_own_diagnostic(core):
    fake = core(list_error={"code": -32601, "message": "Method not found", "data": {
        "message": "transcript store unavailable for 'lilith'", "code": acp_client.UNAVAILABLE_CODE}})
    with pytest.raises(TranscriptsUnavailableError, match="transcript store unavailable"):
        list(acp_client.list_turns(fake.address, TOKEN, "lilith"))


def test_an_unreachable_core_is_an_acp_error():
    with pytest.raises(AcpUnavailableError, match="cannot reach"):
        list(acp_client.list_turns("127.0.0.1:1", TOKEN, "lilith"))


def test_token_file_reading(tmp_path):
    good = tmp_path / "k"
    good.write_text("\n" + TOKEN + "\nrotated-second-key\n")
    assert acp_client.read_token(good) == TOKEN
    with pytest.raises(AcpUnavailableError):
        acp_client.read_token(tmp_path / "missing")
    empty = tmp_path / "e"
    empty.write_text("\n")
    with pytest.raises(AcpUnavailableError):
        acp_client.read_token(empty)


def test_endpoint_parsing():
    assert acp_client.split_endpoint("tcp://127.0.0.1:18444") == ("127.0.0.1", 18444)
    with pytest.raises(AcpUnavailableError):
        acp_client.split_endpoint("nonsense")
