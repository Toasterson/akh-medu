import importlib.util
import json
import sys
from pathlib import Path

P = Path(__file__).with_name("cos_announce_relay.py")
spec = importlib.util.spec_from_file_location("cos_announce_relay", P)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


# --- fixtures ------------------------------------------------------------------

def announcement(*, tag="a", status="succeeded", contract=m.ANNOUNCEMENT_CONTRACT):
    return {
        "contract": contract,
        "akh": "lilith",
        "announced_at": 1_800_000_000,
        "tool": f"tool-{tag}",
        "arguments": {"path": f"/tmp/{tag}"},
        "action": {"kind": "merge", "change": f"PR#{tag}"},
        "receipts": [
            {"kind": "pull-request", "reference": f"https://code.example/pr/{tag}", "observed_at": 1_800_000_000},
        ],
        "reason": f"reason {tag}",
        "rollback": {"kind": "revert", "steps": f"git revert {tag}"},
        "rule_checks": [{"rule": "receipts-present", "evidence": "1 receipt(s)"}],
        "outcome": {"status": status, "detail": f"detail {tag}"},
    }


def spend_reserved(tag="s"):
    return {
        "contract": "lilith/cos-spend-reserved@1",
        "reservation": f"r-{tag}",
        "tool": "spend_tool",
        "amount": {"minor_units": 100, "currency": "USD"},
        "reserved_at": 1_800_000_000,
    }


def write_log(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


class RecordingTransport:
    """Fake board transport. `responses` is a list of (status, body) popped in
    order per call; the last entry repeats once exhausted."""

    def __init__(self, responses=None):
        self.responses = list(responses) if responses else [(201, b'{"id":"m1"}')]
        self.calls: list[dict] = []

    def post(self, url, headers, body):
        self.calls.append({"url": url, "headers": dict(headers), "body": json.loads(body)})
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


def no_sleep(_seconds):
    return None


def run(log, state, quarantine, transport, thread_id="thread-1", token="secret-token-abc", retry=None):
    return m.run_relay(
        log_path=log,
        state_path=state,
        quarantine_path=quarantine,
        board_url="http://127.0.0.1:18089",
        thread_id=thread_id,
        token=token,
        transport=transport,
        retry=retry or m.RetryPolicy(max_attempts=3, initial_delay=0.001, max_delay=0.001),
        sleep=no_sleep,
    )


# --- offset persistence across restart ------------------------------------------

def test_offset_persists_across_restart(tmp_path):
    log = tmp_path / "cos-log.jsonl"
    state = tmp_path / "cos-log.jsonl.relay-state.json"
    quarantine = tmp_path / "cos-log.jsonl.relay-quarantine.jsonl"
    write_log(log, [announcement(tag="1")])
    transport = RecordingTransport()

    result1 = run(log, state, quarantine, transport)
    assert result1.posted == 1
    assert len(transport.calls) == 1
    offset_after_first = json.loads(state.read_text())["byte_offset"]
    assert offset_after_first == log.stat().st_size

    # "Restart": load state fresh (a brand-new process would too) and add a
    # second announcement. Only the new line must be relayed.
    write_log(log, [announcement(tag="2")])
    result2 = run(log, state, quarantine, transport)
    assert result2.posted == 1
    assert len(transport.calls) == 2
    assert json.loads(state.read_text())["byte_offset"] == log.stat().st_size

    # A third "restart" with nothing new posts nothing.
    result3 = run(log, state, quarantine, transport)
    assert result3.posted == 0
    assert len(transport.calls) == 2


def test_partial_trailing_line_is_not_consumed(tmp_path):
    log = tmp_path / "cos-log.jsonl"
    state = tmp_path / "cos-log.jsonl.relay-state.json"
    quarantine = tmp_path / "cos-log.jsonl.relay-quarantine.jsonl"
    log.write_bytes((json.dumps(announcement(tag="1")) + "\n").encode() + b'{"contract": "lilith/cos-announcement@1", "trunc')
    transport = RecordingTransport()

    result = run(log, state, quarantine, transport)
    assert result.posted == 1
    complete_len = len(json.dumps(announcement(tag="1")) + "\n")
    assert json.loads(state.read_text())["byte_offset"] == complete_len


# --- exactly-once ----------------------------------------------------------------

def test_exactly_once_via_digest_and_idempotency_key(tmp_path):
    log = tmp_path / "cos-log.jsonl"
    state = tmp_path / "cos-log.jsonl.relay-state.json"
    quarantine = tmp_path / "cos-log.jsonl.relay-quarantine.jsonl"
    write_log(log, [announcement(tag="dup")])
    transport = RecordingTransport()

    run(log, state, quarantine, transport)
    first_key = transport.calls[0]["body"]["idempotency_key"]

    # Simulate a state file that was reverted to a stale offset (e.g. an
    # operator restore, or a bug) -- the digest ring must still stop a repost.
    stale = json.loads(state.read_text())
    stale["byte_offset"] = 0
    state.write_text(json.dumps(stale))

    result = run(log, state, quarantine, transport)
    assert result.posted == 0
    assert len(transport.calls) == 1  # no second board write

    # The idempotency key is deterministic from line content, so a genuine
    # board-side retry of the same line would also dedupe there.
    write_log(log, [announcement(tag="dup")])  # identical content, appended again
    # Reset offset to just before this new identical line's actual position
    # is not meaningful here; instead assert the key would match if resent.
    digest = m.hashlib.sha256(json.dumps(announcement(tag="dup")).encode()).hexdigest()
    assert first_key == f"lilith-cos-announce:{digest}"


def test_spend_records_are_skipped_not_posted(tmp_path):
    log = tmp_path / "cos-log.jsonl"
    state = tmp_path / "cos-log.jsonl.relay-state.json"
    quarantine = tmp_path / "cos-log.jsonl.relay-quarantine.jsonl"
    write_log(log, [spend_reserved(), announcement(tag="1"), spend_reserved("t2")])
    transport = RecordingTransport()

    result = run(log, state, quarantine, transport)
    assert result.posted == 1
    assert result.skipped == 2
    assert len(transport.calls) == 1


# --- poison quarantine -----------------------------------------------------------

def test_poison_lines_are_quarantined_and_never_block_the_stream(tmp_path):
    log = tmp_path / "cos-log.jsonl"
    state = tmp_path / "cos-log.jsonl.relay-state.json"
    quarantine = tmp_path / "cos-log.jsonl.relay-quarantine.jsonl"
    with log.open("a", encoding="utf-8") as handle:
        handle.write("not json at all\n")
        handle.write(json.dumps({"contract": "lilith/cos-announcement@1", "akh": "lilith"}) + "\n")  # missing fields
        handle.write(json.dumps(announcement(tag="good")) + "\n")  # must still be relayed
        handle.write(json.dumps({"contract": "unknown/thing@1"}) + "\n")
    transport = RecordingTransport()

    result = run(log, state, quarantine, transport)
    assert result.posted == 1
    assert result.quarantined == 3
    assert len(transport.calls) == 1
    assert transport.calls[0]["body"]["payload"]["text"].count("reason good") == 1

    entries = [json.loads(line) for line in quarantine.read_text().splitlines()]
    assert len(entries) == 3
    reasons = [e["reason"] for e in entries]
    assert any("invalid_json" in r for r in reasons)
    assert any("missing_fields" in r for r in reasons)
    assert any("unknown_contract" in r for r in reasons)
    # The offset moved past every line, including the poisoned ones.
    assert json.loads(state.read_text())["byte_offset"] == log.stat().st_size


# --- secret never in logs ---------------------------------------------------------

def test_secret_never_appears_in_state_quarantine_or_posted_body(tmp_path):
    log = tmp_path / "cos-log.jsonl"
    state = tmp_path / "cos-log.jsonl.relay-state.json"
    quarantine = tmp_path / "cos-log.jsonl.relay-quarantine.jsonl"
    token = "sekrit-xyz-999"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"not json but mentions Bearer {token} inline\n")
    write_log(log, [announcement(tag="1")])
    transport = RecordingTransport()

    run(log, state, quarantine, transport, token=token)

    assert token not in state.read_text()
    assert token not in quarantine.read_text()
    # The board call itself legitimately carries the bearer token in its
    # Authorization header (that's how auth works) -- but never in the body,
    # and never anywhere the relay persists to disk outside that one header.
    for call in transport.calls:
        assert token not in json.dumps(call["body"])


def test_secret_never_appears_in_error_output_on_permanent_failure(tmp_path, capsys):
    log = tmp_path / "cos-log.jsonl"
    state = tmp_path / "cos-log.jsonl.relay-state.json"
    quarantine = tmp_path / "cos-log.jsonl.relay-quarantine.jsonl"
    write_log(log, [announcement(tag="1")])
    token = "sekrit-permanent-000"
    # A 403 body that (adversarially) echoes the Authorization header back,
    # as some misconfigured proxies do.
    transport = RecordingTransport(responses=[(403, f"forbidden: Authorization: Bearer {token}".encode())])

    result = run(log, state, quarantine, transport, token=token)
    assert result.stopped_early
    assert token not in result.error
    assert token not in state.read_text()
    captured = capsys.readouterr()
    assert token not in captured.out
    assert token not in captured.err


# --- retry / backoff on transient failure -----------------------------------------

def test_transient_failure_retries_then_succeeds(tmp_path):
    log = tmp_path / "cos-log.jsonl"
    state = tmp_path / "cos-log.jsonl.relay-state.json"
    quarantine = tmp_path / "cos-log.jsonl.relay-quarantine.jsonl"
    write_log(log, [announcement(tag="1")])
    transport = RecordingTransport(responses=[(503, b"unavailable"), (201, b"{}")])

    result = run(log, state, quarantine, transport)
    assert result.posted == 1
    assert not result.stopped_early
    assert len(transport.calls) == 2


def test_permanent_failure_leaves_offset_before_the_line(tmp_path):
    log = tmp_path / "cos-log.jsonl"
    state = tmp_path / "cos-log.jsonl.relay-state.json"
    quarantine = tmp_path / "cos-log.jsonl.relay-quarantine.jsonl"
    write_log(log, [announcement(tag="1"), announcement(tag="2")])
    transport = RecordingTransport(responses=[(401, b"bad token")])

    result = run(log, state, quarantine, transport)
    assert result.stopped_early
    assert result.posted == 0
    assert json.loads(state.read_text())["byte_offset"] == 0
    # Neither line was posted; a bad, non-retryable status stops before
    # trying more lines rather than skipping ahead.
    assert len(transport.calls) == 1


def test_stopped_relay_resumes_on_next_run(tmp_path):
    log = tmp_path / "cos-log.jsonl"
    state = tmp_path / "cos-log.jsonl.relay-state.json"
    quarantine = tmp_path / "cos-log.jsonl.relay-quarantine.jsonl"
    write_log(log, [announcement(tag="1")])
    failing = RecordingTransport(responses=[(500, b"nope")] * 5)
    result1 = run(log, state, quarantine, failing, retry=m.RetryPolicy(max_attempts=2, initial_delay=0.001, max_delay=0.001))
    assert result1.stopped_early
    assert json.loads(state.read_text())["byte_offset"] == 0

    recovered = RecordingTransport(responses=[(201, b"{}")])
    result2 = run(log, state, quarantine, recovered)
    assert result2.posted == 1
    assert not result2.stopped_early


# --- formatting sanity -------------------------------------------------------------

def test_format_announcement_includes_action_reason_receipts_rollback_links():
    text, summary = m.format_announcement(announcement(tag="fmt"))
    assert "Action: merge PR#fmt" in text
    assert "Reason: reason fmt" in text
    assert "Rollback: revert -- git revert fmt" in text
    assert "[pull-request] https://code.example/pr/fmt" in text
    assert "Links:" in text
    assert "https://code.example/pr/fmt" in text.split("Links:")[1]
    assert "merge" in summary
