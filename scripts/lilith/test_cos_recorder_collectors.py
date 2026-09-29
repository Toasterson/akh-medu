"""Board + fleet collectors and the GET-only HTTP client (Lilith C3, ADR 116)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

P = Path(__file__).with_name("cos_recorder.py")
spec = importlib.util.spec_from_file_location("cos_recorder_collectors_under_test", P)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

TOKEN = "board-secret-token-xyz"


class FakeHttp:
    """Scripted HTTP transport; records every request so tests can assert the methods used."""

    def __init__(self, routes):
        self.routes = routes  # path -> (status, headers, body-object)
        self.requests = []

    def request(self, method, url, headers):
        self.requests.append((method, url, headers))
        path = url.split("//", 1)[1].split("/", 1)[1]
        path = "/" + path
        if path not in self.routes:
            return 404, {}, b'{"error":{"code":"not_found"}}'
        status, hdrs, body = self.routes[path]
        return status, hdrs, json.dumps(body).encode()


def ctx(tmp_path):
    return m.SnapshotContext(root=tmp_path, timestamp="t", previous_work_items={})


def secrets(tmp_path, name="board.json", token=TOKEN):
    path = tmp_path / name
    path.write_text(json.dumps({"bearer_token": token, "agent": {"handle": "h"}}))
    return path


# --- ReadOnlyHttpClient -------------------------------------------------------

@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "post", "HEAD"])
def test_read_only_http_client_refuses_non_get_before_transport(method):
    http = FakeHttp({})
    client = m.ReadOnlyHttpClient("http://board", "tok", http)
    with pytest.raises(m.ReadOnlyHttpRefused):
        client.send(method, "/v1/threads/1/messages")
    assert http.requests == []


@pytest.mark.parametrize("path", ["http://evil/v1/x", "//evil/v1/x", "v1/relative"])
def test_read_only_http_client_refuses_foreign_or_relative_paths(path):
    http = FakeHttp({})
    with pytest.raises(m.ReadOnlyHttpRefused):
        m.ReadOnlyHttpClient("http://board", "tok", http).send("GET", path)
    assert http.requests == []


def test_read_only_http_client_sends_get_with_bearer():
    http = FakeHttp({"/v1/agents": (200, {}, [])})
    body, _ = m.ReadOnlyHttpClient("http://board/", "tok", http).get_json("/v1/agents")
    assert body == []
    assert http.requests[0][0] == "GET" and http.requests[0][1] == "http://board/v1/agents"
    assert http.requests[0][2]["Authorization"] == "Bearer tok"


def test_http_error_is_redacted_and_typed():
    http = FakeHttp({"/v1/agents": (500, {}, {"error": "boom Authorization: Bearer leaked"})})
    with pytest.raises(m.HttpReadError) as info:
        m.ReadOnlyHttpClient("http://board", "tok", http).get_json("/v1/agents")
    assert "leaked" not in str(info.value) and info.value.status == 500


def test_read_bearer_token_json_plain_and_bad(tmp_path):
    assert m.read_bearer_token(secrets(tmp_path)) == TOKEN
    plain = tmp_path / "plain.key"
    plain.write_text("  tok-plain\n")
    assert m.read_bearer_token(plain) == "tok-plain"
    (tmp_path / "empty.json").write_text("{}")
    with pytest.raises(RuntimeError):
        m.read_bearer_token(tmp_path / "empty.json")
    with pytest.raises(RuntimeError):
        m.read_bearer_token(tmp_path / "missing")


# --- Board collector ------------------------------------------------------------

THREADS = [
    {"id": "t-open", "title": "open", "closed_at": None},
    {"id": "t-closed", "title": "closed", "closed_at": "2026-09-01T00:00:00Z"},
]


def board_routes():
    return {
        "/v1/agents": (200, {}, [{"id": "a", "handle": "akh-lilith"}]),
        "/v1/message-types": (200, {}, [{"name": "board/note@v1"}]),
        "/v1/threads": (200, {}, THREADS),
        "/v1/threads/t-open/messages?limit=200": (200, {"next-cursor": "m2"}, [{"message_id": "m1"}, {"message_id": "m2"}]),
        "/v1/threads/t-open/messages?limit=200&after_message_id=m2": (200, {}, [{"message_id": "m3"}]),
    }


def test_board_not_configured_without_secrets_file(tmp_path):
    http = FakeHttp({})
    result = m.BoardCollector(transport=http).collect(ctx(tmp_path))
    assert result.status == "not_configured" and "LILITH_BOARD_SECRETS" in result.extra["reason"]
    assert http.requests == []


def test_board_collects_roster_threads_and_paginated_open_thread_messages_get_only(tmp_path):
    http = FakeHttp(board_routes())
    collector = m.BoardCollector(base_url="http://board", secrets_file=secrets(tmp_path), transport=http)
    result = collector.collect(ctx(tmp_path))
    assert result.status == "ok", result.errors
    assert {r.call for r in result.responses} == {"board_agents", "board_message_types", "board_threads", "board_thread_messages"}
    pages = [r for r in result.responses if r.call == "board_thread_messages"]
    assert [r.args.get("after_message_id") for r in pages] == [None, "m2"]  # forward cursor followed
    assert all(r.args["thread_id"] == "t-open" for r in pages)  # closed thread not read
    assert result.extra["thread_count"] == 2 and result.extra["open_thread_count"] == 1
    assert {method for method, _, _ in http.requests} == {"GET"}


def test_board_token_never_stored(tmp_path):
    http = FakeHttp(board_routes())
    m.BoardCollector(base_url="http://board", secrets_file=secrets(tmp_path), transport=http).collect(ctx(tmp_path))
    for obj in (tmp_path / "objects").glob("*.json"):
        assert TOKEN not in obj.read_text()


def test_board_partial_on_endpoint_error_keeps_going(tmp_path):
    routes = board_routes()
    routes["/v1/agents"] = (403, {}, {"error": "forbidden"})
    result = m.BoardCollector(base_url="http://board", secrets_file=secrets(tmp_path), transport=FakeHttp(routes)).collect(ctx(tmp_path))
    assert result.status == "partial"
    assert any("board_agents" in e and "403" in e for e in result.errors)
    assert any(r.call == "board_threads" for r in result.responses)


def test_board_bad_secrets_file_is_error_not_crash(tmp_path):
    result = m.BoardCollector(secrets_file=tmp_path / "nope.json", transport=FakeHttp({})).collect(ctx(tmp_path))
    assert result.status == "error" and result.responses == []


def test_board_page_cap_reports_partial(tmp_path):
    routes = board_routes()
    routes["/v1/threads/t-open/messages?limit=200&after_message_id=m2"] = (200, {"next-cursor": "m2"}, [{"message_id": "m3"}])
    result = m.BoardCollector(base_url="http://board", secrets_file=secrets(tmp_path), transport=FakeHttp(routes), max_pages=2).collect(ctx(tmp_path))
    assert result.status == "partial" and any("stopped after 2 pages" in e for e in result.errors)


# --- Fleet collector -------------------------------------------------------------

def fleet_config(tmp_path, endpoints):
    path = tmp_path / "fleet.json"
    path.write_text(json.dumps({"endpoints": endpoints}))
    return path


def test_fleet_not_configured_without_config(tmp_path):
    result = m.FleetCollector().collect(ctx(tmp_path))
    assert result.status == "not_configured" and "LILITH_FLEET_CONFIG" in result.extra["reason"]


def test_fleet_fetches_configured_endpoints_get_only(tmp_path):
    key = tmp_path / "operator.key"
    key.write_text("fleet-token\n")
    cfg = fleet_config(tmp_path, [
        {"name": "sessions", "url": "http://127.0.0.1:18431/v1/sessions?compact=true", "secrets_file": str(key)},
        {"name": "deliveries", "url": "http://127.0.0.1:18435/v1/deliveries?pending=true", "secrets_file": str(key)},
    ])
    http = FakeHttp({"/v1/sessions?compact=true": (200, {}, [{"id": "s1"}]), "/v1/deliveries?pending=true": (200, {}, [])})
    result = m.FleetCollector(config_path=cfg, transport=http).collect(ctx(tmp_path))
    assert result.status == "ok", result.errors
    assert [r.args["endpoint"] for r in result.responses] == ["sessions", "deliveries"]
    assert {method for method, _, _ in http.requests} == {"GET"}
    assert http.requests[0][2]["Authorization"] == "Bearer fleet-token"
    for obj in (tmp_path / "objects").glob("*.json"):
        assert "fleet-token" not in obj.read_text()


def test_fleet_partial_when_one_endpoint_fails(tmp_path):
    key = tmp_path / "k"
    key.write_text("t")
    cfg = fleet_config(tmp_path, [
        {"name": "good", "url": "http://h/v1/a", "secrets_file": str(key)},
        {"name": "bad", "url": "http://h/v1/b", "secrets_file": str(key)},
        {"name": "no-secret", "url": "http://h/v1/a", "secrets_file": str(tmp_path / "gone")},
    ])
    result = m.FleetCollector(config_path=cfg, transport=FakeHttp({"/v1/a": (200, {}, {"ok": 1})})).collect(ctx(tmp_path))
    assert result.status == "partial"
    assert [r.args["endpoint"] for r in result.responses] == ["good"]
    assert len(result.errors) == 2
    assert any("'bad'" in e and "404" in e for e in result.errors) and any("'no-secret'" in e for e in result.errors)


def test_fleet_bad_config_is_error(tmp_path):
    bad = tmp_path / "fleet.json"
    bad.write_text("not json")
    assert m.FleetCollector(config_path=bad).collect(ctx(tmp_path)).status == "error"


# --- build_collectors / CLI wiring ---------------------------------------------------

def test_build_collectors_wires_projects_board_and_fleet(tmp_path):
    cs = m.build_collectors(["p1", m.ANIMA_PROJECT_ID], detail_project_ids=["p1"], board_secrets_file=tmp_path / "b", fleet_config=tmp_path / "f")
    assert [c.name for c in cs] == ["anima:p1", "anima", "git", "board", "fleet"]
    assert [c.profile for c in cs[:2]] == ["detail", "full"]
    board = next(c for c in cs if c.name == "board")
    assert board.secrets_file == tmp_path / "b" and board.base_url == m.DEFAULT_BOARD_URL
    assert next(c for c in cs if c.name == "fleet").config_path == tmp_path / "f"


def test_main_passes_env_configuration_to_collectors(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_build(project_ids, detail_project_ids=(), board_secrets_file=None, board_url=m.DEFAULT_BOARD_URL, fleet_config=None):
        seen.update(project_ids=project_ids, detail=list(detail_project_ids), board=board_secrets_file, url=board_url, fleet=fleet_config)
        return ()

    monkeypatch.setattr(m, "build_collectors", fake_build)
    monkeypatch.setenv("LILITH_ANIMA_PROJECT_IDS", "p1,p2")
    monkeypatch.setenv("LILITH_ANIMA_DETAIL_PROJECT_IDS", "p2")
    monkeypatch.setenv("LILITH_BOARD_SECRETS", "/s/board.json")
    monkeypatch.setenv("LILITH_BOARD_URL", "http://b:1")
    monkeypatch.setenv("LILITH_FLEET_CONFIG", "/s/fleet.json")
    assert m.main(["run", "--root", str(tmp_path)]) == 0
    assert seen == {"project_ids": ["p1", "p2"], "detail": ["p2"], "board": Path("/s/board.json"), "url": "http://b:1", "fleet": Path("/s/fleet.json")}
    capsys.readouterr()
