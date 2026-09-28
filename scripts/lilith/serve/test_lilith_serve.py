"""Lilith C2 serving tests (Anima WI 01a0e270-314e): CPU only, no Runpod, no GPU, no network.

A local HTTP server mocks the Runpod REST + GraphQL APIs and the pod's vLLM proxy
(/health, /metrics); a fake `op` script stands in for op-service-account. Covered:
budget refusal, idle stop, session cap, stable-URL bookkeeping across stop/start (and its
one exception, recreate), zero-GPU resume, no secret leakage into logs/state/argv, plus the
on-pod pieces (serve_pod.sh with a fake vLLM, serve_adapter.py, log_redact.py).

    python3 -m unittest scripts/lilith/serve/test_lilith_serve.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import http.server
import io
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HERE = Path(__file__).resolve().parent
LILITH = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LILITH))

import lilith_serve as ls  # noqa: E402
import log_redact  # noqa: E402
import serve_adapter  # noqa: E402

RUNPOD_KEY = "rpa_FAKE_runpod_key_0123456789abcdefXYZ"
SERVE_KEY_PREEXISTING = "serve-key-preexisting-0123456789abcdefghijklmnopqrstuvwxyz"

FAKE_OP = r'''#!/usr/bin/env python3
import json, os, re, sys
db_path = os.environ["FAKE_OP_DB"]
with open(os.environ["FAKE_OP_ARGV"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\n")
db = json.load(open(db_path)) if os.path.exists(db_path) else {}
args = sys.argv[1:]
def save():
    json.dump(db, open(db_path, "w"))
def assign(item, pairs):
    for pair in pairs:
        m = re.match(r"^([^\[=]+)(\[[a-z]+\])?=(.*)$", pair)
        if m:
            item[m.group(1)] = m.group(3)
if args[0] == "read":
    ref = [a for a in args[1:] if a.startswith("op://")][0]
    _, _, vault, item, field = ref.split("/", 4)
    value = db.get(item, {}).get(field)
    if value is None:
        sys.exit(1)
    sys.stdout.write(value)
elif args[:2] == ["item", "get"]:
    item = db.get(args[2])
    if item is None:
        sys.exit(1)
    fields = [{"label": k, "type": "CONCEALED" if k == "credential" else "STRING", "value": v} for k, v in item.items()]
    print(json.dumps({"title": args[2], "fields": fields}))
elif args[:2] == ["item", "create"]:
    if "--title" in args:
        title = args[args.index("--title") + 1]
        item = db.setdefault(title, {})
        assign(item, [a for a in args if "=" in a])
    else:
        stdin = sys.stdin.read()
        with open(os.environ["FAKE_OP_STDIN"], "a") as log:
            log.write(stdin + "\n")
        template = json.loads(stdin)
        db[template["title"]] = {f["label"]: f["value"] for f in template["fields"]}
    save()
elif args[:2] == ["item", "edit"]:
    assign(db.setdefault(args[2], {}), [a for a in args[3:] if "=" in a])
    save()
else:
    sys.exit(2)
'''


class MockCloud:
    """Runpod REST/GraphQL + the pod's vLLM behind the proxy, in one local HTTP server."""

    def __init__(self):
        self.pods: dict[str, dict] = {}
        self.balance = 60.0
        self.start_gpu_count = 1
        self.healthy = True
        self.metrics_ok = True
        self.success_total = 0.0
        self.running = 0.0
        self.fail_stop = False
        self.chat_mode = "tool"  # tool | text | malformed | unparsed | refused
        self.chat_requests: list[dict] = []
        self.requests: list[dict] = []
        self.counter = 0
        cloud = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status, body=None, content_type="application/json"):
                data = b"" if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _handle(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                body = json.loads(raw) if raw else None
                cloud.requests.append({"method": self.command, "path": self.path, "headers": dict(self.headers),
                                       "body": body})
                parts = self.path.strip("/").split("/")
                if parts[:1] == ["graphql"]:
                    return self._reply(200, {"data": {"myself": {"clientBalance": cloud.balance,
                                                                  "currentSpendPerHr": 0, "spendLimit": 80}}})
                if parts[:2] == ["rest", "v1"] and parts[2:3] == ["pods"]:
                    if self.headers.get("Authorization") != f"Bearer {RUNPOD_KEY}":
                        return self._reply(401, {"error": "unauthorized"})
                    rest = parts[3:]
                    if not rest and self.command == "POST":
                        cloud.counter += 1
                        pod_id = f"pod{cloud.counter}abc"
                        cloud.pods[pod_id] = {"id": pod_id, "desiredStatus": "RUNNING", "gpu": {"count": 1},
                                              "costPerHr": 3.49, "publicIp": "203.0.113.9",
                                              "portMappings": {"22": 40022}, "body": body,
                                              "lastStartedAt": "2026-09-30T12:00:00Z"}
                        return self._reply(201, cloud.pods[pod_id])
                    pod = cloud.pods.get(rest[0]) if rest else None
                    if pod is None:
                        return self._reply(404, {"error": "no pod"})
                    if len(rest) == 1 and self.command == "GET":
                        return self._reply(200, pod)
                    if len(rest) == 1 and self.command == "DELETE":
                        pod["desiredStatus"] = "TERMINATED"
                        return self._reply(204)
                    if rest[1:] == ["start"]:
                        pod["desiredStatus"] = "RUNNING"
                        pod["gpu"] = {"count": cloud.start_gpu_count}
                        return self._reply(200)
                    if rest[1:] == ["stop"]:
                        if cloud.fail_stop:
                            return self._reply(500, {"error": f"echo {self.headers.get('Authorization')}"})
                        pod["desiredStatus"] = "EXITED"
                        return self._reply(200)
                if parts[:1] == ["proxy"]:
                    pod_id = parts[1].rsplit("-", 1)[0]
                    running = cloud.pods.get(pod_id, {}).get("desiredStatus") == "RUNNING"
                    if parts[2:] == ["v1", "chat", "completions"] and self.command == "POST":
                        cloud.chat_requests.append({"auth": self.headers.get("Authorization"), "body": body})
                        return self._reply(*cloud.chat_reply())
                    if parts[2:] == ["health"]:
                        return self._reply(200 if running and cloud.healthy else 502, b"")
                    if parts[2:] == ["metrics"]:
                        if not (running and cloud.metrics_ok):
                            return self._reply(502, b"bad gateway", "text/plain")
                        text = (
                            "# HELP vllm:request_success_total Count of successfully processed requests.\n"
                            "# TYPE vllm:request_success_total counter\n"
                            f'vllm:request_success_total{{finished_reason="stop",model_name="lilith"}} {cloud.success_total}\n'
                            f'vllm:request_success_total{{finished_reason="length",model_name="lilith"}} 0.0\n'
                            f'vllm:num_requests_running{{model_name="lilith"}} {cloud.running}\n'
                            'vllm:num_requests_waiting{model_name="lilith"} 0.0\n')
                        return self._reply(200, text.encode(), "text/plain")
                return self._reply(404, {"error": "unknown path"})

            do_GET = do_POST = do_DELETE = _handle

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def chat_reply(self) -> tuple[int, dict]:
        def completion(message: dict, finish: str) -> dict:
            return {"id": "chatcmpl-x", "object": "chat.completion", "model": "lilith",
                    "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}]}
        call = {"id": "chatcmpl-tool-1", "type": "function",
                "function": {"name": "get_time", "arguments": json.dumps({"timezone": "UTC"})}}
        if self.chat_mode == "tool":
            return 200, completion({"content": None, "tool_calls": [call]}, "tool_calls")
        if self.chat_mode == "text":
            return 200, completion({"content": "It is noon."}, "stop")
        if self.chat_mode == "malformed":
            bad = {**call, "function": {"name": "get_time", "arguments": "timezone=UTC"}}
            return 200, completion({"content": None, "tool_calls": [bad]}, "tool_calls")
        if self.chat_mode == "unparsed":
            return 200, completion({"content": "<|tool_call>call:get_time{timezone:UTC}<tool_call|>"}, "stop")
        return 400, {"error": {"message": '"auto" tool choice requires --enable-auto-tool-choice and '
                                          "--tool-call-parser to be set"}}

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def proxy_template(self) -> str:
        return f"{self.base}/proxy/{{pod_id}}-{{port}}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def calls(self, method: str, suffix: str) -> list[dict]:
        return [r for r in self.requests if r["method"] == method and r["path"].endswith(suffix)]


class FakeClock:
    def __init__(self, start: dt.datetime):
        self.moment = start

    def now(self) -> dt.datetime:
        return self.moment

    def sleep(self, seconds: float) -> None:
        self.moment += dt.timedelta(seconds=seconds)

    def advance(self, **delta) -> None:
        self.moment += dt.timedelta(**delta)


WEDNESDAY_NOON = dt.datetime(2026, 9, 30, 12, 0, tzinfo=dt.timezone.utc)


class Harness(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.cloud = MockCloud()
        self.addCleanup(self.cloud.close)
        self.op = self.tmp / "op"
        self.op.write_text(FAKE_OP.replace("#!/usr/bin/env python3", f"#!{sys.executable}", 1))
        self.op.chmod(0o755)
        self.db = self.tmp / "op-db.json"
        self.db.write_text(json.dumps({"Runpod API Key": {"credential": RUNPOD_KEY}}))
        self.state_dir = self.tmp / "state"
        os.environ.update(FAKE_OP_DB=str(self.db), FAKE_OP_ARGV=str(self.tmp / "op-argv.log"),
                          FAKE_OP_STDIN=str(self.tmp / "op-stdin.log"))
        self.clock = FakeClock(WEDNESDAY_NOON)

    def context(self, **policy) -> ls.Context:
        store = ls.OnePassword(str(self.op))
        runpod = ls.Runpod(ls.KeyProvider(store, ls.RUNPOD_KEY_REF, cache=True),
                           f"{self.cloud.base}/rest/v1", f"{self.cloud.base}/graphql")
        return ls.Context(state_dir=self.state_dir, store=store, runpod=runpod, policy=ls.Policy(**policy),
                          clock=self.clock, log=ls.Log(self.state_dir, echo=False), proxy_template=self.cloud.proxy_template)

    def op_db(self) -> dict:
        return json.loads(self.db.read_text())

    def ledger_events(self) -> list[dict]:
        path = self.state_dir / "ledger.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def created(self, **policy) -> ls.Context:
        ctx = self.context(**policy)
        ls.cmd_create(ctx, yes=True, key_source="env", kv_cache_dtype="auto")
        ls.cmd_stop(ctx, "test-setup")
        return ctx


# ------------------------------------------------------------------ budget
class BudgetTests(unittest.TestCase):
    policy = ls.Policy()

    def usage(self, week=0.0, month=0.0, spend=0.0):
        return ls.Usage(week_hours=week, month_hours=month, month_spend_usd=spend, total_hours=month,
                        total_spend_usd=spend, open_session=None)

    def test_fresh_budget_allows_a_full_two_hour_session(self):
        decision = ls.evaluate_budget(self.policy, self.usage(), 66.73, 3.49, WEDNESDAY_NOON)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.cap_seconds, 7200)
        self.assertEqual(decision.binding, "session_cap")

    def test_weekly_gpu_hours_exhausted_is_refused_with_the_reset_time(self):
        decision = ls.evaluate_budget(self.policy, self.usage(week=5.9, month=5.9, spend=5.9 * 3.49), 200, 3.49,
                                      WEDNESDAY_NOON)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.binding, "weekly_gpu_hours")
        self.assertIn("2026-10-05T00:00:00Z", decision.refusal)

    def test_weekly_remainder_shortens_the_session(self):
        decision = ls.evaluate_budget(self.policy, self.usage(week=5.0, month=5.0, spend=17.45), 200, 3.49,
                                      WEDNESDAY_NOON)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.cap_seconds, 3600)

    def test_monthly_credit_share_is_a_quarter_of_the_credit_available_this_month(self):
        # balance 40 + 12.8 spent = 52.8 available; 25% = 13.2; 0.4 left = 6.9 min < 10 min
        decision = ls.evaluate_budget(self.policy, self.usage(week=1, month=3.67, spend=12.8), 40, 3.49,
                                      WEDNESDAY_NOON)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.binding, "monthly_credit_share")
        self.assertAlmostEqual(decision.month_allowance_usd, 13.2)
        self.assertIn("2026-10-01T00:00:00Z", decision.refusal)

    def test_credit_reserve_protects_training(self):
        decision = ls.evaluate_budget(self.policy, self.usage(), 5.3, 3.49, WEDNESDAY_NOON)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.binding, "credit_reserve")
        self.assertIn("never adds funds", decision.refusal)

    def test_ledger_splits_weeks_and_months_and_counts_open_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = ls.Ledger(Path(tmp) / "ledger.jsonl")
            at = lambda s: dt.datetime.fromisoformat(s)
            ledger.start("a", "p", at("2026-09-27T23:00:00+00:00"), 3.49, 7200, "start")  # Sunday, last week
            ledger.stop("a", "p", at("2026-09-28T01:00:00+00:00"), "idle")  # 1 h this week (Mon)
            ledger.start("b", "p", at("2026-09-30T11:00:00+00:00"), 3.49, 7200, "start")  # open
            usage = ledger.usage(WEDNESDAY_NOON)
            self.assertAlmostEqual(usage.week_hours, 2.0)
            self.assertAlmostEqual(usage.month_hours, 3.0)
            self.assertAlmostEqual(usage.month_spend_usd, 3 * 3.49)
            self.assertEqual(usage.open_session, "b")


class BudgetRefusalCliTests(Harness):
    def test_start_is_refused_over_the_weekly_budget_and_the_pod_stays_stopped(self):
        self.cloud.balance = 500.0  # plenty of credit: the weekly GPU-hour cap must bind on its own
        ctx = self.created()
        ledger = ctx.ledger
        ledger.start("old", "x", WEDNESDAY_NOON - dt.timedelta(hours=10), 3.49, 7200, "start")
        ledger.stop("old", "x", WEDNESDAY_NOON - dt.timedelta(hours=4, minutes=5), "idle")
        starts_before = len(self.cloud.calls("POST", "/start"))
        with self.assertRaises(ls.BudgetRefused) as caught:
            ls.cmd_start(ctx, wait_seconds=60)
        self.assertIn("weekly_gpu_hours", str(caught.exception))
        self.assertEqual(caught.exception.exit_code, 3)
        self.assertEqual(len(self.cloud.calls("POST", "/start")), starts_before)
        pod_id = ctx.state().pod_id
        self.assertEqual(self.cloud.pods[pod_id]["desiredStatus"], "EXITED")

    def test_create_is_refused_when_the_credit_is_nearly_gone(self):
        self.cloud.balance = 4.0
        with self.assertRaises(ls.BudgetRefused):
            ls.cmd_create(self.context(), yes=True, key_source="env", kv_cache_dtype="auto")
        self.assertEqual(self.cloud.calls("POST", "/rest/v1/pods"), [])


# ------------------------------------------------------------------ stable URL bookkeeping
class StableUrlTests(Harness):
    def test_proxy_url_is_the_documented_runpod_form(self):
        self.assertEqual(ls.proxy_url("abc123xyz"), "https://abc123xyz-8000.proxy.runpod.net")
        with self.assertRaises(ls.UsageError):
            ls.proxy_url("bad id/../x")

    def test_create_requires_explicit_confirmation(self):
        with self.assertRaises(ls.ConfirmationRequired):
            ls.cmd_create(self.context(), yes=False, key_source="env", kv_cache_dtype="auto")
        self.assertEqual(self.cloud.requests, [])

    def test_create_records_the_endpoint_and_stop_start_keeps_the_url(self):
        ctx = self.context()
        result = ls.cmd_create(ctx, yes=True, key_source="env", kv_cache_dtype="auto")
        pod_id, base_url = result["pod_id"], result["base_url"]
        self.assertEqual(base_url, ctx.proxy_template.format(pod_id=pod_id, port=8000))
        endpoint = self.op_db()[ls.ENDPOINT_ITEM]
        self.assertEqual((endpoint["pod_id"], endpoint["base_url"]), (pod_id, base_url))
        body = self.cloud.pods[pod_id]["body"]
        self.assertEqual(body["name"], "akh-medu-lilith-serve")
        self.assertEqual(body["dataCenterIds"], ["EU-FR-1"])
        self.assertEqual(body["networkVolumeId"], "c6kdciozex")
        self.assertEqual(body["allowedCudaVersions"], ["13.0"])
        self.assertEqual(body["gpuTypeIds"], ["NVIDIA H100 80GB HBM3"])
        self.assertIn("8000/http", body["ports"])
        self.assertEqual(body["env"]["LILITH_SERVE_API_KEY"], self.op_db()[ls.API_KEY_ITEM]["credential"])
        self.assertGreaterEqual(len(body["env"]["LILITH_SERVE_API_KEY"]), 32)
        with self.assertRaises(ls.AlreadyCreated):
            ls.cmd_create(ctx, yes=True, key_source="env", kv_cache_dtype="auto")

        edits_before = self.op_argv_count("edit")
        for _ in range(2):
            ls.cmd_stop(ctx, "manual")
            self.assertEqual(self.cloud.pods[pod_id]["desiredStatus"], "EXITED")
            started = ls.cmd_start(ctx, wait_seconds=60)
            self.assertEqual((started["pod_id"], started["base_url"]), (pod_id, base_url))
        self.assertEqual(ctx.state().base_url, base_url)
        self.assertEqual(self.op_argv_count("edit"), edits_before, "stop/start must not rewrite the endpoint")
        self.assertEqual(len(self.cloud.calls("POST", "/rest/v1/pods")), 1, "stop/start never creates a pod")

    def test_a_fresh_checkout_adopts_the_endpoint_from_1password(self):
        ctx = self.created()
        pod_id = ctx.state().pod_id
        (self.state_dir / "state.json").unlink()
        started = ls.cmd_start(ctx, wait_seconds=60)
        self.assertEqual(started["pod_id"], pod_id)

    def test_zero_gpu_resume_is_stopped_again_with_options(self):
        ctx = self.created()
        self.cloud.start_gpu_count = 0
        with self.assertRaises(ls.ZeroGpu) as caught:
            ls.cmd_start(ctx, wait_seconds=60)
        self.assertIn("recreate", caught.exception.help)
        self.assertEqual(self.cloud.pods[ctx.state().pod_id]["desiredStatus"], "EXITED")
        self.assertEqual(self.ledger_events()[-1]["reason"], "zero_gpu")
        self.assertIsNone(ctx.state().session)

    def test_install_waits_for_the_ssh_mapping_then_refuses_after_the_deadline(self):
        ctx = self.created()
        ls.cmd_start(ctx, wait_seconds=60)
        pod_id = ctx.state().pod_id
        self.assertEqual(ls.wait_for_ssh(ctx, pod_id).port_mappings["22"], 40022)
        self.cloud.pods[pod_id]["portMappings"] = {}
        before = self.clock.now()
        with self.assertRaises(ls.UsageError) as caught:
            ls.wait_for_ssh(ctx, pod_id)
        self.assertIn("without public ssh", str(caught.exception))
        self.assertGreaterEqual((self.clock.now() - before).total_seconds(), ls.SSH_MAPPING_WAIT_S)

    def test_install_refuses_a_stopped_pod_without_waiting(self):
        ctx = self.created()
        before = self.clock.now()
        with self.assertRaises(ls.UsageError):
            ls.wait_for_ssh(ctx, ctx.state().pod_id)
        self.assertEqual(self.clock.now(), before)

    def test_payload_extraction_does_not_chown_on_the_network_volume(self):
        source = Path(ls.__file__).read_text()
        self.assertIn('tar --no-same-owner -xzf - -C "$T.partial"', source)

    def test_health_timeout_stops_the_pod(self):
        ctx = self.created()
        self.cloud.healthy = False
        with self.assertRaises(ls.HealthTimeout):
            ls.cmd_start(ctx, wait_seconds=60)
        self.assertEqual(self.cloud.pods[ctx.state().pod_id]["desiredStatus"], "EXITED")
        self.assertEqual(self.ledger_events()[-1]["reason"], "health_timeout")

    def test_recreate_is_the_only_path_that_changes_the_url(self):
        ctx = self.created()
        old = ctx.state()
        with self.assertRaises(ls.ConfirmationRequired):
            ls.cmd_recreate(ctx, yes=True, confirm="someotherpod", key_source="env", kv_cache_dtype="auto")
        key_before = self.op_db()[ls.API_KEY_ITEM]["credential"]
        result = ls.cmd_recreate(ctx, yes=True, confirm=old.pod_id, key_source="env", kv_cache_dtype="auto")
        self.assertTrue(result["base_url_changed"])
        self.assertNotEqual(result["pod_id"], old.pod_id)
        self.assertEqual(self.cloud.pods[old.pod_id]["desiredStatus"], "TERMINATED")
        endpoint = self.op_db()[ls.ENDPOINT_ITEM]
        self.assertEqual((endpoint["pod_id"], endpoint["base_url"]), (result["pod_id"], result["base_url"]))
        self.assertEqual(self.op_db()[ls.API_KEY_ITEM]["credential"], key_before, "recreate reuses the key")

    def test_a_recreate_whose_create_failed_can_be_retried(self):
        ctx = self.created()
        old = ctx.state().pod_id
        del self.cloud.pods[old]          # terminated by the failed recreate, still in the local state
        result = ls.cmd_recreate(ctx, yes=True, confirm=old, key_source="env", kv_cache_dtype="auto")
        self.assertNotEqual(result["pod_id"], old)
        self.assertEqual(ctx.state().pod_id, result["pod_id"])

    def test_runpod_secret_key_source_keeps_the_value_out_of_the_create_body(self):
        ctx = self.context()
        result = ls.cmd_create(ctx, yes=True, key_source="runpod-secret", kv_cache_dtype="fp8")
        env = self.cloud.pods[result["pod_id"]]["body"]["env"]
        self.assertEqual(env["LILITH_SERVE_API_KEY"], "{{ RUNPOD_SECRET_lilith_serve_api_key }}")
        self.assertEqual(env["LILITH_SERVE_KV_CACHE_DTYPE"], "fp8")

    def op_argv_count(self, verb: str) -> int:
        path = self.tmp / "op-argv.log"
        return sum(1 for line in path.read_text().splitlines() if verb in json.loads(line)) if path.exists() else 0


# ------------------------------------------------------------------ idle watch + session cap
class IdleWatchTests(Harness):
    def running(self, **policy) -> tuple[ls.Context, ls.Vllm]:
        ctx = self.created(**policy)
        ls.cmd_start(ctx, wait_seconds=60)
        state = ctx.state()
        return ctx, ls.Vllm(state.base_url, lambda: None)

    def test_no_requests_for_twenty_minutes_stops_the_pod(self):
        ctx, probe = self.running()
        pod_id = ctx.state().pod_id
        self.assertEqual(ls.watch_tick(ctx, probe), ls.TickOutcome.ACTIVE)  # baseline
        self.clock.advance(minutes=10)
        self.cloud.success_total += 3  # a request
        self.assertEqual(ls.watch_tick(ctx, probe), ls.TickOutcome.ACTIVE)
        self.clock.advance(minutes=19)
        self.assertEqual(ls.watch_tick(ctx, probe), ls.TickOutcome.ACTIVE)
        self.clock.advance(minutes=1)
        self.assertEqual(ls.watch_tick(ctx, probe), ls.TickOutcome.IDLE_STOP)
        self.assertEqual(self.cloud.pods[pod_id]["desiredStatus"], "EXITED")
        self.assertEqual(self.ledger_events()[-1]["reason"], "idle")
        self.assertIsNone(ctx.state().session)

    def test_an_in_flight_request_is_activity(self):
        ctx, probe = self.running()
        ls.watch_tick(ctx, probe)
        self.cloud.running = 1
        for _ in range(3):
            self.clock.advance(minutes=15)
            self.assertEqual(ls.watch_tick(ctx, probe), ls.TickOutcome.ACTIVE)

    def test_unreachable_metrics_are_not_activity(self):
        ctx, probe = self.running()
        ls.watch_tick(ctx, probe)
        self.cloud.metrics_ok = False
        self.clock.advance(minutes=10)
        self.assertEqual(ls.watch_tick(ctx, probe), ls.TickOutcome.ACTIVE)
        self.clock.advance(minutes=10)
        self.assertEqual(ls.watch_tick(ctx, probe), ls.TickOutcome.IDLE_STOP)

    def test_session_cap_stops_a_busy_pod_at_two_hours(self):
        ctx, probe = self.running()
        pod_id = ctx.state().pod_id
        outcome = ls.TickOutcome.ACTIVE
        minutes = 0
        while outcome is ls.TickOutcome.ACTIVE and minutes < 200:
            self.cloud.success_total += 1
            outcome = ls.watch_tick(ctx, probe)
            self.clock.advance(minutes=5)
            minutes += 5
        self.assertEqual(outcome, ls.TickOutcome.CAP_STOP)
        self.assertGreaterEqual(minutes, 120)
        self.assertLessEqual(minutes, 125)
        self.assertEqual(self.cloud.pods[pod_id]["desiredStatus"], "EXITED")
        self.assertEqual(self.ledger_events()[-1]["reason"], "session_cap")
        usage = ctx.ledger.usage(self.clock.now())
        self.assertIsNone(usage.open_session)

    def test_session_cap_follows_a_short_weekly_remainder(self):
        self.cloud.balance = 500.0
        ctx = self.created()
        ctx.ledger.start("old", "x", WEDNESDAY_NOON - dt.timedelta(hours=10), 3.49, 7200, "start")
        ctx.ledger.stop("old", "x", WEDNESDAY_NOON - dt.timedelta(hours=4, minutes=30), "idle")  # 5.5 h used
        result = ls.cmd_start(ctx, wait_seconds=60)
        self.assertEqual(result["session_cap_seconds"], 1800)
        probe = ls.Vllm(ctx.state().base_url, lambda: None)
        self.cloud.running = 1
        self.clock.advance(minutes=31)
        self.assertEqual(ls.watch_tick(ctx, probe), ls.TickOutcome.CAP_STOP)

    def test_a_failed_stop_keeps_the_session_open_and_retries(self):
        ctx, probe = self.running()
        ls.watch_tick(ctx, probe)
        self.cloud.fail_stop = True
        self.clock.advance(minutes=25)
        self.assertEqual(ls.watch_tick(ctx, probe), ls.TickOutcome.STOP_FAILED)
        self.assertIsNotNone(ctx.state().session)
        self.cloud.fail_stop = False
        self.assertEqual(ls.watch_tick(ctx, probe), ls.TickOutcome.IDLE_STOP)
        log = (self.state_dir / "logs" / "supervisor.log").read_text()
        self.assertNotIn(RUNPOD_KEY, log, "the echoed Authorization header must be redacted")
        self.assertIn("[redacted]", log)

    def test_an_external_stop_closes_the_session(self):
        ctx, probe = self.running()
        ls.watch_tick(ctx, probe)
        self.cloud.pods[ctx.state().pod_id]["desiredStatus"] = "EXITED"
        self.clock.advance(minutes=6)
        self.assertEqual(ls.watch_tick(ctx, probe), ls.TickOutcome.EXTERNAL_STOP)
        self.assertEqual(self.ledger_events()[-1]["reason"], "external")

    def test_the_daemon_adopts_a_pod_started_outside_the_supervisor(self):
        ctx = self.created()
        pod_id = ctx.state().pod_id
        self.cloud.pods[pod_id]["desiredStatus"] = "RUNNING"
        probe = ls.Vllm(ctx.state().base_url, lambda: None)
        self.assertEqual(ls.watch_tick(ctx, probe), ls.TickOutcome.ADOPTED)
        self.assertEqual(self.ledger_events()[-1]["source"], "adopted")
        self.assertEqual(ls.watch_tick(ctx, probe), ls.TickOutcome.ACTIVE)
        self.clock.advance(minutes=21)
        self.assertEqual(ls.watch_tick(ctx, probe), ls.TickOutcome.IDLE_STOP)

    def test_failed_ticks_back_off_exponentially_to_the_cap(self):
        self.assertEqual([ls.retry_interval(60, n) for n in range(6)], [60, 60, 120, 240, 300, 300])
        self.assertEqual(ls.retry_interval(600, 3), 600)   # a long base interval is never shortened

    def test_parse_metrics_sums_labelled_series(self):
        text = ('vllm:request_success_total{finished_reason="stop"} 4.0\n'
                'vllm:request_success_total{finished_reason="length"} 1.0\n'
                'vllm:num_requests_running{model_name="lilith"} 2.0\n# comment\n')
        metrics = ls.parse_metrics(text)
        self.assertEqual(metrics.activity[0], 5.0)
        self.assertEqual(metrics.in_flight, 2.0)
        with self.assertRaises(ls.ProbeError):
            ls.parse_metrics("process_cpu_seconds_total 3\n")


# ------------------------------------------------------------------ tool-calling readiness
class ToolReadinessTests(Harness):
    def test_start_verifies_a_well_formed_tool_call(self):
        ctx = self.created()
        result = ls.cmd_start(ctx, wait_seconds=60)
        self.assertEqual(result["tool_calling"]["status"], "verified")
        self.assertEqual(result["tool_calling"]["finish_reason"], "tool_calls")
        (request,) = self.cloud.chat_requests
        self.assertEqual(request["auth"], f"Bearer {self.op_db()[ls.API_KEY_ITEM]['credential']}")
        self.assertEqual(request["body"]["model"], "lilith")
        self.assertEqual(request["body"]["tool_choice"], "auto")
        self.assertEqual(request["body"]["tools"][0]["function"]["name"], "get_time")
        state = ctx.state()
        self.assertEqual(state.session.tool_calling["status"], "verified")
        evidence = json.loads((self.state_dir / "readiness" / f"{state.session.id}.json").read_text())
        self.assertEqual((evidence["status"], evidence["pod_id"]), ("verified", state.pod_id))

    def test_plain_text_answer_is_recorded_unverified_with_the_finish_reason(self):
        ctx = self.created()
        self.cloud.chat_mode = "text"
        result = ls.cmd_start(ctx, wait_seconds=60)
        self.assertEqual(result["tool_calling"]["status"], "unverified")
        self.assertEqual(result["tool_calling"]["finish_reason"], "stop")
        self.assertEqual(self.cloud.pods[result["pod_id"]]["desiredStatus"], "RUNNING")

    def test_broken_tool_calling_fails_start_and_stops_the_pod(self):
        for mode in ("malformed", "unparsed", "refused"):
            with self.subTest(mode=mode):
                ctx = self.created() if mode == "malformed" else ctx
                self.cloud.chat_mode = mode
                with self.assertRaises(ls.ToolCallingBroken) as caught:
                    ls.cmd_start(ctx, wait_seconds=60)
                self.assertEqual(caught.exception.exit_code, 6)
                self.assertIn("pod stopped", str(caught.exception))
                self.assertEqual(self.cloud.pods[ctx.state().pod_id]["desiredStatus"], "EXITED")
                self.assertEqual(self.ledger_events()[-1]["reason"], "tool_calling_broken")

    def test_wait_ready_reports_but_does_not_stop(self):
        ctx = self.created()
        self.cloud.pods[ctx.state().pod_id]["desiredStatus"] = "RUNNING"
        self.cloud.chat_mode = "malformed"
        with self.assertRaises(ls.ToolCallingBroken):
            ls.wait_ready(ctx, ctx.state(), 60, stop_on_failure=False)
        self.assertEqual(self.cloud.pods[ctx.state().pod_id]["desiredStatus"], "RUNNING")

    def test_check_tool_calls_shapes(self):
        now = WEDNESDAY_NOON
        good = {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}]}}]}
        self.assertEqual(ls.check_tool_calls(200, good, now).status, "verified")
        for bad in ({"choices": []},
                    {"choices": [{"message": {"tool_calls": [{"id": "c1", "type": "function",
                                                               "function": {"name": "other", "arguments": "{}"}}]}}]},
                    {"choices": [{"message": {"tool_calls": [{"type": "function",
                                                               "function": {"name": "get_time", "arguments": "{}"}}]}}]},
                    {"choices": [{"message": {"tool_calls": [{"id": "c", "type": "function",
                                                               "function": {"name": "get_time", "arguments": "[]"}}]}}]}):
            with self.assertRaises(ls.ToolCallingBroken):
                ls.check_tool_calls(200, bad, now)
        with self.assertRaises(ls.ToolCallingBroken):
            ls.check_tool_calls(500, {"error": "x"}, now)
        plain = ls.check_tool_calls(200, {"choices": [{"finish_reason": "length", "message": {"content": "hi"}}]}, now)
        self.assertEqual((plain.status, plain.finish_reason), ("unverified", "length"))


# ------------------------------------------------------------------ secrets
class SecretLeakTests(Harness):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        env = {**os.environ, "LILITH_SERVE_OP": str(self.op), "LILITH_SERVE_STATE_DIR": str(self.state_dir),
               "LILITH_SERVE_RUNPOD_REST": f"{self.cloud.base}/rest/v1",
               "LILITH_SERVE_RUNPOD_GRAPHQL": f"{self.cloud.base}/graphql",
               "LILITH_SERVE_PROXY_TEMPLATE": self.cloud.proxy_template}
        return subprocess.run([sys.executable, str(HERE / "lilith_serve.py"), *args], env=env,
                              capture_output=True, text=True, timeout=120)

    def test_no_secret_reaches_stdout_logs_state_or_argv(self):
        outputs = []
        for args in (("create", "--yes"), ("status", "--json"), ("idle-watch", "--once"), ("stop",),
                     ("start", "--wait-seconds", "30"), ("ledger", "--json"), ("stop",)):
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 0, f"{args}: {result.stderr}")
            outputs += [result.stdout, result.stderr]
        self.cloud.fail_stop = True  # the error path echoes the Authorization header back
        result = self.run_cli("start", "--wait-seconds", "30")
        result = self.run_cli("stop")
        self.assertNotEqual(result.returncode, 0)
        outputs += [result.stdout, result.stderr]

        serve_key = self.op_db()[ls.API_KEY_ITEM]["credential"]
        files = [p for p in self.state_dir.rglob("*") if p.is_file()] + [self.tmp / "op-argv.log"]
        haystacks = outputs + [p.read_text() for p in files]
        for secret in (RUNPOD_KEY, serve_key):
            for text in haystacks:
                self.assertNotIn(secret, text)
        # The serve key went to op on stdin only, and the Runpod key only in the Authorization header.
        self.assertIn(serve_key, (self.tmp / "op-stdin.log").read_text())
        rest = [r for r in self.cloud.requests if r["path"].startswith("/rest/")]
        self.assertTrue(all(r["headers"].get("Authorization") == f"Bearer {RUNPOD_KEY}" for r in rest))
        self.assertTrue(all(r["headers"].get("User-Agent") == "curl/8.7.1" for r in self.cloud.requests))
        self.assertTrue(all(RUNPOD_KEY not in r["path"] for r in self.cloud.requests), "no key in URLs")
        for path in files:
            if path.parent.name in ("state", "logs"):
                self.assertEqual(path.stat().st_mode & 0o077, 0, f"{path} must be private")

    def test_existing_serve_key_is_reused_not_regenerated(self):
        db = self.op_db()
        db[ls.API_KEY_ITEM] = {"credential": SERVE_KEY_PREEXISTING}
        self.db.write_text(json.dumps(db))
        result = self.run_cli("create", "--yes")
        self.assertEqual(result.returncode, 0, result.stderr)
        pod = next(iter(self.cloud.pods.values()))
        self.assertEqual(pod["body"]["env"]["LILITH_SERVE_API_KEY"], SERVE_KEY_PREEXISTING)
        self.assertNotIn(SERVE_KEY_PREEXISTING, result.stdout + result.stderr)


# ------------------------------------------------------------------ on-pod pieces
class LogRedactTests(unittest.TestCase):
    def test_bodies_and_keys_are_redacted_and_access_lines_kept(self):
        key = "k" * 40
        lines = [
            "INFO 09-30 12:00:01 [logger.py:43] Received request chatcmpl-abc123def: prompt: '<start>secret plan', "
            "params: SamplingParams(n=1)",
            'INFO:     10.0.0.1:0 - "POST /v1/chat/completions HTTP/1.1" 200 OK',
            f"DEBUG header Authorization: Bearer {key}",
            "INFO Avg prompt throughput: 120.0 tokens/s, Avg generation throughput: 79.3 tokens/s",
            "x" * 5000,
        ]
        with tempfile.TemporaryDirectory() as tmp:
            log_redact.run(io.StringIO("\n".join(lines) + "\n"), Path(tmp), "vllm", None, (key,))
            (path,) = Path(tmp).glob("vllm-*.log")
            text = path.read_text()
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertNotIn("secret plan", text)
        self.assertIn("[redacted: request body] id=chatcmpl-abc123def", text)
        self.assertIn('"POST /v1/chat/completions HTTP/1.1" 200', text)
        self.assertNotIn(key, text)
        self.assertIn("79.3 tokens/s", text)
        self.assertIn("[truncated 3000 chars]", text)


def make_active_registry(root: Path, rank: int = 16, tamper: bool = False) -> Path:
    from exam.certify import AdapterRegistry, ModelManifest, OperatorAcceptance, C5_TRAJECTORIES_V3, gate_report
    registry = root / "adapters" / "active"
    blob = b"lilith adapter tensors"
    adapter_sha = hashlib.sha256(blob).hexdigest()
    weights = registry / "weights" / adapter_sha
    weights.mkdir(parents=True)
    (weights / "adapter_model.safetensors").write_bytes(b"tampered" if tamper else blob)
    (weights / "adapter_config.json").write_text(json.dumps({"r": rank, "peft_type": "LORA"}))
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
    candidate = ModelManifest(run_id="lilith-train-r1", checkpoint_step=100, adapter_sha256=adapter_sha,
                              tool_contract_sha256=sha("tools"), data_manifests=(C5_TRAJECTORIES_V3,),
                              training_config_sha256=sha("config"), code_main_sha="a" * 40,
                              exam_manifest_sha256=sha("exam"), created_at="2026-09-30T00:00:00Z")
    reg = AdapterRegistry(registry)
    reg.write_candidate(candidate)
    matrix = {"arm": "candidate", "families": {"briefing": {"verdict": "pass", "blockers": [], "missing": []}}}
    report = gate_report(sha("exam"), [matrix])
    reg.accept(candidate.manifest_sha256, report,
               OperatorAcceptance("toasty", "2026-09-30T01:00:00Z", candidate.manifest_sha256,
                                  report["gate_report_sha256"], "Stage 0 test"))
    return registry


class ServeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_no_active_manifest_serves_the_frozen_parent(self):
        selection = serve_adapter.select(self.tmp / "adapters" / "active", serve_adapter.Mode.AUTO)
        self.assertEqual(selection.mode, "parent")

    def test_certified_active_adapter_is_served_with_its_rank(self):
        registry = make_active_registry(self.tmp, rank=16)
        selection = serve_adapter.select(registry, serve_adapter.Mode.AUTO)
        self.assertEqual(selection.mode, "adapter")
        self.assertEqual((selection.lora_rank, selection.max_lora_rank), (16, 16))
        self.assertEqual(selection.accepted_by, "toasty")
        self.assertEqual(serve_adapter.vllm_max_lora_rank(24), 32)

    def test_tampered_adapter_is_refused_not_silently_replaced_by_the_parent(self):
        registry = make_active_registry(self.tmp, tamper=True)
        with self.assertRaises(serve_adapter.SelectionError) as caught:
            serve_adapter.select(registry, serve_adapter.Mode.AUTO)
        self.assertEqual(caught.exception.code, "lilith::serve::adapter_uncertified")
        self.assertIn("LILITH_SERVE_ADAPTER=parent", caught.exception.help)

    def test_parent_mode_is_a_recorded_override(self):
        registry = make_active_registry(self.tmp)
        selection = serve_adapter.select(registry, serve_adapter.Mode.PARENT)
        self.assertEqual(selection.mode, "parent")
        self.assertIn("NOT served", selection.reason)


FAKE_BOOTSTRAP = """#!/usr/bin/env bash
[[ "${1:-}" == --check ]] && echo '{"status": "ok"}'
exit 0
"""

FAKE_VLLM = r'''#!/usr/bin/env python3
import http.server, json, os, sys
args = sys.argv[1:]
if "--help" in args or "--help=all" in args:
    print("--enable-log-requests  Enable logging requests.")
    parsers = os.environ.get("FAKE_VLLM_PARSERS", "gemma4,hermes,pythonic")
    if parsers:
        print("--tool-call-parser {%s}" % parsers)
        print("                        Select the tool call parser depending on the model.")
    sys.exit(0)
with open(os.environ["FAKE_VLLM_ARGV"], "w") as out:
    json.dump(args, out)
port = int(args[args.index("--port") + 1])
key = os.environ.get("VLLM_API_KEY", "")
name = args[args.index("--served-model-name") + 1]
lora = [a.split("=")[0] for a in args if a.startswith("lilith=")]
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        sys.stderr.write('INFO:     127.0.0.1:0 - "%s %s HTTP/1.1" 200 OK\n' % (self.command, self.path))
        sys.stderr.flush()
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200); self.end_headers(); return
        if self.path == "/v1/models":
            if os.environ.get("FAKE_NO_AUTH") != "1" and self.headers.get("Authorization") != "Bearer " + key:
                self.send_response(401); self.end_headers(); return
            body = json.dumps({"data": [{"id": n} for n in [name] + lora]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body); return
        self.send_response(404); self.end_headers()
print("INFO Received request chatcmpl-abcdef123: prompt: 'TOP SECRET PROMPT', params: SamplingParams()", flush=True)
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ServePodTests(unittest.TestCase):
    KEY = "serve-pod-test-key-0123456789abcdefghijklmnop"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = self.tmp / "lilith"
        (self.root / "parent").mkdir(parents=True)
        (self.root / "parent" / "chat_template.jinja").write_text("{{ bos_token }}{% for m in messages %}{% endfor %}")
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        self.bootstrap = bin_dir / "bootstrap.sh"
        self.bootstrap.write_text(FAKE_BOOTSTRAP)
        self.vllm = bin_dir / "vllm"
        self.vllm.write_text(FAKE_VLLM.replace("#!/usr/bin/env python3", f"#!{sys.executable}", 1))
        for path in (self.bootstrap, self.vllm):
            path.chmod(0o755)
        self.port = free_port()

    def env(self, **extra) -> dict:
        return {**os.environ, "LILITH_ROOT": str(self.root), "LILITH_ALLOW_NON_WORKSPACE": "1",
                "LILITH_ALLOW_UNMERGED": "1", "LILITH_SERVE_BOOTSTRAP": str(self.bootstrap),
                "LILITH_SERVE_VLLM": str(self.vllm), "LILITH_SERVE_PYTHON": sys.executable,
                "LILITH_SERVE_PORT": str(self.port), "LILITH_SERVE_API_KEY": self.KEY,
                "LILITH_SERVE_POD_HARD_CAP_SECONDS": "0", "LILITH_SERVE_HEALTH_TIMEOUT_SECONDS": "60",
                "LILITH_SERVE_SKIP_PID1_ENV": "1", "FAKE_VLLM_ARGV": str(self.tmp / "vllm-argv.json"), **extra}

    def launch(self, **extra) -> subprocess.Popen:
        process = subprocess.Popen(["bash", str(HERE / "serve_pod.sh")], env=self.env(**extra),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   start_new_session=True)
        self.addCleanup(self.kill, process)
        return process

    @staticmethod
    def kill(process: subprocess.Popen) -> None:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        if process.stdout is not None and not process.stdout.closed:
            process.stdout.close()

    def wait_state(self, process: subprocess.Popen, *states: str, timeout: float = 60) -> dict:
        deadline = time.time() + timeout
        status_path = self.root / "serve" / "status.json"
        while time.time() < deadline:
            if status_path.exists():
                status = json.loads(status_path.read_text())
                if status["state"] in states:
                    return status
            if process.poll() is not None and not status_path.exists():
                break
            time.sleep(0.3)
        output = process.stdout.read() if process.poll() is not None and process.stdout else ""
        self.fail(f"no state {states}; output: {output}")

    def test_serves_the_frozen_parent_with_an_enforced_key_and_private_logs(self):
        process = self.launch()
        status = self.wait_state(process, "ready", "failed")
        self.assertEqual(status["state"], "ready", status)
        argv = json.loads((self.tmp / "vllm-argv.json").read_text())
        joined = " ".join(argv)
        self.assertNotIn(self.KEY, joined, "the key never goes on argv")
        self.assertIn("--served-model-name lilith", joined)
        self.assertIn("--max-model-len 65536", joined)
        self.assertIn("--kv-cache-dtype auto", joined)
        self.assertNotIn("--enable-log-requests", argv)
        self.assertNotIn("--enable-lora", argv)
        session = json.loads(next((self.root / "serve" / "sessions").glob("*/session.json")).read_text())
        self.assertEqual(session["mode"], "parent")
        self.assertEqual(session["provenance"]["status"], "unverified_allowed")
        self.kill(process)
        logs = "".join(p.read_text() for p in (self.root / "serve" / "logs").glob("vllm-*.log"))
        self.assertIn("[redacted: request body] id=chatcmpl-abcdef123", logs)
        self.assertNotIn("TOP SECRET PROMPT", logs)
        self.assertIn('"GET /v1/models HTTP/1.1"', logs)
        for path in self.root.rglob("*"):
            if path.is_file():
                self.assertNotIn(self.KEY, path.read_text(errors="replace"), path)

    def test_serves_a_certified_adapter_as_lilith(self):
        make_active_registry(self.root, rank=32)
        process = self.launch()
        self.assertEqual(self.wait_state(process, "ready", "failed")["state"], "ready")
        argv = json.loads((self.tmp / "vllm-argv.json").read_text())
        self.assertIn("--enable-lora", argv)
        self.assertEqual(argv[argv.index("--max-lora-rank") + 1], "32")
        self.assertEqual(argv[argv.index("--served-model-name") + 1], "lilith-parent")
        self.assertTrue(argv[argv.index("--lora-modules") + 1].startswith("lilith="))

    def test_an_unenforced_key_is_a_refusal(self):
        process = self.launch(FAKE_NO_AUTH="1")
        status = self.wait_state(process, "failed", "ready")
        self.assertEqual(status["state"], "failed")
        self.assertIn("lilith::serve::auth", status["detail"])

    def test_refusals_before_vllm_starts(self):
        for extra, code in (({"LILITH_SERVE_API_KEY": "short"}, "api_key"),
                            ({"LILITH_SERVE_KV_CACHE_DTYPE": "int8"}, "kv_dtype"),
                            ({"LILITH_ALLOW_NON_WORKSPACE": "0"}, "root")):
            result = subprocess.run(["bash", str(HERE / "serve_pod.sh")], env=self.env(**extra),
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 1, extra)
            self.assertIn(f"lilith::serve::{code}", result.stderr)
            self.assertFalse((self.tmp / "vllm-argv.json").exists())

    def test_tool_calling_is_on_with_the_gemma4_parser_and_the_pinned_template(self):
        process = self.launch()
        self.assertEqual(self.wait_state(process, "ready", "failed")["state"], "ready")
        argv = json.loads((self.tmp / "vllm-argv.json").read_text())
        self.assertIn("--enable-auto-tool-choice", argv)
        self.assertEqual(argv[argv.index("--tool-call-parser") + 1], "gemma4")
        template = self.root / "parent" / "chat_template.jinja"
        self.assertEqual(argv[argv.index("--chat-template") + 1], str(template))
        session = json.loads(next((self.root / "serve" / "sessions").glob("*/session.json")).read_text())
        tools = session["tool_calling"]
        self.assertEqual((tools["parser"], tools["requested"]), ("gemma4", "auto"))
        self.assertEqual(tools["chat_template_sha256"], hashlib.sha256(template.read_bytes()).hexdigest())
        self.assertIn("--help", tools["registry_source"])
        self.assertEqual(tools["available"], ["gemma4", "hermes", "pythonic"])

    def test_explicit_parser_must_be_registered(self):
        process = self.launch(LILITH_SERVE_TOOL_PARSER="hermes")
        self.assertEqual(self.wait_state(process, "ready", "failed")["state"], "ready")
        argv = json.loads((self.tmp / "vllm-argv.json").read_text())
        self.assertEqual(argv[argv.index("--tool-call-parser") + 1], "hermes")

    def test_no_fitting_parser_is_a_refusal_listing_the_available_ones(self):
        cases = (({"FAKE_VLLM_PARSERS": "hermes,pythonic"}, "lilith::serve::tool_parser", "hermes, pythonic"),
                 ({"LILITH_SERVE_TOOL_PARSER": "nope"}, "lilith::serve::tool_parser", "gemma4, hermes, pythonic"),
                 ({"FAKE_VLLM_PARSERS": ""}, "lilith::serve::tool_parser_registry", "cannot list"),
                 ({"LILITH_SERVE_CHAT_TEMPLATE": str(self.tmp / "missing.jinja")}, "lilith::serve::chat_template",
                  "does not exist"))
        for extra, code, needle in cases:
            with self.subTest(extra=extra):
                result = subprocess.run(["bash", str(HERE / "serve_pod.sh")], env=self.env(**extra),
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"error[{code}]", result.stderr)
                self.assertIn(needle, result.stderr)
                self.assertIn("help:", result.stderr)
                self.assertFalse((self.tmp / "vllm-argv.json").exists(), "vLLM must not start without tools")
                status = json.loads((self.root / "serve" / "status.json").read_text())
                self.assertEqual(status["state"], "failed")

    def test_the_registry_is_read_from_tool_parser_manager_when_vllm_imports(self):
        fake = self.tmp / "site"
        (fake / "vllm" / "tool_parsers").mkdir(parents=True)
        (fake / "vllm" / "__init__.py").write_text("")
        (fake / "vllm" / "tool_parsers" / "__init__.py").write_text(
            "class ToolParserManager:\n    tool_parsers = {'hermes': object}\n"
            "    lazy_parsers = {'gemma4': ('gemma4_engine_tool_parser', 'Gemma4EngineToolParser')}\n")
        help_file = self.tmp / "empty-help.txt"
        help_file.write_text("")
        result = subprocess.run([sys.executable, str(HERE / "serve_tools.py"), "resolve", "--help-file",
                                 str(help_file), "--chat-template", str(self.root / "parent" / "chat_template.jinja")],
                                env={**os.environ, "PYTHONPATH": str(fake)}, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        selection = json.loads(result.stdout)
        self.assertEqual(selection["parser"], "gemma4")
        self.assertEqual(selection["registry_source"], "vllm.tool_parsers.ToolParserManager")
        self.assertEqual(selection["available"], ["gemma4", "hermes"])

    # ── reasoning parser (Lilith C3: gemma4's thinking channel leaks into
    # `content` without one; unlike tool calling this is non-fatal — see the
    # serve_tools.py module docstring). ────────────────────────────────────

    def _fake_reasoning_registry(self, names: dict[str, str]) -> Path:
        """A fake `vllm.reasoning.ReasoningParserManager` on its own site dir,
        `{name: module_path}` becoming `lazy_parsers`."""
        fake = self.tmp / "reasoning-site"
        (fake / "vllm" / "reasoning").mkdir(parents=True)
        (fake / "vllm" / "__init__.py").write_text("")
        lazy = ", ".join(f"'{n}': ('{m}', 'X')" for n, m in names.items())
        (fake / "vllm" / "reasoning" / "__init__.py").write_text(
            f"class ReasoningParserManager:\n    reasoning_parsers = {{}}\n    lazy_parsers = {{{lazy}}}\n")
        return fake

    def test_reasoning_capture_is_disabled_by_default_when_vllm_is_not_importable(self):
        # The sandboxed test env has no real vLLM installed (FAKE_VLLM is a
        # stub binary, not an importable Python package), so `vllm.reasoning`
        # genuinely doesn't import — exactly like an off-pod dev checkout.
        # `auto` must degrade to disabled rather than refuse: reasoning
        # capture is an enhancement, not a requirement to serve.
        process = self.launch()
        self.assertEqual(self.wait_state(process, "ready", "failed")["state"], "ready")
        argv = json.loads((self.tmp / "vllm-argv.json").read_text())
        self.assertNotIn("--reasoning-parser", argv)
        session = json.loads(next((self.root / "serve" / "sessions").glob("*/session.json")).read_text())
        reasoning = session["reasoning"]
        self.assertEqual(reasoning["schema"], "lilith/serve-reasoning@1")
        self.assertEqual((reasoning["parser"], reasoning["enabled"], reasoning["requested"]),
                         (None, False, "auto"))

    def test_reasoning_parser_none_explicitly_disables_it(self):
        process = self.launch(LILITH_SERVE_REASONING_PARSER="none")
        self.assertEqual(self.wait_state(process, "ready", "failed")["state"], "ready")
        argv = json.loads((self.tmp / "vllm-argv.json").read_text())
        self.assertNotIn("--reasoning-parser", argv)
        session = json.loads(next((self.root / "serve" / "sessions").glob("*/session.json")).read_text())
        self.assertEqual(session["reasoning"]["requested"], "none")
        self.assertFalse(session["reasoning"]["enabled"])

    def test_reasoning_parser_auto_picks_gemma4_when_registered(self):
        fake = self._fake_reasoning_registry({"gemma4": "gemma4_reasoning_parser",
                                               "deepseek_r1": "deepseek_r1_reasoning_parser"})
        process = self.launch(PYTHONPATH=str(fake))
        self.assertEqual(self.wait_state(process, "ready", "failed")["state"], "ready")
        argv = json.loads((self.tmp / "vllm-argv.json").read_text())
        self.assertEqual(argv[argv.index("--reasoning-parser") + 1], "gemma4")
        session = json.loads(next((self.root / "serve" / "sessions").glob("*/session.json")).read_text())
        reasoning = session["reasoning"]
        self.assertEqual((reasoning["parser"], reasoning["enabled"]), ("gemma4", True))
        self.assertEqual(reasoning["registry_source"], "vllm.reasoning.ReasoningParserManager")
        self.assertEqual(reasoning["available"], ["deepseek_r1", "gemma4"])

    def test_reasoning_parser_explicit_name_is_honored_when_registered(self):
        fake = self._fake_reasoning_registry({"gemma4": "x", "deepseek_r1": "y"})
        process = self.launch(PYTHONPATH=str(fake), LILITH_SERVE_REASONING_PARSER="deepseek_r1")
        self.assertEqual(self.wait_state(process, "ready", "failed")["state"], "ready")
        argv = json.loads((self.tmp / "vllm-argv.json").read_text())
        self.assertEqual(argv[argv.index("--reasoning-parser") + 1], "deepseek_r1")

    def test_reasoning_parser_explicit_name_not_registered_is_a_refusal(self):
        fake = self._fake_reasoning_registry({"gemma4": "x"})
        result = subprocess.run(["bash", str(HERE / "serve_pod.sh")],
                                env=self.env(PYTHONPATH=str(fake), LILITH_SERVE_REASONING_PARSER="nope"),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 1)
        self.assertIn("error[lilith::serve::reasoning_parser]", result.stderr)
        self.assertIn("gemma4", result.stderr)
        self.assertIn("help:", result.stderr)
        self.assertFalse((self.tmp / "vllm-argv.json").exists(), "vLLM must not start after an explicit refusal")
        status = json.loads((self.root / "serve" / "status.json").read_text())
        self.assertEqual(status["state"], "failed")

    def test_reasoning_auto_with_nothing_registered_does_not_refuse(self):
        # A registry that exists but has no gemma4 entry: still non-fatal,
        # unlike the equivalent tool-parser case.
        fake = self._fake_reasoning_registry({"deepseek_r1": "y"})
        process = self.launch(PYTHONPATH=str(fake))
        self.assertEqual(self.wait_state(process, "ready", "failed")["state"], "ready")
        argv = json.loads((self.tmp / "vllm-argv.json").read_text())
        self.assertNotIn("--reasoning-parser", argv)
        session = json.loads(next((self.root / "serve" / "sessions").glob("*/session.json")).read_text())
        reasoning = session["reasoning"]
        self.assertFalse(reasoning["enabled"])
        self.assertEqual(reasoning["available"], ["deepseek_r1"])

    def test_the_registry_is_read_from_reasoning_parser_manager_when_vllm_imports(self):
        fake = self._fake_reasoning_registry({"gemma4": "gemma4_reasoning_parser"})
        result = subprocess.run([sys.executable, str(HERE / "serve_tools.py"), "resolve-reasoning",
                                 "--requested", "auto"],
                                env={**os.environ, "PYTHONPATH": str(fake)}, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        selection = json.loads(result.stdout)
        self.assertEqual(selection["schema"], "lilith/serve-reasoning@1")
        self.assertEqual(selection["parser"], "gemma4")
        self.assertTrue(selection["enabled"])
        self.assertEqual(selection["registry_source"], "vllm.reasoning.ReasoningParserManager")

    def test_resolve_reasoning_none_never_imports_vllm(self):
        # No PYTHONPATH override and no real vLLM installed: if `none` still
        # tried a registry lookup it would either raise or hang on import;
        # it must short-circuit before that.
        result = subprocess.run([sys.executable, str(HERE / "serve_tools.py"), "resolve-reasoning",
                                 "--requested", "none"], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        selection = json.loads(result.stdout)
        self.assertEqual((selection["parser"], selection["enabled"]), (None, False))

    def test_the_payload_ships_every_serving_file(self):
        sys.path.insert(0, str(LILITH / "runpod"))
        import verify_payload
        payload = set(verify_payload.PAYLOAD_FILES)
        for name in ("serve_pod.sh", "serve_adapter.py", "log_redact.py", "serve_tools.py"):
            self.assertIn(f"scripts/lilith/serve/{name}", payload)
        closure = subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import exam.certify, os; "
             "print('\\n'.join(sorted(m.__file__ for m in list(sys.modules.values()) "
             "if getattr(m, '__file__', None) and m.__file__.startswith(sys.argv[1]))))", str(LILITH)],
            capture_output=True, text=True, check=True).stdout.split()
        repo = LILITH.parents[1]
        for path in closure:
            self.assertIn(str(Path(path).relative_to(repo)), payload)


if __name__ == "__main__":
    unittest.main()
