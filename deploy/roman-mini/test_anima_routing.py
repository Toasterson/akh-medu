"""Phase 53g PR 6: data-driven Anima routing replaces laya_controller.host_action.

- The Python filter evaluator agrees with the shared fixture that the Rust
  evaluator (`route_filter.rs`) also runs.
- For every fixture snapshot, the filters `full_akh.py` renders route to the
  same role as `laya_controller.host_action`.
- `compare_anima_shadow.py` reads both databases read-only and reports
  agreement.

Regenerate the shared fixture after changing the routing table with
`python3 test_anima_routing.py --regenerate`.
"""
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_anima_shadow as cmp  # noqa: E402
import full_akh as fa  # noqa: E402
from laya_controller import host_action  # noqa: E402
import route_filter as rf  # noqa: E402

HERE = Path(__file__).resolve().parent
CASES = HERE / "fixtures/route_filter_cases.json"
SNAPSHOTS = HERE / "fixtures/host_action_snapshots.json"
PROPOSAL = {"choice": "investigate", "classifier": "retired"}

# Operator semantics, independent of the Roman routing table.
GENERIC = {"current": {"current_phase": "parked", "parked_reason": "lane-circuit-breaker: 3", "live_sessions": 0,
                       "flags": {"a/b": True, "t~x": 1}, "nothing": None, "list": ["x", "y"]}}
GENERIC_CASES = [
    ("eq string", {"eq": {"pointer": "/current/current_phase", "value": "parked"}}, True),
    ("eq other string", {"eq": {"pointer": "/current/current_phase", "value": "working"}}, False),
    ("eq int", {"eq": {"pointer": "/current/live_sessions", "value": 0}}, True),
    ("eq int vs float", {"eq": {"pointer": "/current/live_sessions", "value": 0.0}}, False),
    ("eq int vs bool", {"eq": {"pointer": "/current/live_sessions", "value": False}}, False),
    ("eq missing", {"eq": {"pointer": "/current/missing", "value": None}}, False),
    ("eq null", {"eq": {"pointer": "/current/nothing", "value": None}}, True),
    ("in hit", {"in": {"pointer": "/current/current_phase", "values": ["draft", "parked"]}}, True),
    ("in miss", {"in": {"pointer": "/current/parked_reason", "values": ["too-big"]}}, False),
    ("prefix hit", {"prefix": {"pointer": "/current/parked_reason", "value": "lane-circuit-breaker"}}, True),
    ("prefix on number", {"prefix": {"pointer": "/current/live_sessions", "value": "0"}}, False),
    ("exists escaped slash", {"exists": {"pointer": "/current/flags/a~1b"}}, True),
    ("eq escaped tilde", {"eq": {"pointer": "/current/flags/t~0x", "value": 1}}, True),
    ("exists null", {"exists": {"pointer": "/current/nothing"}}, False),
    ("exists missing", {"exists": {"pointer": "/current/missing"}}, False),
    ("array index", {"eq": {"pointer": "/current/list/1", "value": "y"}}, True),
    ("array leading zero", {"exists": {"pointer": "/current/list/01"}}, False),
    ("not missing eq", {"not": {"eq": {"pointer": "/current/missing", "value": "x"}}}, True),
    ("all", {"all": [{"eq": {"pointer": "/current/current_phase", "value": "parked"}},
                     {"exists": {"pointer": "/current/parked_reason"}}]}, True),
    ("all one false", {"all": [{"eq": {"pointer": "/current/current_phase", "value": "parked"}},
                               {"exists": {"pointer": "/current/missing"}}]}, False),
    ("any", {"any": [{"exists": {"pointer": "/current/missing"}},
                     {"eq": {"pointer": "/current/live_sessions", "value": 0}}]}, True),
    ("whole document", {"exists": {"pointer": ""}}, True),
]


def snapshots():
    return json.loads(SNAPSHOTS.read_text())["snapshots"]


def laya_role(snapshot):
    return host_action(snapshot, PROPOSAL)[1]


def build_cases():
    """The shared fixture: generic operator cases plus, per snapshot and per
    rendered role filter, whether that role's route matches."""
    cases = [{"name": name, "filter": f, "data": GENERIC, "expected": expected}
             for name, f, expected in GENERIC_CASES]
    filters = fa.anima_route_filters()
    for entry in snapshots():
        event_type, data = cmp.event_from_laya_snapshot(entry["snapshot"])
        if event_type != fa.WORK_ITEM_EVENT:
            continue
        role = laya_role(entry["snapshot"])
        for target, when in filters.items():
            cases.append({"name": f"host_action {entry['name']} -> {target}", "filter": when, "data": data,
                          "expected": role == target})
    return {"comment": "Shared by route_filter.rs and test_anima_routing.py; regenerate with "
                       "`python3 test_anima_routing.py --regenerate`.",
            "cases": cases}


class FilterEvaluatorTests(unittest.TestCase):
    def test_shared_fixture_is_current_and_python_agrees(self):
        fixture = json.loads(CASES.read_text())
        self.assertEqual(fixture, build_cases(), "fixture is stale: run test_anima_routing.py --regenerate")
        for case in fixture["cases"]:
            rf.validate(case["filter"])
            self.assertEqual(rf.matches(case["filter"], case["data"]), case["expected"], case["name"])

    def test_invalid_filters_are_refused(self):
        for bad in ({"eq": {"pointer": "current", "value": 1}}, {"exists": {"pointer": "/a~2"}}, {"all": []},
                    {"in": {"pointer": "/a", "values": []}}, {"prefix": {"pointer": "/a", "value": ""}},
                    {"gt": {"pointer": "/a", "value": 1}}, {"eq": {"pointer": "/a", "value": 1, "x": 2}},
                    {"eq": {"pointer": "/a", "value": 1}, "not": {"exists": {"pointer": "/a"}}}):
            with self.assertRaises(rf.FilterError, msg=json.dumps(bad)):
                rf.validate(bad)
        deep = {"exists": {"pointer": "/a"}}
        for _ in range(16):
            deep = {"not": deep}
        with self.assertRaises(rf.FilterError):
            rf.validate(deep)

    def test_rendered_filters_validate(self):
        for route in fa.anima_routes():
            if "when" in route:
                rf.validate(route["when"])
        source = fa.anima_source(live=False)
        for key in ("when", "park_event", "live_session", "thread_message"):
            rf.validate(source["hydrate_parked"][key])
        rf.validate(source["project_filter"])


class HostActionParityTests(unittest.TestCase):
    def test_filters_route_every_fixture_like_host_action(self):
        routes = fa.anima_routes()
        seen = set()
        for entry in snapshots():
            event_type, data = cmp.event_from_laya_snapshot(entry["snapshot"])
            expected = laya_role(entry["snapshot"])
            roles = cmp.routed_roles(routes, fa.ANIMA_SOURCE, event_type, data)
            self.assertEqual(roles, {expected} if expected else set(), entry["name"])
            seen.add(expected)
        # Every host_action outcome is exercised: nobody and all three roles.
        self.assertEqual(seen, {None, "janus", "minerva", "vesta"})

    def test_filters_are_mutually_exclusive(self):
        filters = fa.anima_route_filters()
        for entry in snapshots():
            _, data = cmp.event_from_laya_snapshot(entry["snapshot"])
            self.assertLessEqual(sum(rf.matches(f, data) for f in filters.values()), 1, entry["name"])

    def test_nothing_in_the_table_is_prose(self):
        # Reason codes are kebab-case machine codes; prefixes stop at the code.
        for code in fa.HOLD_REASONS + fa.INFRA_REASONS + fa.PRODUCT_REASONS + fa.HUMAN_GATE_PREFIXES + fa.INFRA_PREFIXES:
            self.assertRegex(code, r"^[a-z]+(-[a-z]+)*$")


def laya_db(path: Path, rows):
    db = sqlite3.connect(path)
    db.execute("create table decisions(id text primary key, entity text not null, snapshot text not null, "
               "proposal text not null, action text not null, role text, state text not null, event_id text, "
               "created_at real not null)")
    for index, (entity, snapshot, role, action, created_at) in enumerate(rows):
        db.execute("insert into decisions values (?,?,?,?,?,?,?,?,?)",
                   (f"d{index}", entity, json.dumps(snapshot), "{}", action, role, "pending", None, created_at))
    db.commit()
    db.close()


def events_db(path: Path, records):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE records_v1 (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
    for index, record in enumerate(records):
        db.execute("insert into records_v1 values (?,?)",
                   (f"anima_shadow/11:roman-anima37:{record['recorded_at_ms']:020}-{index:016x}",
                    json.dumps(record).encode()))
    # Unrelated keys around the prefix must be ignored.
    db.execute("insert into records_v1 values (?,?)", ("anima_shadow_count/11:roman-anima", b"3"))
    db.execute("insert into records_v1 values (?,?)", ("anima_entity/x", b"{}"))
    db.commit()
    db.close()


def shadow(entity, akhs, at_ms, source="roman-anima"):
    return {"source": source, "recorded_at_ms": at_ms,
            "event": {"schema_version": 1, "id": entity + "@x", "source": source, "type": fa.WORK_ITEM_EVENT,
                      "occurred_at": at_ms, "subject": entity, "data": {}},
            "routes": [{"route": f"{a}-{fa.WORK_ITEM_EVENT}", "akh": a} for a in akhs]}


class CompareScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "state/events").mkdir(parents=True)
        by_name = {e["name"]: e["snapshot"] for e in snapshots()}
        now = 2_000_000_000.0
        self.now = now
        rows = [
            ("work_item:w27", by_name["parked-too-big"], "minerva", "product", now - 10),
            ("work_item:w23", by_name["parked-ci-infra-failing"], "vesta", "infrastructure", now - 10),
            ("work_item:w13", by_name["parked-no-progress"], "janus", "investigate", now - 10),
            ("work_item:w18", by_name["parked-operator-hold"], None, "preserve_hold", now - 10),
            ("work_item:w14", by_name["parked-retry-budget"], "janus", "investigate", now - 10),
            # Outside the window.
            ("work_item:w30", by_name["parked-no-progress-live"], None, "wait_for_owner", now - 90_000),
        ]
        laya_db(self.root / "state/laya-decisions.sqlite3", rows)
        ms = int(now * 1000)
        events_db(self.root / "state/events/akheventd.sqlite3", [
            shadow("work_item:w27", ["minerva"], ms - 5000),
            shadow("work_item:w23", ["vesta"], ms - 5000),
            shadow("work_item:w13", ["minerva"], ms - 5000),   # disagrees
            shadow("work_item:w18", [], ms - 5000),            # held: routed nowhere
            shadow("work_item:w99", ["janus"], ms - 5000),     # shadow only
            shadow("work_item:w27", ["minerva"], ms - 200_000_000),  # outside the window
            shadow("work_item:w23", ["janus"], ms - 5000, source="other"),
        ])

    def tearDown(self):
        self.temp.cleanup()

    def run_json(self, *args):
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = cmp.main(["--root", str(self.root), "--since", str(self.now - 3600), "--json", *args])
        return status, json.loads(out.getvalue())

    def test_shadow_comparison_reports_agreement_and_differences(self):
        before = {p: p.stat().st_mtime_ns for p in (self.root / "state").rglob("*")}
        status, report = self.run_json()
        self.assertEqual(status, 1)
        self.assertEqual(report["routed_by_both"], 3)
        self.assertEqual(report["same_roles"], 2)
        self.assertEqual(report["mismatches"], [{"entity": "work_item:w13", "laya": ["janus"], "shadow": ["minerva"]}])
        self.assertEqual(report["laya_only"], ["work_item:w14"])
        self.assertEqual(report["shadow_only"], ["work_item:w99"])
        # Read-only: no file changed and no journal was created.
        after = {p: p.stat().st_mtime_ns for p in (self.root / "state").rglob("*")}
        self.assertEqual(before, after)

    def test_shadow_json_dump_is_accepted(self):
        dump = self.root / "shadow.json"
        dump.write_text(json.dumps([shadow("work_item:w27", ["minerva"], int(self.now * 1000) - 1000)]))
        status, report = self.run_json("--shadow-json", str(dump))
        self.assertEqual(status, 0)
        self.assertEqual((report["routed_by_both"], report["same_roles"]), (1, 1))

    def test_replay_matches_host_action_on_recorded_snapshots(self):
        status, report = self.run_json("--replay")
        self.assertEqual(status, 0)
        self.assertEqual(report["agree"], report["decisions"])
        self.assertEqual(report["decisions"], 6)

    def test_missing_database_is_an_input_error(self):
        with self.assertRaises(FileNotFoundError):
            cmp.connect_ro(self.root / "missing.sqlite3")


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        CASES.write_text(json.dumps(build_cases(), indent=1, sort_keys=True) + "\n")
        print(f"wrote {CASES}")
        sys.exit(0)
    unittest.main()
