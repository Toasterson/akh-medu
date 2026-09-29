"""Rendering tests for the full-akh Roman cutover (Phase 53f, ADR 113).

Renders a synthetic deployment root into a temporary directory and checks the
TOML/JSON shapes akhomed and akheventd will load. No git remote, network or
real credential is used.
"""
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11: the CI runner has 3.10
    tomllib = None

NEEDS_TOMLLIB = unittest.skipIf(tomllib is None, "needs Python >= 3.11 (tomllib)")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import configure_core  # noqa: E402
import full_akh as fa  # noqa: E402
import retarget_events  # noqa: E402
import apply_reactor_seeds  # noqa: E402

SOURCE = Path(__file__).resolve().parent
ROLES = fa.ROLES
IDS = {"janus": "01a0abc4-7c16-76f1-bda5-ae879cc3e919", "minerva": "01a0abc4-7e7a-7053-bfd7-a6c9181a2827",
       "vesta": "01a0abc4-8129-7870-b3e1-64385b12a186"}


def mcp_tool(name, service, tool):
    return {"name": name, "description": "", "input_schema": {"type": "object"}, "read_only": False,
            "destination": {"kind": "mcp", "service": service, "tool": tool}}


def fixture(root: Path) -> None:
    for d in ("config", "secrets", "bin", "launchd", "evidence", "logs", "core-config/akh-medu/workspaces"):
        (root / d).mkdir(parents=True, exist_ok=True)
    keys = ["core", "operator", "observer-tool", "janus-escalation", "vesta-ops"]
    keys += [f"t3-{r}" for r in ROLES] + [f"laya-{r}" for r in ROLES]
    for key in keys:
        (root / f"secrets/{key}.key").write_text(f"test-{key}-" + "x" * 32 + "\n")
    # PR 5 / D2: the Grafana service-account token and the exported lab CA.
    (root / fa.GRAFANA_TOKEN_FILE).write_text("glsa_" + "x" * 32 + "\n")
    (root / fa.LAB_CA_FILE).write_text("-----BEGIN CERTIFICATE-----\n")
    endpoints = {
        "anima": {"url": "https://anima.example/mcp", "token_env": "ROMAN_ANIMA_TOKEN"},
        "seshat": {"url": "http://seshat.lab.home/mcp", "token_env": "ROMAN_ARCHIVE_TOKEN", "allow_insecure_http": True},
        "sokar": {"url": "http://127.0.0.1:18439/mcp", "token_env": "ROMAN_ARCHIVE_TOKEN"},
        "observer": {"url": "http://127.0.0.1:18436/mcp", "token_file": str(root / "secrets/observer-tool.key")},
        "janus-escalation": {"url": "http://127.0.0.1:18438/mcp", "token_file": str(root / "secrets/janus-escalation.key")},
        "vesta-ops": {"url": "http://127.0.0.1:18437/mcp", "token_file": str(root / "secrets/vesta-ops.key")},
    }
    akhs, automations = {}, {}
    for role in ROLES:
        endpoints[f"board-{role}"] = {"url": "http://127.0.0.1:18089/mcp", "token_env": f"ROMAN_BOARD_{role.upper()}"}
        endpoints[f"laya-{role}"] = {"url": "http://127.0.0.1:18447/mcp", "token_file": str(root / f"secrets/laya-{role}.key")}
        tools = [{"name": "kg_query", "destination": {"kind": "cognitive", "service": "core", "tool": "kg_query"}},
                 mcp_tool("anima__approve_and_merge", "anima", "approve_and_merge"),
                 mcp_tool("anima__get_session_retirement", "anima", "get_session_retirement"),
                 mcp_tool("seshat__search", "seshat", "search"),
                 mcp_tool("sokar__put", "sokar", "put"),
                 mcp_tool("board__board_send_inbox", f"board-{role}", "board_send_inbox"),
                 mcp_tool("board__board_agents", f"board-{role}", "board_agents"),
                 mcp_tool("monitoring_snapshot", "observer", "monitoring_snapshot"),
                 mcp_tool("record_fixer_result", f"laya-{role}", "record_fixer_result")]
        if role == "janus":
            tools += [mcp_tool(name, "janus-escalation", name) for name in
                      ("escalate_to_astra", "astra_escalation_status", "dispatch_mergiraf", "mergiraf_status")]
        if role == "vesta":
            tools += [mcp_tool(name, "vesta-ops", name) for name in
                      ("gitops_exec", "homelab_inspect", "homelab_write", "gitops_publish")]
        akhs[role] = {"core_service": "core", "instructions": "", "tools": tools, "workspace_profiles": [role]}
        if role == "janus":
            akhs[role]["workspace_profiles"] += ["janus-worker-akh-medu", "janus-worker-droneworld"]
        wake = ("You are now on duty. Coordinate with board_send_inbox. " + json.dumps(IDS) +
                "\nReport on thread " + fa.STATUS_THREAD + ".")
        if role == "janus":
            wake += ("\n# Janus merge authority\nUse anima__approve_and_merge."
                     "\n# Astra escalation contract\nUse escalate_to_astra.")
        if role == "vesta":
            wake += "\nVesta has homelab_inspect for live Kubernetes."
        automations[role] = {"id": role, "instructions": wake}
    principals = [{"id": "runtime", "token_file": str(root / "secrets/core.key"), "roles": ["runtime"], "akhs": list(ROLES)}]
    principals += [{"id": f"t3-{r}", "token_file": str(root / f"secrets/t3-{r}.key"), "roles": ["client"], "akhs": [r]} for r in ROLES]
    workspaces = {name: {"revision": "1", "site": "armini", "provisioner": {"kind": "existing", "executor": "executor",
                                                                             "root": str(root / "workspaces" / name)}}
                  for name in ("janus", "minerva", "vesta", "janus-worker-akh-medu", "janus-worker-droneworld")}
    (root / "config/sessions.json").write_text(json.dumps(
        {"endpoints": endpoints, "principals": principals, "akhs": akhs, "automations": automations,
         "workspaces": workspaces, "listen": "127.0.0.1:18431"}))
    (root / "config/vesta-grafana-tools.json").write_text(json.dumps(
        [{"name": "query_prometheus", "description": ""}, {"name": "list_datasources", "description": ""}]))
    # The retired broker's registry: its keys are the registered projects.
    (root / "config/janus-escalation.json").write_text(json.dumps(
        {"projects": {"akh-medu": "t3-project-1", "Droneworld": "t3-project-2"}}))
    routes, ev_endpoints = [], {"sessions": {"url": "http://127.0.0.1:18431", "token_file": str(root / "secrets/core.key")}}
    for role in ROLES:
        ev_endpoints[f"board-{role}"] = {"url": "http://127.0.0.1:18089", "token_env": f"ROMAN_BOARD_{role.upper()}"}
        routes.append({"id": f"{role}-djehuty.inbox", "source": f"roman-inbox-{role}", "event_type": "djehuty.inbox",
                       "automation": role, "coalesce_pending": False})
        src, typ = (("roman-alerts", "monitoring.changed") if role == "vesta" else ("roman-anima", f"anima.{role}.changed"))
        routes.append({"id": f"{role}-{typ}", "source": src, "event_type": typ, "automation": role,
                       "coalesce_pending": role == "vesta"})
        routes.append({"id": f"laya-{role}", "source": src, "event_type": f"laya.{role}.intervention",
                       "automation": role, "coalesce_pending": False})
    (root / "config/events.json").write_text(json.dumps({"service_id": "events", "endpoints": ev_endpoints, "routes": routes}))
    (root / "config/core.json").write_text(json.dumps({
        "listen": fa.CORE_LISTEN, "data_dir": str(root / "state/core"), "sqlite": True,
        "principals": [{"id": "sessions", "token_file": str(root / "secrets/core.key"), "akhs": list(ROLES)}]}))
    # A pre-existing workspace whose unrelated keys must survive, with the old
    # (wrong for cloud) endpoint and autonomy disabled.
    (root / "core-config/akh-medu/workspaces/janus.toml").write_text(
        'name = "janus"\ndimension = 10000\nseed_packs = ["identity"]\nsystem_prompt = "old"\n'
        '[ollama_endpoint]\nkind = "local"\nbase_url = "http://localhost:11434"\n'
        '[autonomy]\nenabled = false\ntasks = {}\n[mcp]\nrequire_pushed_mcp = false\n[exec]\nkind = "none"\n')


@NEEDS_TOMLLIB
class RenderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fixture(self.root)
        self.prov = fa.Provenance(commit="c" * 40, short="c" * 12, origin_main="c" * 40, committed_at=0)
        self.binary = self.root / "bin" / self.prov.binary_name
        self.binary.write_text("#!/bin/sh\n")
        self.warnings = configure_core.render(self.root, self.prov, self.binary, "0" * 64, source=SOURCE)

    def tearDown(self):
        self.temp.cleanup()

    def toml(self, rel):
        return tomllib.loads((self.root / rel).read_text())

    def test_host_config_is_deployed_full_akh(self):
        host = self.toml("core-config/akh-medu/config.toml")
        self.assertEqual(host["deployment"], "deployed")
        self.assertEqual(host["daemon"]["auto_start"], list(ROLES))
        self.assertEqual(host["exec"], {"kind": "none"})
        self.assertEqual(host["acp"]["tcp_port"], fa.ACP_TCP_PORT)
        scoped = {p["id"]: p["akhs"] for p in host["acp"]["principals"]}
        self.assertEqual(scoped["t3-minerva"], ["minerva"])
        self.assertEqual(scoped["operator"], list(ROLES))
        self.assertTrue(host["database"]["url"].startswith("sqlite://"))

    def test_workspace_is_cloud_autonomous_with_mcp_servers(self):
        for role in ROLES:
            ws = self.toml(f"core-config/akh-medu/workspaces/{role}.toml")
            self.assertEqual(ws["model"], "glm-5.3")
            self.assertEqual(ws["ollama_endpoint"], fa.OLLAMA_CLOUD)
            self.assertTrue(ws["autonomy"]["enabled"])
            self.assertEqual(ws["autonomy"]["budget"], {"max_tokens_per_day": 250000, "max_turns_per_day": 40})
            self.assertTrue(ws["capture_transcripts"])
            self.assertEqual(ws["exec"], {"kind": "none"})
            self.assertEqual(ws["board"], {"url": "http://127.0.0.1:18089",
                                           "auth": {"token_env": f"ROMAN_BOARD_{role.upper()}"}})
            servers = {s["name"]: s for s in ws["mcp"]["servers"]}
            expected = {"anima", "seshat", "sokar", "board", "laya"}
            expected |= {"vesta": {"vesta-ops", "grafana"}}.get(role, set())
            self.assertEqual(set(servers), expected)
            self.assertEqual(servers["board"]["auth"], {"token_env": f"ROMAN_BOARD_{role.upper()}"})
            self.assertEqual(servers["laya"]["auth"], {"token_file": str(self.root / f"secrets/laya-{role}.key")})
            self.assertEqual(servers["laya"]["tools"], ["record_fixer_result"])
            self.assertTrue(servers["seshat"]["allow_insecure_http"])
            self.assertNotIn("allow_insecure_http", servers["sokar"])
            if role == "vesta":
                self.assertEqual(servers["grafana"]["url"], fa.GRAFANA_URL)
                self.assertEqual(servers["grafana"]["tools"], ["list_datasources", "query_prometheus"])
                self.assertNotIn("auth", servers["grafana"])
                self.assertNotIn("allow_insecure_http", servers["grafana"])
            else:
                self.assertNotIn("grafana", servers)
            self.assertEqual(ws["monitoring"], {
                "endpoint": "http://127.0.0.1:18435", "source": "roman-alerts",
                "token_file": str(self.root / "secrets/monitoring-reader.key"),
                "observer_status_file": str(self.root / "state/observer-status.json")})
            self.assertTrue(ws["system_prompt"].startswith((SOURCE / "roles/common.md").read_text()))
            self.assertIn((SOURCE / f"roles/{role}.md").read_text(), ws["system_prompt"])
        self.assertEqual(self.warnings[1:], [])

    def test_existing_workspace_keys_survive_and_scalars_precede_tables(self):
        text = (self.root / "core-config/akh-medu/workspaces/janus.toml").read_text()
        ws = tomllib.loads(text)
        self.assertEqual(ws["dimension"], 10000)
        self.assertEqual(ws["seed_packs"], ["identity"])
        self.assertNotEqual(ws["system_prompt"], "old")
        first_table = text.index("\n[")
        for key in ("name", "model", "system_prompt", "capture_transcripts"):
            self.assertLess(text.index(f"\n{key} = "), first_table, key)

    def test_prompts_use_53i_tool_names(self):
        prompts = [self.toml(f"core-config/akh-medu/workspaces/{r}.toml")["system_prompt"] for r in ROLES]
        for role in ROLES:
            seeds = json.loads((self.root / f"config/reactor-seeds/{role}.json").read_text())
            prompts += [a["schedule"]["action"]["prompt"] for a in seeds["apply"]]
        legacy = re.compile(r"(?<![\w-])(anima__|board__|seshat__|sokar__)")
        for prompt in prompts:
            self.assertIsNone(legacy.search(prompt), prompt)
            self.assertNotRegex(prompt, r"(?<!__)\bboard_send_inbox\b")
        janus = prompts[0]
        self.assertIn("mcp__anima__approve_and_merge", janus)
        for native in ("escalate", "dispatch_merge_repair", "escalation_status"):
            self.assertRegex(janus, rf"(?<![\w-]){native}(?!\w)")
        for prompt in prompts:
            self.assertNotIn("janus-escalation", prompt)
            self.assertNotRegex(prompt, r"escalate_to_astra|astra_escalation_status|dispatch_mergiraf|mergiraf_status")
        self.assertIn("mcp__vesta-ops__homelab_inspect", prompts[2])
        self.assertIn("mcp__grafana__", prompts[2])
        self.assertNotIn("mcp__vesta-ops__grafana__", prompts[2])
        self.assertEqual(prompts[2].count("grafana__"), prompts[2].count("mcp__grafana__"),
                         "every grafana__ mention must be qualified mcp__grafana__")

    def test_janus_delegates_escalations_natively(self):
        janus = self.toml("core-config/akh-medu/workspaces/janus.toml")
        delegation = janus["delegation"]
        self.assertFalse(delegation["enabled"], "the generic delegate tool stays off")
        native = delegation["native"]
        self.assertTrue(native["enabled"])
        self.assertEqual(native["url"], "http://127.0.0.1:18431")
        self.assertEqual(native["token_file"], str(self.root / fa.DELEGATION_TOKEN))
        self.assertEqual(native["workspace_profile"], "janus")
        self.assertEqual(native["core_principal"], "sessions")
        self.assertEqual(native["max_new_per_day"], 4)
        self.assertEqual(native["profiles"], {
            "Droneworld": {"workspace_profile": "janus-worker-droneworld", "runtime_profile": "codex-astra"},
            "akh-medu": {"workspace_profile": "janus-worker-akh-medu", "runtime_profile": "codex-astra"}})
        escalation = native["escalation"]
        self.assertTrue(escalation["enabled"])
        self.assertEqual(escalation["report_thread"], fa.STATUS_THREAD)
        for key, name in (("escalation_skill", "astra-escalation"), ("merge_repair_skill", "mergiraf")):
            skill = Path(escalation[key])
            self.assertEqual(skill, self.root / "skills" / name)
            manifest = json.loads((skill / "skill.json").read_text())
            self.assertEqual(manifest["id"], name)
            self.assertTrue((skill / manifest["procedures_file"]).read_text().strip())
        token = self.root / fa.DELEGATION_TOKEN
        self.assertEqual(token.stat().st_mode & 0o777, 0o600)
        self.assertNotIn("janus-escalation", {s["name"] for s in janus["mcp"]["servers"]})
        for role in ("minerva", "vesta"):
            self.assertNotIn("delegation", self.toml(f"core-config/akh-medu/workspaces/{role}.toml"))
        # D4 is in the merge worker's procedures, not in code.
        mergiraf = (self.root / "skills/mergiraf/procedures.md").read_text()
        self.assertIn("fast-forward", mergiraf)
        self.assertIn("Never force-push", mergiraf)
        self.assertIn("Never merge the PR", mergiraf)

    def test_observer_monitoring_moves_to_the_native_tool(self):
        text = (self.root / "core-config/akh-medu/workspaces/vesta.toml").read_text()
        # [monitoring] is a table after [board] and before [[mcp.servers]].
        self.assertLess(text.index("\n[board]"), text.index("\n[monitoring]"))
        self.assertLess(text.index("\n[monitoring]"), text.index("\n[[mcp.servers]]"))
        self.assertNotIn("observer-tool.key", text)
        for role in ROLES:
            prompt = self.toml(f"core-config/akh-medu/workspaces/{role}.toml")["system_prompt"]
            self.assertNotIn("mcp__observer__", prompt)
            self.assertIn("monitoring_snapshot is a native read-only tool", prompt)
        session = json.loads((self.root / "config/sessions.json").read_text())
        self.assertEqual(fa.tool_renames(session, "vesta")["monitoring_snapshot"], "monitoring_snapshot")
        self.assertEqual(fa.rename_tools("call mcp__observer__monitoring_snapshot now", {}),
                         "call monitoring_snapshot now")

    def test_observer_server_stays_for_tools_that_did_not_move(self):
        session = json.loads((self.root / "config/sessions.json").read_text())
        session["akhs"]["janus"]["tools"].append(mcp_tool("coverage_report", "observer", "coverage_report"))
        servers, _ = fa.mcp_servers(session, "janus", self.root)
        observer = {s["name"]: s for s in servers}["observer"]
        self.assertEqual(observer["tools"], ["coverage_report"])
        # A role without the monitoring grant gets no [monitoring] table.
        session["akhs"]["minerva"]["tools"] = [t for t in session["akhs"]["minerva"]["tools"]
                                               if t["name"] != "monitoring_snapshot"]
        self.assertIsNone(fa.monitoring_config(session, "minerva", self.root))

    def test_staged_events_carry_the_roman_alerts_monitor(self):
        staged = json.loads((self.root / "config/events.akh-target.json").read_text())
        [monitor] = staged["monitors"]
        self.assertEqual((monitor["source"], monitor["event_type"], monitor["interval_secs"]),
                         ("roman-alerts", "monitoring.changed", 60))
        self.assertEqual([h["name"] for h in monitor["health"]],
                         ["core", "sessions", "runtime", "runtime-minerva", "runtime-vesta", "events"])
        endpoints = staged["endpoints"]
        for check in monitor["health"]:
            self.assertEqual(endpoints[check["endpoint"]]["anonymous"], True)
            self.assertNotIn("token_file", endpoints[check["endpoint"]])
        self.assertEqual(endpoints["health-runtime-vesta"]["url"], "http://127.0.0.1:18443")
        grafana = endpoints[monitor["alertmanager"]["endpoint"]]
        self.assertEqual(grafana, {"url": "https://grafana.lab.home",
                                   "token_file": str(self.root / "secrets/grafana-sa.token"),
                                   "ca_file": str(self.root / "secrets/lab-ca.pem")})
        self.assertEqual(monitor["alertmanager"]["path"], "/api/alertmanager/grafana/api/v2/alerts")
        self.assertEqual(monitor["prometheus"]["query"], 'ALERTS{alertstate="firing"}')
        self.assertEqual(endpoints["victoriametrics"], {"url": "http://127.0.0.1:18440", "anonymous": True})
        reader = {p["id"]: p for p in staged["principals"]}["monitoring-reader"]
        self.assertEqual(reader["roles"], ["operator"])
        self.assertEqual((self.root / "secrets/monitoring-reader.key").stat().st_mode & 0o777, 0o600)
        # The Vesta route stays coalescing and the reactor seed is unchanged.
        route = {r["id"]: r for r in staged["routes"]}["vesta-monitoring.changed"]
        self.assertTrue(route["coalesce_pending"])
        self.assertEqual(route["target"], {"kind": "akh", "endpoint": "core", "akh": "vesta"})
        # Rendering again from the installed config is idempotent.
        again = fa.render_monitoring_events(staged, self.root)
        self.assertEqual(again, staged)

    def test_retarget_refuses_until_the_grafana_token_and_ca_exist(self):
        (self.root / fa.GRAFANA_TOKEN_FILE).unlink()
        warnings = configure_core.render(self.root, self.prov, self.binary, "0" * 64, source=SOURCE)
        self.assertTrue(any("grafana-sa.token" in w for w in warnings), warnings)
        with self.assertRaisesRegex(fa.RenderError, "grafana-sa.token"):
            retarget_events.retarget(self.root)
        live = json.loads((self.root / "config/events.json").read_text())
        self.assertNotIn("monitors", live)

    def test_rename_is_single_pass(self):
        renames = {"board__board_agents": "mcp__board__board_agents", "board_agents": "mcp__board__board_agents"}
        self.assertEqual(fa.rename_tools("use board__board_agents or board_agents", renames),
                         "use mcp__board__board_agents or mcp__board__board_agents")

    def test_grafana_prefix_prose_is_rewritten_to_the_direct_server(self):
        # The prose names a prefix, not an exact tool, so it isn't covered by
        # the per-tool `renames` map; `rename_tools` special-cases it.
        self.assertEqual(
            fa.rename_tools("Grafana MCP tools prefixed grafana__ support queries.", {}),
            "Grafana MCP tools prefixed mcp__grafana__ support queries.")

    def test_grafana_server_is_vesta_only(self):
        for role in ("janus", "minerva"):
            ws = self.toml(f"core-config/akh-medu/workspaces/{role}.toml")
            self.assertNotIn("grafana", {s["name"] for s in ws["mcp"]["servers"]})
        vesta = self.toml("core-config/akh-medu/workspaces/vesta.toml")
        servers = {s["name"]: s for s in vesta["mcp"]["servers"]}
        self.assertEqual(servers["grafana"], {
            "name": "grafana", "url": "http://127.0.0.1:18441/mcp",
            "tools": ["list_datasources", "query_prometheus"]})

    def test_missing_grafana_catalog_fails_rendering_closed(self):
        (self.root / "config/vesta-grafana-tools.json").unlink()
        with self.assertRaisesRegex(fa.RenderError, "vesta-grafana-tools.json"):
            configure_core.render(self.root, self.prov, self.binary, "0" * 64, source=SOURCE)

    def test_empty_grafana_catalog_fails_rendering_closed(self):
        (self.root / "config/vesta-grafana-tools.json").write_text("[]")
        with self.assertRaisesRegex(fa.RenderError, "no Grafana tools"):
            configure_core.render(self.root, self.prov, self.binary, "0" * 64, source=SOURCE)

    def test_core_json_scopes_sessions_events_operator(self):
        core = json.loads((self.root / "config/core.json").read_text())
        by_id = {p["id"]: p for p in core["principals"]}
        self.assertEqual(by_id["sessions"]["tools"], {"kind": "legacy_cognitive"})
        self.assertEqual(by_id["sessions"]["grants"], "optional")
        self.assertEqual(by_id["events"]["roles"], ["event"])
        self.assertEqual(by_id["events"]["tools"], {"kind": "named", "tools": []})
        self.assertIn("reactor_manage", by_id["operator"]["tools"]["tools"])
        self.assertIn("monitoring_snapshot", by_id["operator"]["tools"]["tools"])
        self.assertTrue(core["sqlite"])
        self.assertTrue((self.root / "secrets/core-events.key").exists())
        self.assertEqual((self.root / "secrets/core-events.key").stat().st_mode & 0o777, 0o600)

    def test_rerender_is_stable_and_keeps_generated_key(self):
        key = (self.root / "secrets/core-events.key").read_text()
        before = {p: p.read_bytes() for p in (self.root / "core-config").rglob("*.toml")}
        configure_core.render(self.root, self.prov, self.binary, "0" * 64, source=SOURCE)
        self.assertEqual(key, (self.root / "secrets/core-events.key").read_text())
        self.assertEqual(before, {p: p.read_bytes() for p in (self.root / "core-config").rglob("*.toml")})
        core = json.loads((self.root / "config/core.json").read_text())
        self.assertEqual([p["id"] for p in core["principals"]], ["sessions", "events", "operator"])

    def test_reactor_seeds_follow_routes_with_exit_conditions(self):
        vesta = json.loads((self.root / "config/reactor-seeds/vesta.json").read_text())
        schedules = {a["schedule"]["rule"]["source"]: a["schedule"] for a in vesta["apply"]}
        self.assertEqual(set(schedules), {"roman-inbox-vesta/djehuty.inbox", "roman-alerts/monitoring.changed",
                                          "roman-alerts/laya.vesta.intervention",
                                          "roman-anima/anima.work_item.changed"})
        for schedule in schedules.values():
            action = schedule["action"]
            self.assertEqual(action["kind"], "llm_turn")
            self.assertTrue(action["allow_tools"])
            self.assertGreater(len(action["exit_condition"]), 40)
            self.assertNotIn("Vesta has", action["prompt"])
        self.assertTrue(schedules["roman-alerts/monitoring.changed"]["coalesce"])
        self.assertFalse(schedules["roman-inbox-vesta/djehuty.inbox"]["coalesce"])
        self.assertEqual(schedules["roman-inbox-vesta/djehuty.inbox"]["action"]["report_thread"], fa.STATUS_THREAD)
        self.assertNotIn("report_thread", schedules["roman-alerts/monitoring.changed"]["action"])
        self.assertEqual([e["id"] for e in vesta["enable"]], [s["id"] for s in (a["schedule"] for a in vesta["apply"])])
        body = apply_reactor_seeds.requests_for(vesta, enable=False)[0]
        self.assertEqual((body["akh"], body["tool"], body["arguments"]["operation"]), ("vesta", "reactor_manage", "add"))

    # ── PR 6: Anima source, data-driven Laya routing, skill, schema ──────────

    def render_live(self):
        return configure_core.render(self.root, self.prov, self.binary, "0" * 64, source=SOURCE, live=True)

    def test_staged_anima_source_is_shadow_next_to_the_legacy_routes(self):
        staged = json.loads((self.root / "config/events.akh-target.json").read_text())
        [source] = staged["anima"]
        self.assertTrue(source["shadow"])
        self.assertEqual((source["source"], source["endpoint"], source["max_emissions_per_episode"]),
                         ("roman-anima", "anima", 2))
        self.assertEqual(source["fields"], fa.ANIMA_FIELDS)
        self.assertEqual(staged["endpoints"]["anima"], {"url": fa.ANIMA_MCP_URL, "token_env": "ROMAN_ANIMA_TOKEN"})
        self.assertNotIn("watchdog", staged, "the observer still owns the watchdog in shadow mode")
        routes = {r["id"]: r for r in staged["routes"]}
        for role in ROLES:
            route = routes[f"{role}-anima.work_item.changed"]
            self.assertEqual(route["when"], fa.anima_route_filters()[role])
            self.assertEqual(route["target"], {"kind": "akh", "endpoint": "core", "akh": role})
            self.assertIn(f"laya-{role}", routes, "legacy routes stay until the cutover")
        self.assertNotIn("when", routes["minerva-anima.plan.changed"])
        self.assertEqual(fa.render_anima_events(staged), staged, "idempotent")

    def test_live_render_is_the_cutover(self):
        self.render_live()
        staged = json.loads((self.root / "config/events.akh-target.json").read_text())
        self.assertFalse(staged["anima"][0]["shadow"])
        ids = {r["id"] for r in staged["routes"]}
        self.assertFalse({i for i in ids if i.startswith("laya-") or i.endswith((".janus.changed", ".minerva.changed"))})
        self.assertIn("vesta-monitoring.changed", ids)
        self.assertEqual(staged["watchdog"]["thread"], fa.STATUS_THREAD)
        self.assertEqual(staged["endpoints"]["board-watchdog"],
                         {"url": "http://127.0.0.1:18089", "token_env": "ROMAN_BOARD_WATCHDOG"})
        for role in ROLES:
            ws = self.toml(f"core-config/akh-medu/workspaces/{role}.toml")
            names = [s["name"] for s in ws["mcp"]["servers"]]
            self.assertNotIn("laya", names)
            self.assertNotIn("observer_status_file", ws["monitoring"])
            self.assertIn("(your servers: " + ", ".join(names) + ")", ws["system_prompt"])
            seeds = json.loads((self.root / f"config/reactor-seeds/{role}.json").read_text())
            removed = {r["id"] for r in seeds["remove"]}
            self.assertIn(f"roman-laya-{role}", removed)
            self.assertNotIn(f"roman-laya-{role}", {a["schedule"]["id"] for a in seeds["apply"]})
            body = apply_reactor_seeds.requests_for(seeds, enable=False, remove=True)[0]
            self.assertEqual(body["arguments"]["operation"], "remove")
        janus = json.loads((self.root / "config/reactor-seeds/janus.json").read_text())
        self.assertIn("roman-janus-anima.janus.changed", {r["id"] for r in janus["remove"]})
        # Rendering live again removes nothing more (the legacy routes are gone).
        live_events = self.root / "config/events.json"
        live_events.write_text((self.root / "config/events.akh-target.json").read_text())
        self.render_live()
        self.assertEqual(json.loads((self.root / "config/reactor-seeds/janus.json").read_text())["remove"], [])

    def test_shadow_workspaces_keep_laya_and_name_their_servers(self):
        for role in ROLES:
            ws = self.toml(f"core-config/akh-medu/workspaces/{role}.toml")
            names = [s["name"] for s in ws["mcp"]["servers"]]
            self.assertIn("laya", names)
            self.assertNotIn("{mcp_servers}", ws["system_prompt"])
            self.assertIn("(your servers: " + ", ".join(names) + ")", ws["system_prompt"])
            seeds = json.loads((self.root / f"config/reactor-seeds/{role}.json").read_text())
            self.assertEqual(seeds["remove"], [])

    def test_anima_seeds_carry_the_skill_scope_exit_condition_and_report_thread(self):
        prompts = json.loads((SOURCE / "skills/laya-intervention/prompts.json").read_text())
        for role in ROLES:
            seeds = json.loads((self.root / f"config/reactor-seeds/{role}.json").read_text())
            schedules = {a["schedule"]["rule"]["source"]: a["schedule"] for a in seeds["apply"]}
            schedule = schedules["roman-anima/anima.work_item.changed"]
            self.assertEqual(schedule["id"], f"roman-{role}-anima.work_item.changed")
            action = schedule["action"]
            self.assertIn(prompts["scope"][role], action["prompt"])
            self.assertIn("intervention_outcome", action["prompt"])
            self.assertNotIn("record_fixer_result", action["prompt"])
            self.assertEqual(action["exit_condition"], prompts["exit_condition"])
            self.assertEqual(action["report_thread"], fa.STATUS_THREAD)
            self.assertFalse(schedule["coalesce"])
            for other in set(ROLES) - {role}:
                self.assertNotIn(prompts["scope"][other], action["prompt"])
        minerva = json.loads((self.root / "config/reactor-seeds/minerva.json").read_text())
        self.assertIn("roman-anima/anima.plan.changed", {a["schedule"]["rule"]["source"] for a in minerva["apply"]})

    def test_skillpack_is_installed_into_each_engine_skills_dir(self):
        manifest = json.loads((SOURCE / "skills/laya-intervention/skill.json").read_text())
        self.assertEqual(manifest["id"], fa.LAYA_SKILL)
        for name in ("triples_file", "actions_file", "procedures_file"):
            self.assertTrue((SOURCE / "skills/laya-intervention" / manifest[name]).is_file(), name)
        for role in ROLES:
            dest = self.root / f"core-data/akh-medu/workspaces/{role}/kg/skills/laya-intervention"
            self.assertEqual((dest / "skill.json").read_bytes(), (SOURCE / "skills/laya-intervention/skill.json").read_bytes())
            self.assertEqual((dest / "prompts.json").stat().st_mode & 0o777, 0o600)
        actions = json.loads((SOURCE / "skills/laya-intervention/actions.json").read_text())
        self.assertTrue(all(set(a["effects"]) == {"initiates", "terminates"} for a in actions))

    def test_roman_memory_1_1_declares_the_intervention_predicates(self):
        pack = json.loads((SOURCE / "schemas/roman-memory.json").read_text())
        self.assertEqual(pack["pack"]["version"], "1.1.0")
        relations = {r["name"]: r for r in pack["relations"]}
        for name in ("intervention_outcome", "intervention_next_event"):
            self.assertEqual((relations[name]["cardinality"], relations[name]["temporal"]), ("single_valued", True))
        # The 1.0.0 relations are unchanged.
        self.assertTrue({"next_check", "status", "state"} <= set(relations))
        # The skill's vocabulary is exactly the one the prompt tells the akh to write.
        prompt = json.loads((SOURCE / "skills/laya-intervention/prompts.json").read_text())["instruction"]
        triples = json.loads((SOURCE / "skills/laya-intervention/triples.json").read_text())
        for kind in ("intervention_outcome_value", "intervention_next_event_value"):
            for value in [t["subject"] for t in triples if t["object"] == kind]:
                self.assertIn(value, prompt)

    def test_events_credentials_cover_the_anima_source_and_watchdog(self):
        text = (SOURCE / "daemon_credentials.py").read_text()
        section = text[text.index("elif role == 'events':"):text.index("binary = 'akheventd'")]
        self.assertIn("ROMAN_ANIMA_TOKEN", section)
        self.assertIn("ROMAN_BOARD_WATCHDOG", section)
        self.assertIn("board-watchdog.json", section)

    def test_events_are_staged_then_retargeted_with_backup(self):
        live = json.loads((self.root / "config/events.json").read_text())
        self.assertIn("automation", live["routes"][0])
        staged = json.loads((self.root / "config/events.akh-target.json").read_text())
        self.assertEqual(staged["endpoints"]["core"]["url"], "http://" + fa.CORE_LISTEN)
        for route in staged["routes"]:
            self.assertNotIn("automation", route)
            self.assertEqual(route["target"]["kind"], "akh")
            self.assertEqual(route["target"]["endpoint"], "core")
        retarget_events.retarget(self.root)
        self.assertEqual(json.loads((self.root / "config/events.json").read_text()), staged)
        retarget_events.rollback(self.root)
        self.assertEqual(json.loads((self.root / "config/events.json").read_text()), live)

    def test_plist_execs_credentials_wrapper_and_release_names_main_sha(self):
        plist = plistlib.loads((self.root / "launchd/dev.roman-akhs.core.plist").read_bytes())
        self.assertEqual(plist["ProgramArguments"][1:], [str(self.root / "bin/daemon_credentials.py"), "core", str(self.root)])
        env = plist["EnvironmentVariables"]
        self.assertEqual(env["AKH_DEPLOYMENT"], "deployed")
        self.assertNotIn("ROMAN_OLLAMA_KEY", env)
        self.assertNotIn("AKH_AUTO_START", env)
        release = json.loads((self.root / "config/core-release.json").read_text())
        self.assertRegex(Path(release["binary"]).name, fa.BINARY_NAME)
        self.assertEqual(release["build_command"], "cargo build --release --locked --features server --bin akhomed")
        keys = (self.root / "secrets/acp-clients.keys").read_text().splitlines()
        self.assertEqual(len(keys), 4)
        t3 = json.loads((self.root / "config/t3-acp.json").read_text())
        self.assertEqual(t3["endpoint"], f"tcp://127.0.0.1:{fa.ACP_TCP_PORT}")


class DelegationTests(unittest.TestCase):
    def session(self, profiles):
        return {"listen": "127.0.0.1:19999",
                "akhs": {"janus": {"workspace_profiles": ["janus"] + profiles, "tools": []}},
                "workspaces": {name: {} for name in ["janus"] + profiles}}

    def test_projects_without_a_worker_profile_are_not_registered(self):
        root = Path("/deploy")
        rendered, warnings = fa.render_delegation(self.session(["janus-worker-anima"]), root,
                                                  ["anima", "Secrets Share"])
        self.assertEqual(list(rendered["native"]["profiles"]), ["anima"])
        self.assertEqual(rendered["native"]["url"], "http://127.0.0.1:19999")
        self.assertEqual(len(warnings), 1)
        self.assertIn("janus-worker-secrets-share", warnings[0])
        rendered, warnings = fa.render_delegation(self.session([]), root, ["anima"])
        self.assertFalse(rendered["native"]["enabled"])
        self.assertFalse(rendered["native"]["escalation"]["enabled"])
        self.assertIn("stay disabled", warnings[-1])

    def test_broker_tool_names_map_to_native_tools(self):
        session = {"akhs": {"janus": {"tools": [
            mcp_tool(name, "janus-escalation", name) for name in fa.NATIVE_ESCALATION_TOOLS]}}}
        renames = fa.tool_renames(session, "janus")
        self.assertEqual(renames["mcp__janus-escalation__escalate_to_astra"], "escalate")
        self.assertEqual(renames["dispatch_mergiraf"], "dispatch_merge_repair")
        self.assertEqual(renames["mergiraf_status"], "escalation_status")
        self.assertEqual(fa.rename_tools("use mcp__janus-escalation__astra_escalation_status or escalate_to_astra",
                                         renames), "use escalation_status or escalate")
        session["akhs"]["janus"]["tools"].append(mcp_tool("new_tool", "janus-escalation", "new_tool"))
        with self.assertRaises(fa.RenderError):
            fa.tool_renames(session, "janus")

    def cluster(self, **overrides):
        settings = {"sessions_url": "https://anima.wegmueller.it/akh", "token_file": "/secrets/x.key",
                    "runtime_profile": "astra-impl-4", "projects": {"anima": "janus-worker-anima"}}
        settings.update(overrides)
        return settings

    def test_cluster_switch_registers_only_its_own_projects_and_warns_on_the_rest(self):
        root = Path("/deploy")
        rendered, warnings = fa.render_delegation({}, root, ["anima", "reddwarf"], cluster=self.cluster())
        self.assertEqual(rendered["native"]["url"], "https://anima.wegmueller.it/akh")
        self.assertEqual(rendered["native"]["token_file"], "/secrets/x.key")
        self.assertEqual(dict(rendered["native"]["profiles"]["anima"]),
                         {"workspace_profile": "janus-worker-anima", "runtime_profile": "astra-impl-4"})
        self.assertEqual(list(rendered["native"]["profiles"]), ["anima"])
        self.assertTrue(rendered["native"]["enabled"])
        self.assertTrue(rendered["native"]["escalation"]["enabled"])
        self.assertFalse(rendered["enabled"], "the generic delegate tool stays off")
        self.assertEqual(len(warnings), 1)
        self.assertIn("reddwarf", warnings[0])

    def test_cluster_switch_renders_a_declared_repository(self):
        # 53g PR bootstrap fix: the {workspace_profile, repository} table
        # form, read directly (as `render_delegation` sees it whether it
        # came from `janus_delegation_switch` or a caller's raw dict).
        root = Path("/deploy")
        rendered, warnings = fa.render_delegation({}, root, ["anima"], cluster=self.cluster(
            projects={"anima": {"workspace_profile": "janus-worker-anima",
                                "repository": "https://code.aopc.cloud/toasterson/Anima.git"}}))
        self.assertEqual(dict(rendered["native"]["profiles"]["anima"]),
                         {"workspace_profile": "janus-worker-anima", "runtime_profile": "astra-impl-4",
                          "repository": "https://code.aopc.cloud/toasterson/Anima.git"})
        self.assertEqual(warnings, [])

    def test_cluster_switch_ignores_the_local_sessions_json(self):
        root = Path("/deploy")
        # No local akhsessd grants at all: the cluster side is authoritative,
        # not the Mac's sessions.json, once the switch file is present.
        rendered, warnings = fa.render_delegation({}, root, ["anima"], cluster=self.cluster())
        self.assertEqual(list(rendered["native"]["profiles"]), ["anima"])
        self.assertEqual(warnings, [])

    def test_cluster_switch_warns_on_projects_it_registers_that_are_not_tracked_locally(self):
        root = Path("/deploy")
        cluster = self.cluster(projects={"anima": "janus-worker-anima", "stray": "janus-worker-stray"})
        _, warnings = fa.render_delegation({}, root, ["anima"], cluster=cluster)
        self.assertTrue(any("stray" in w and "not in config/janus-escalation.json" in w for w in warnings), warnings)

    def test_cluster_switch_disables_when_no_project_matches(self):
        root = Path("/deploy")
        rendered, warnings = fa.render_delegation({}, root, ["reddwarf"], cluster=self.cluster())
        self.assertFalse(rendered["native"]["enabled"])
        self.assertFalse(rendered["native"]["escalation"]["enabled"])
        self.assertIn("stay disabled", warnings[-1])


@NEEDS_TOMLLIB
class VestaSandboxTests(unittest.TestCase):
    """53g PR 8: Vesta's side effects move to akhsbxd/akhexecd sandboxes."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fixture(self.root)
        self.prov = fa.Provenance(commit="c" * 40, short="c" * 12, origin_main="c" * 40, committed_at=0)
        self.binary = self.root / "bin" / self.prov.binary_name
        self.binary.write_text("#!/bin/sh\n")
        self.token = self.root / "secrets/vesta-sandbox.key"

    def tearDown(self):
        self.temp.cleanup()

    def cutover(self, **settings):
        self.token.write_text("t" * 48 + "\n")
        (self.root / fa.VESTA_SANDBOX_FILE).write_text(json.dumps({"token_file": str(self.token), **settings}))

    def render(self):
        return configure_core.render(self.root, self.prov, self.binary, "0" * 64, source=SOURCE)

    def workspace(self, role):
        return tomllib.loads((self.root / f"core-config/akh-medu/workspaces/{role}.toml").read_text())

    def vesta_ops_tools(self):
        servers = {s["name"]: s for s in self.workspace("vesta")["mcp"]["servers"]}
        return servers["vesta-ops"]["tools"]

    def test_before_cutover_vesta_keeps_the_broker_and_no_exec(self):
        self.render()
        vesta = self.workspace("vesta")
        self.assertEqual(vesta["exec"], {"kind": "none"})
        self.assertEqual(self.vesta_ops_tools(),
                         ["gitops_exec", "gitops_publish", "homelab_inspect", "homelab_write"])
        self.assertIn((SOURCE / "roles/addenda/vesta.md").read_text(), vesta["system_prompt"])

    def test_cutover_renders_execd_with_both_sandbox_profiles(self):
        self.cutover()
        self.render()
        exec_table = self.workspace("vesta")["exec"]
        self.assertEqual(exec_table, {
            "kind": "execd", "sandbox_url": "http://vesta-sandbox.lab.home",
            "sandbox_token_file": str(self.token), "profile": "vesta-gitops", "principal": "vesta",
            "allow_insecure_http": True,
            "sandboxes": [{"name": "ops", "profile": "vesta-ops",
                           "description": fa.VESTA_OPS_SANDBOX["description"]}]})
        text = (self.root / "core-config/akh-medu/workspaces/vesta.toml").read_text()
        exec_start = text.index("\n[exec]\n")
        self.assertIn("\n[[exec.sandboxes]]\n", text[exec_start:])
        # Scalars of [exec] precede its array of tables (the TOML ordering gotcha).
        self.assertLess(text.index('principal = "vesta"', exec_start),
                        text.index("\n[[exec.sandboxes]]\n", exec_start))
        for role in ("janus", "minerva"):
            self.assertEqual(self.workspace(role)["exec"], {"kind": "none"})

    def test_cutover_removes_the_broker_write_tools_only(self):
        self.cutover()
        self.render()
        self.assertEqual(self.vesta_ops_tools(), ["homelab_inspect"])
        for tool in fa.VESTA_SANDBOXED_TOOLS:
            self.assertNotIn(tool, self.vesta_ops_tools())
        # vesta_ops.py itself stays until the cutover is verified.
        self.assertTrue((SOURCE / "vesta_ops.py").exists())

    def test_cutover_prompt_is_the_skillpack_not_the_broker_addendum(self):
        self.cutover()
        self.render()
        prompt = self.workspace("vesta")["system_prompt"]
        procedures = (SOURCE / "skills/vesta-remediation/procedures.md").read_text()
        self.assertIn(procedures, prompt)
        self.assertNotIn((SOURCE / "roles/addenda/vesta.md").read_text(), prompt)
        for tool in fa.VESTA_SANDBOXED_TOOLS:
            self.assertNotIn(tool, prompt)
        self.assertIn("sandbox_exec", prompt)
        # Janus and Minerva prompts are unchanged by Vesta's cutover.
        self.assertNotIn(procedures, self.workspace("janus")["system_prompt"])

    def test_skillpack_is_present_and_installed(self):
        manifest = json.loads((SOURCE / "skills/vesta-remediation/skill.json").read_text())
        self.assertEqual(manifest["id"], "vesta-remediation")
        self.assertEqual(manifest["procedures_file"], "procedures.md")
        self.assertTrue(fa.skill_procedures(SOURCE / "skills/vesta-remediation").strip())
        self.render()
        self.assertTrue((self.root / "skills/vesta-remediation/procedures.md").exists())

    def test_explicit_settings_override_the_defaults(self):
        self.cutover(sandbox_url="https://sandbox.example", allow_insecure_http=False)
        self.render()
        exec_table = self.workspace("vesta")["exec"]
        self.assertEqual(exec_table["sandbox_url"], "https://sandbox.example")
        self.assertNotIn("allow_insecure_http", exec_table)

    def test_bad_cutover_files_fail_closed(self):
        cases = [({"token_file": "relative/key"}, "absolute path"),
                 ({"token_file": str(self.root / "secrets/missing.key")}, "does not exist"),
                 ({"token_file": "/x", "extra": 1}, "unknown keys"),
                 ({"token_file": str(self.root / "secrets/core.key"), "sandbox_url": "ftp://x"}, "http"),
                 ({"token_file": str(self.root / "secrets/core.key"), "allow_insecure_http": "yes"}, "boolean")]
        for settings, message in cases:
            (self.root / fa.VESTA_SANDBOX_FILE).write_text(json.dumps(settings))
            with self.subTest(settings=settings), self.assertRaisesRegex(fa.RenderError, message):
                self.render()
        (self.root / fa.VESTA_SANDBOX_FILE).write_text("[not json")
        with self.assertRaisesRegex(fa.RenderError, "not valid JSON"):
            self.render()

    def test_skill_procedures_refuse_escaping_paths(self):
        skill = self.root / "skill"
        skill.mkdir()
        (skill / "skill.json").write_text(json.dumps({"id": "x", "procedures_file": "../secrets/core.key"}))
        with self.assertRaisesRegex(fa.RenderError, "inside the skill directory"):
            fa.skill_procedures(skill)
        (skill / "skill.json").write_text(json.dumps({"id": "x"}))
        (skill / "procedures.md").write_text("  \n")
        with self.assertRaisesRegex(fa.RenderError, "empty"):
            fa.skill_procedures(skill)


class JanusDelegationSwitchTests(unittest.TestCase):
    """53g PR 7 cluster variant: config/janus-delegation.json (operator decision
    2026-09-27 option a) switches Janus's workers onto the cluster akhsessd."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "secrets").mkdir()
        (self.root / "config").mkdir()
        self.token = self.root / "secrets/sessions-janus-delegation.key"

    def tearDown(self):
        self.temp.cleanup()

    def write(self, settings):
        (self.root / fa.JANUS_DELEGATION_FILE).write_text(json.dumps(settings))

    def valid(self, **overrides):
        settings = {"sessions_url": "https://anima.wegmueller.it/akh", "token_file": str(self.token),
                    "runtime_profile": "astra-impl-4",
                    "projects": {"akh-medu": "janus-worker-akh-medu", "anima": "janus-worker-anima"}}
        settings.update(overrides)
        return settings

    def test_absent_file_is_none(self):
        self.assertIsNone(fa.janus_delegation_switch(self.root))

    def test_valid_switch_is_parsed(self):
        self.token.write_text("t" * 40 + "\n")
        self.write(self.valid())
        cluster = fa.janus_delegation_switch(self.root)
        self.assertEqual(cluster, {
            "sessions_url": "https://anima.wegmueller.it/akh", "token_file": str(self.token),
            "runtime_profile": "astra-impl-4",
            "projects": {"akh-medu": "janus-worker-akh-medu", "anima": "janus-worker-anima"}})

    def test_bad_switch_files_fail_closed(self):
        self.token.write_text("t" * 40 + "\n")
        cases = [
            ({**self.valid(), "sessions_url": "ftp://x"}, "http"),
            ({k: v for k, v in self.valid().items() if k != "sessions_url"}, "sessions_url"),
            ({**self.valid(), "token_file": "relative/key"}, "absolute path"),
            ({**self.valid(), "token_file": str(self.root / "secrets/missing.key")}, "does not exist"),
            ({**self.valid(), "extra": 1}, "unknown keys"),
            ({**self.valid(), "runtime_profile": ""}, "runtime_profile"),
            ({**self.valid(), "runtime_profile": 7}, "runtime_profile"),
            ({k: v for k, v in self.valid().items() if k != "runtime_profile"}, "runtime_profile"),
            ({**self.valid(), "projects": {}}, "projects"),
            ({**self.valid(), "projects": {"x": ""}}, "projects"),
            ({**self.valid(), "projects": {"": "y"}}, "projects"),
            ({**self.valid(), "projects": "not-a-dict"}, "projects"),
            # 53g PR bootstrap fix: the {workspace_profile, repository} table form.
            ({**self.valid(), "projects": {"akh-medu": {}}}, "workspace_profile"),
            ({**self.valid(), "projects": {"akh-medu": {"workspace_profile": ""}}}, "workspace_profile"),
            ({**self.valid(), "projects": {"akh-medu": {"repository": "https://x/y.git"}}}, "workspace_profile"),
            ({**self.valid(), "projects": {"akh-medu": {"workspace_profile": "p", "repository": ""}}}, "repository"),
            ({**self.valid(), "projects": {"akh-medu": {"workspace_profile": "p", "repository": 5}}}, "repository"),
            ({**self.valid(), "projects": {"akh-medu": {"workspace_profile": "p", "extra": 1}}}, "unknown keys"),
            ({**self.valid(), "projects": {"akh-medu": 7}}, "projects"),
        ]
        for settings, message in cases:
            self.write(settings)
            with self.subTest(settings=settings), self.assertRaisesRegex(fa.RenderError, message):
                fa.janus_delegation_switch(self.root)
        (self.root / fa.JANUS_DELEGATION_FILE).write_text("[not json")
        with self.assertRaisesRegex(fa.RenderError, "not valid JSON"):
            fa.janus_delegation_switch(self.root)

    def test_non_object_fails_closed(self):
        (self.root / fa.JANUS_DELEGATION_FILE).write_text("[]")
        with self.assertRaisesRegex(fa.RenderError, "JSON object"):
            fa.janus_delegation_switch(self.root)

    def test_project_object_form_with_repository_is_parsed(self):
        # 53g PR bootstrap fix: a `projects` entry may be a
        # {workspace_profile, repository} table instead of the plain string,
        # and the two forms may be mixed in the same file.
        self.token.write_text("t" * 40 + "\n")
        self.write(self.valid(projects={
            "akh-medu": {"workspace_profile": "janus-worker-akh-medu",
                        "repository": "https://code.aopc.cloud/toasterson/akh-medu.git"},
            "anima": "janus-worker-anima",
        }))
        cluster = fa.janus_delegation_switch(self.root)
        self.assertEqual(cluster["projects"], {
            "akh-medu": {"workspace_profile": "janus-worker-akh-medu",
                        "repository": "https://code.aopc.cloud/toasterson/akh-medu.git"},
            "anima": "janus-worker-anima",
        })

    def test_project_object_form_without_repository_is_parsed(self):
        self.token.write_text("t" * 40 + "\n")
        self.write(self.valid(projects={"akh-medu": {"workspace_profile": "janus-worker-akh-medu"}}))
        cluster = fa.janus_delegation_switch(self.root)
        self.assertEqual(cluster["projects"], {"akh-medu": {"workspace_profile": "janus-worker-akh-medu"}})


@NEEDS_TOMLLIB
class JanusDelegationClusterRenderTests(unittest.TestCase):
    """The full render() with config/janus-delegation.json present, absent and malformed."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fixture(self.root)
        self.prov = fa.Provenance(commit="c" * 40, short="c" * 12, origin_main="c" * 40, committed_at=0)
        self.binary = self.root / "bin" / self.prov.binary_name
        self.binary.write_text("#!/bin/sh\n")
        self.token = self.root / "secrets/cluster-janus-delegation.key"
        self.token.write_text("c" * 48 + "\n")

    def tearDown(self):
        self.temp.cleanup()

    def switch(self, **overrides):
        settings = {"sessions_url": "https://anima.wegmueller.it/akh", "token_file": str(self.token),
                    "runtime_profile": "astra-impl-4",
                    "projects": {"akh-medu": "janus-worker-akh-medu", "Droneworld": "janus-worker-droneworld"}}
        settings.update(overrides)
        (self.root / fa.JANUS_DELEGATION_FILE).write_text(json.dumps(settings))

    def render(self):
        return configure_core.render(self.root, self.prov, self.binary, "0" * 64, source=SOURCE)

    def workspace(self, role):
        return tomllib.loads((self.root / f"core-config/akh-medu/workspaces/{role}.toml").read_text())

    def test_switch_renders_the_cluster_akhsessd(self):
        self.switch()
        self.render()
        native = self.workspace("janus")["delegation"]["native"]
        self.assertEqual(native["url"], "https://anima.wegmueller.it/akh")
        self.assertEqual(native["token_file"], str(self.token))
        self.assertEqual(native["profiles"]["akh-medu"],
                         {"workspace_profile": "janus-worker-akh-medu", "runtime_profile": "astra-impl-4"})
        self.assertEqual(native["profiles"]["Droneworld"],
                         {"workspace_profile": "janus-worker-droneworld", "runtime_profile": "astra-impl-4"})
        self.assertTrue(native["enabled"])
        self.assertTrue(native["escalation"]["enabled"])
        self.assertEqual(native["escalation"]["escalation_skill"], str(self.root / "skills/astra-escalation"))
        # The cluster token is the operator's own, already in place; render()
        # never generates a local one to sit alongside it.
        self.assertFalse((self.root / fa.DELEGATION_TOKEN).exists())

    def test_without_the_switch_janus_keeps_the_local_akhsessd(self):
        self.render()
        native = self.workspace("janus")["delegation"]["native"]
        self.assertEqual(native["url"], "http://127.0.0.1:18431")
        self.assertTrue((self.root / fa.DELEGATION_TOKEN).exists())

    def test_switch_renders_the_repository_bootstrap_field(self):
        # 53g PR bootstrap fix: a {workspace_profile, repository} table entry
        # renders `repository` into the project's profile; a plain-string
        # entry (no bootstrap) still renders with no `repository` key.
        self.switch(projects={
            "akh-medu": {"workspace_profile": "janus-worker-akh-medu",
                        "repository": "https://code.aopc.cloud/toasterson/akh-medu.git"},
            "Droneworld": "janus-worker-droneworld",
        })
        self.render()
        native = self.workspace("janus")["delegation"]["native"]
        self.assertEqual(native["profiles"]["akh-medu"],
                         {"workspace_profile": "janus-worker-akh-medu", "runtime_profile": "astra-impl-4",
                          "repository": "https://code.aopc.cloud/toasterson/akh-medu.git"})
        self.assertEqual(native["profiles"]["Droneworld"],
                         {"workspace_profile": "janus-worker-droneworld", "runtime_profile": "astra-impl-4"})
        self.assertNotIn("repository", native["profiles"]["Droneworld"])

    def test_malformed_switch_fails_the_whole_render_closed(self):
        # The switch is read before the per-role loop, so a malformed file
        # fails before Janus's workspace TOML is (re)written: the fixture's
        # pre-existing copy is left exactly as it was, never half-switched.
        before = (self.root / "core-config/akh-medu/workspaces/janus.toml").read_text()
        (self.root / fa.JANUS_DELEGATION_FILE).write_text("not json")
        with self.assertRaisesRegex(fa.RenderError, "not valid JSON"):
            self.render()
        self.assertEqual((self.root / "core-config/akh-medu/workspaces/janus.toml").read_text(), before)


class WakePromptTests(unittest.TestCase):
    def session(self, role, instructions):
        return {"endpoints": {}, "akhs": {role: {"tools": []}},
                "automations": {role: {"instructions": instructions}}}

    def test_role_policies_are_cut(self):
        text = "On duty.\n# Janus merge authority\nmerge\n# Astra escalation contract\nescalate"
        self.assertEqual(fa.wake_prompt(self.session("janus", text), "janus"), "On duty.")
        self.assertEqual(fa.wake_prompt(self.session("minerva", "Research."), "minerva"), "Research.")

    def test_a_missing_policy_marker_fails_closed(self):
        # The Astra heading was reworded: cutting would leave the policy in.
        text = "On duty.\n# Janus merge authority\nmerge\n# Astra escalation (v2)\nescalate"
        with self.assertRaises(fa.RenderError) as raised:
            fa.wake_prompt(self.session("janus", text), "janus")
        self.assertIn("# Astra escalation contract", str(raised.exception))
        with self.assertRaises(fa.RenderError):
            fa.wake_prompt(self.session("vesta", "On duty. Vesta can write the cluster."), "vesta")
        with self.assertRaises(fa.RenderError):
            fa.wake_prompt(self.session("ceres", "On duty."), "ceres")


class PrivateWriteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.umask = os.umask(0o022)

    def tearDown(self):
        os.umask(self.umask)
        self.temp.cleanup()

    def test_secret_is_never_wider_than_0600(self):
        path = self.root / "secrets/acp-clients.keys"
        seen = []
        real_replace = os.replace

        def replace(src, dst):
            seen.append(Path(src).stat().st_mode & 0o777)
            real_replace(src, dst)

        with unittest.mock.patch.object(fa.os, "replace", replace):
            fa.write_atomic(path, "token\n")
        self.assertEqual(seen, [0o600])
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(path.read_text(), "token\n")
        fa.write_atomic(self.root / "plist", b"<plist/>", mode=0o644)
        self.assertEqual((self.root / "plist").stat().st_mode & 0o777, 0o644)

    def test_a_failed_write_keeps_the_old_file_and_no_temp(self):
        path = self.root / "config/events.json"
        fa.write_atomic(path, '{"old": true}\n')
        with unittest.mock.patch.object(fa.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                fa.write_atomic(path, '{"new": true}\n')
        self.assertEqual(path.read_text(), '{"old": true}\n')
        self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["events.json"])

    @NEEDS_TOMLLIB
    def test_render_keeps_secrets_private(self):
        fixture(self.root)
        (self.root / "secrets").chmod(0o755)
        prov = fa.Provenance(commit="c" * 40, short="c" * 12, origin_main="c" * 40, committed_at=0)
        binary = self.root / "bin" / prov.binary_name
        binary.write_text("#!/bin/sh\n")
        configure_core.render(self.root, prov, binary, "0" * 64, source=SOURCE)
        self.assertEqual((self.root / "secrets").stat().st_mode & 0o777, 0o700)
        for name in ("acp-clients.keys", "core-events.key"):
            self.assertEqual((self.root / "secrets" / name).stat().st_mode & 0o777, 0o600, name)
        self.assertEqual((self.root / "config/core.json").stat().st_mode & 0o777, 0o600)


class ProvenanceTests(unittest.TestCase):
    def git(self, *args, cwd):
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                       env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                                GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_GLOBAL="/dev/null"))

    def test_refuses_branch_and_dirty_tree_accepts_main(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            self.git("init", "-q", "-b", "main", cwd=repo)
            (repo / "a").write_text("1")
            self.git("add", "a", cwd=repo)
            self.git("-c", "commit.gpgsign=false", "commit", "-qm", "one", cwd=repo)
            self.git("update-ref", "refs/remotes/origin/main", "HEAD", cwd=repo)
            prov = fa.main_provenance(repo, fetch=False)
            self.assertEqual(prov.binary_name, "akhomed-main-" + prov.short)
            self.assertEqual(len(prov.short), 12)
            (repo / "a").write_text("2")
            with self.assertRaisesRegex(fa.RenderError, "uncommitted"):
                fa.main_provenance(repo, fetch=False)
            self.git("-c", "commit.gpgsign=false", "commit", "-qam", "branch-only", cwd=repo)
            with self.assertRaisesRegex(fa.RenderError, "not an ancestor of origin/main"):
                fa.main_provenance(repo, fetch=False)


class CodesignConfigTests(unittest.TestCase):
    """`config/codesign.json` (operator decision 2026-09-27, stable-path signing)."""

    def test_absent_file_returns_none(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(configure_core.load_codesign_config(Path(temp)))

    def test_reads_identity_and_defaults_the_identifier(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config").mkdir()
            (root / "config/codesign.json").write_text(json.dumps({"identity": "Developer ID Application: X"}))
            self.assertEqual(
                configure_core.load_codesign_config(root),
                {"identity": "Developer ID Application: X", "identifier": configure_core.DEFAULT_CODESIGN_IDENTIFIER})

    def test_explicit_identifier_is_kept(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config").mkdir()
            (root / "config/codesign.json").write_text(json.dumps({"identity": "X", "identifier": "dev.other"}))
            self.assertEqual(configure_core.load_codesign_config(root)["identifier"], "dev.other")

    def test_missing_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config").mkdir()
            (root / "config/codesign.json").write_text(json.dumps({"identifier": "dev.other"}))
            with self.assertRaisesRegex(fa.RenderError, "identity"):
                configure_core.load_codesign_config(root)


class FakeCodesignRun:
    """Stands in for `subprocess.run(["codesign", ...])`: records every call
    (including the `codesign` argv[0]) and answers `-dr -` with a fixed
    designated requirement, without a real signing identity or filesystem
    signature."""

    def __init__(self):
        self.calls = []

    def __call__(self, args, capture_output=True, text=True):
        self.calls.append(args)

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        result = Result()
        if args[1:3] == ["-dr", "-"]:
            result.stdout = 'designated => identifier "dev.akh-medu.akhomed" and anchor apple generic\n'
        return result


class CoreBinaryInstallTests(unittest.TestCase):
    """`install_core_binary` (stable signed `bin/akhomed` + versioned copy)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "bin").mkdir(parents=True)
        self.built = self.root / "built-akhomed"
        self.built.write_bytes(b"#!/bin/sh\necho hi\n")
        self.prov = fa.Provenance(commit="d" * 40, short="d" * 12, origin_main="d" * 40, committed_at=0)

    def tearDown(self):
        self.temp.cleanup()

    def install(self, codesign_config):
        fake = FakeCodesignRun()
        warnings = []
        with unittest.mock.patch.object(configure_core.subprocess, "run", fake):
            info = configure_core.install_core_binary(self.built, self.root, self.prov, codesign_config, warnings)
        return info, warnings, fake.calls

    def test_stable_path_is_signed_atomically_and_matches_the_build(self):
        info, warnings, calls = self.install(
            {"identity": "Developer ID Application: X", "identifier": "dev.akh-medu.akhomed"})
        stable = self.root / "bin/akhomed"
        self.assertEqual(stable.read_bytes(), self.built.read_bytes())
        self.assertEqual([p.name for p in (self.root / "bin").iterdir() if p.name.startswith(".")], [])
        self.assertEqual(warnings, [])
        sign_call = next(c for c in calls if c[1] == "--force")
        self.assertEqual(sign_call[3], "Developer ID Application: X")
        self.assertEqual(sign_call[sign_call.index("--identifier") + 1], "dev.akh-medu.akhomed")
        self.assertNotIn("--options", sign_call)
        self.assertIn("--timestamp=none", sign_call)
        self.assertEqual(info["binary"], str(stable))
        self.assertEqual(info["codesign"], {
            "identity": "Developer ID Application: X", "identifier": "dev.akh-medu.akhomed",
            "designated_requirement": 'identifier "dev.akh-medu.akhomed" and anchor apple generic'})

    def test_versioned_copy_is_kept_unsigned_for_rollback_and_matches_build_sha256(self):
        info, _, calls = self.install({"identity": "X", "identifier": "dev.akh-medu.akhomed"})
        versioned = self.root / "bin" / self.prov.binary_name
        self.assertEqual(versioned.read_bytes(), self.built.read_bytes())
        self.assertEqual(info["build_sha256"], hashlib.sha256(self.built.read_bytes()).hexdigest())
        self.assertEqual(info["versioned_binary"], str(versioned))
        self.assertFalse(any(str(versioned) in c for c in calls))  # never handed to codesign

    def test_no_config_signs_ad_hoc_and_warns(self):
        info, warnings, calls = self.install(None)
        self.assertEqual(info["codesign"], {"identity": "-", "identifier": None, "designated_requirement": None})
        sign_call = next(c for c in calls if c[1] == "--force")
        self.assertEqual(sign_call[3], "-")
        self.assertTrue(any("ad-hoc" in w and "TCC" in w for w in warnings))

    def test_a_failed_signature_leaves_no_temp_file_and_the_stable_path_untouched(self):
        def failing(args, capture_output=True, text=True):
            class Result:
                returncode = 1
                stdout = ""
                stderr = "codesign failed"
            return Result()
        with unittest.mock.patch.object(configure_core.subprocess, "run", failing):
            with self.assertRaises(fa.RenderError):
                configure_core.install_core_binary(
                    self.built, self.root, self.prov, {"identity": "X", "identifier": "dev.akh-medu.akhomed"}, [])
        self.assertFalse((self.root / "bin/akhomed").exists())
        self.assertEqual(list((self.root / "bin").glob(".akhomed.*.tmp")), [])

    def test_rerunning_with_the_same_build_is_stable(self):
        self.install({"identity": "X", "identifier": "dev.akh-medu.akhomed"})
        info, warnings, _ = self.install({"identity": "X", "identifier": "dev.akh-medu.akhomed"})
        self.assertEqual(warnings, [])
        self.assertEqual(info["build_sha256"], hashlib.sha256(self.built.read_bytes()).hexdigest())


@NEEDS_TOMLLIB
class CoreReleaseFieldsTests(unittest.TestCase):
    """`render()` threads `install_core_binary`'s fields into core-release.json."""

    def test_release_records_build_sha256_versioned_binary_and_codesign(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            prov = fa.Provenance(commit="e" * 40, short="e" * 12, origin_main="e" * 40, committed_at=0)
            binary = root / "bin" / "akhomed"
            binary.write_text("#!/bin/sh\n")
            versioned = root / "bin" / prov.binary_name
            versioned.write_text("#!/bin/sh\n")
            configure_core.render(
                root, prov, binary, "s" * 64, source=SOURCE, build_sha256="b" * 64, versioned_binary=versioned,
                codesign={"identity": "X", "identifier": "dev.akh-medu.akhomed", "designated_requirement": "..."})
            release = json.loads((root / "config/core-release.json").read_text())
            self.assertEqual(release["binary"], str(binary))
            self.assertEqual(release["versioned_binary"], str(versioned))
            self.assertEqual(release["build_sha256"], "b" * 64)
            self.assertEqual(release["sha256"], "s" * 64)
            self.assertEqual(release["codesign"], {
                "identity": "X", "identifier": "dev.akh-medu.akhomed", "designated_requirement": "..."})
            evidence = json.loads((root / "evidence/core-binary.json").read_text())
            self.assertEqual(evidence, release)

    def test_omitted_fields_default_sanely(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            prov = fa.Provenance(commit="f" * 40, short="f" * 12, origin_main="f" * 40, committed_at=0)
            binary = root / "bin" / prov.binary_name
            binary.write_text("#!/bin/sh\n")
            configure_core.render(root, prov, binary, "s" * 64, source=SOURCE)
            release = json.loads((root / "config/core-release.json").read_text())
            self.assertIsNone(release["versioned_binary"])
            self.assertIsNone(release["codesign"])
            self.assertEqual(release["build_sha256"], "s" * 64)


# ── Lilith C2 (ADR 116) ─────────────────────────────────────────────────────

GOLDEN = SOURCE / "fixtures/golden"
LILITH_URL = "https://abc123xyz-8000.proxy.runpod.net/v1"
# WI 01a0e270-314e (ADR 116, C2): her own Board agent id, in the same
# `secrets/board-<role>.json` shape as the Romans'.
LILITH_AGENT_ID = "01a0abc4-90b1-7f43-8b13-2f7c9e6a1d44"


def lilith_fixture(root: Path, secrets=True, repository=True, **settings) -> None:
    """Configure Lilith on a fixture root: config/lilith.json, the endpoint
    URL file, and (optionally) her secret files and ancestry clone."""
    (root / fa.LILITH_BASE_URL_FILE).write_text(LILITH_URL + "\n")
    config = {"budget": {"minor_units": 6673, "currency": "USD"}}
    config.update(settings)
    (root / fa.LILITH_CONFIG_FILE).write_text(json.dumps(config))
    if secrets:
        (root / "secrets/lilith-openai-key").write_text("k" * 40 + "\n")
        (root / "secrets/anima-lilith-pat").write_text("p" * 40 + "\n")
        (root / "secrets/board-lilith.json").write_text(json.dumps({
            "bearer_token": "b" * 40,
            "agent": {"id": LILITH_AGENT_ID, "handle": "lilith", "role": "agent",
                      "created_at": "2026-09-28T00:00:00Z"}}))
    if repository:
        (root / fa.LILITH_REPOSITORY).mkdir(parents=True)


def golden(root: Path, rel: str) -> str:
    return (root / rel).read_text().replace(str(root), "{root}")


class RenderFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fixture(self.root)
        self.prov = fa.Provenance(commit="c" * 40, short="c" * 12, origin_main="c" * 40, committed_at=0)
        self.binary = self.root / "bin" / self.prov.binary_name
        self.binary.write_text("#!/bin/sh\n")

    def tearDown(self):
        self.temp.cleanup()

    def render(self, **kwargs):
        return configure_core.render(self.root, self.prov, self.binary, "0" * 64, source=SOURCE, **kwargs)

    def text(self, rel):
        return (self.root / rel).read_text()

    def toml(self, rel):
        return tomllib.loads(self.text(rel))


@NEEDS_TOMLLIB
class RomanGoldenTests(RenderFixture):
    """The data-driven role refactor keeps every Roman output byte-identical.

    fixtures/golden/ was rendered from this fixture by the pre-Lilith
    renderer (origin/main 5b4c9efc). An accidental change to a Roman's output
    fails here. After a deliberate one (a roles/*.md edit, a new server),
    regenerate with `ROMAN_GOLDEN_REGEN=1 python3 -m unittest
    test_full_akh.RomanGoldenTests` and review the fixture diff.
    """

    ROMAN_FILES = [f"core-config/akh-medu/workspaces/{r}.toml" for r in ROLES] + \
                  [f"config/reactor-seeds/{r}.json" for r in ROLES]

    def assert_romans_golden(self, host=False, lilith=False):
        files = self.ROMAN_FILES + (["core-config/akh-medu/config.toml"] if host else [])
        # lilith.toml is also the fixture the Rust lib test
        # `akh_config::roman_fixture_tests` loads through AkhConfig::load_checked
        # (include_str!), so a renderer change that drops or misplaces a key
        # fails either here or there.
        files += ["core-config/akh-medu/workspaces/lilith.toml"] if lilith else []
        for rel in files:
            name = {"config.toml": "host-config.toml"}.get(Path(rel).name, Path(rel).name)
            name = name.replace(".json", ".reactor-seeds.json")
            if os.environ.get("ROMAN_GOLDEN_REGEN") == "1":
                (GOLDEN / name).write_text(golden(self.root, rel))
            with self.subTest(rel=rel):
                self.assertEqual(golden(self.root, rel), (GOLDEN / name).read_text())

    def test_without_lilith_everything_is_byte_identical(self):
        self.render()
        self.assert_romans_golden(host=True)
        self.assertFalse((self.root / "core-config/akh-medu/workspaces/lilith.toml").exists())

    def test_with_lilith_the_romans_stay_byte_identical(self):
        lilith_fixture(self.root)
        self.render()
        self.assert_romans_golden(lilith=True)

    def test_the_rust_loader_check_reads_these_fixtures(self):
        # The other half of the chain: the lib test (run by CI's
        # `cargo test --lib`) loads fixtures/golden/lilith.toml through
        # AkhConfig::load_checked and fails on any dropped key.
        rust = (SOURCE.parents[1] / "src/akh_config_roman_fixture_tests.rs").read_text()
        for name in ("lilith.toml", "janus.toml", "minerva.toml", "vesta.toml"):
            self.assertIn(f'include_str!("../deploy/roman-mini/fixtures/golden/{name}")', rust)
        self.assertIn("#[path = \"akh_config_roman_fixture_tests.rs\"]",
                      (SOURCE.parents[1] / "src/akh_config.rs").read_text())


@NEEDS_TOMLLIB
class LilithRenderTests(RenderFixture):
    WS = "core-config/akh-medu/workspaces/lilith.toml"

    def test_workspace_runs_on_her_vllm_with_the_chief_of_staff_guard(self):
        lilith_fixture(self.root)
        self.render()
        ws = self.toml(self.WS)
        self.assertEqual((ws["name"], ws["backend"], ws["model"]), ("lilith", "openai", "lilith"))
        self.assertEqual(ws["openai_endpoint"], {"base_url": LILITH_URL, "api_key": "env:LILITH_OPENAI_KEY",
                                                 "context_window": 49152})
        self.assertNotIn("ollama_endpoint", ws)
        self.assertNotIn(LILITH_URL, (SOURCE / "full_akh.py").read_text(), "the URL is a render input")
        self.assertEqual(ws["exec"], {"kind": "none"})
        self.assertTrue(ws["autonomy"]["enabled"])
        self.assertTrue(ws["capture_transcripts"])
        self.assertNotIn("delegation", ws)
        self.assertEqual(ws["board"], {"url": fa.BOARD_URL, "agent_id": LILITH_AGENT_ID,
                                       "auth": {"token_env": "ROMAN_BOARD_LILITH"}})
        cos = ws["chief_of_staff"]
        # deny_unknown_fields on ChiefOfStaffConfig and Money.
        self.assertEqual(set(cos), {"profile", "repository", "announcements", "budget"})
        self.assertEqual(cos["budget"], {"minor_units": 6673, "currency": "USD"})
        self.assertEqual(cos["repository"], str(self.root / "lilith/ancestry/akh-medu.git"))
        profile = Path(cos["profile"])
        self.assertEqual(profile, self.root / "core-data/akh-medu/workspaces/lilith/kg/skills/"
                                              "chief-of-staff/permission-profile.json")
        source_pack = SOURCE.parents[1] / "skills/chief-of-staff"
        self.assertEqual(profile.read_bytes(), (source_pack / "permission-profile.json").read_bytes())
        self.assertEqual(json.loads(profile.read_text())["akh"], "lilith")
        for schema in (source_pack / "schemas").iterdir():
            installed = profile.parent / "schemas" / schema.name
            self.assertEqual(installed.read_bytes(), schema.read_bytes())
            self.assertEqual(installed.stat().st_mode & 0o777, 0o600)
        # The log is outside every root she can write, and its directory exists.
        announcements = Path(cos["announcements"])
        self.assertEqual(announcements, self.root / "state/lilith/cos-announcements.jsonl")
        for write_root in fa.lilith_write_roots(self.root) + [
                self.root / "core-data/akh-medu/workspaces/lilith/kg/scratch"]:
            self.assertFalse(fa._inside(announcements, write_root), write_root)
            self.assertFalse(str(announcements).startswith(str(write_root) + os.sep))
        self.assertTrue(announcements.parent.is_dir())
        self.assertEqual(announcements.parent.stat().st_mode & 0o777, 0o700)

    def test_her_own_mcp_identities(self):
        lilith_fixture(self.root)
        self.render()
        servers = {s["name"]: s for s in self.toml(self.WS)["mcp"]["servers"]}
        self.assertEqual(list(servers), ["anima", "djehuty", "seshat"])
        self.assertEqual(servers["anima"], {"name": "anima", "url": fa.ANIMA_MCP_URL,
                                            "auth": {"token_env": "LILITH_ANIMA_TOKEN"}})
        self.assertEqual(servers["djehuty"], {"name": "djehuty", "url": "http://127.0.0.1:18089/mcp",
                                              "auth": {"token_env": "ROMAN_BOARD_LILITH"}})
        self.assertEqual(servers["seshat"], {"name": "seshat", "url": "http://seshat.lab.home/mcp",
                                             "allow_insecure_http": True,
                                             "auth": {"token_env": "ROMAN_ARCHIVE_TOKEN"}})
        # Never a Roman's or Toasty's identity.
        text = self.text(self.WS)
        for foreign in ("ROMAN_ANIMA_TOKEN", "ROMAN_BOARD_JANUS", "ROMAN_BOARD_MINERVA", "ROMAN_BOARD_VESTA"):
            self.assertNotIn(foreign, text)
        # Her board tools land in the profile's T2 coordination class, not T3.
        tiers = json.loads((SOURCE.parents[1] / "skills/chief-of-staff/permission-profile.json").read_text())["tools"]
        self.assertEqual(tiers["mcp__djehuty__*"], "t2-coordination")

    def test_top_level_keys_precede_the_first_table(self):
        lilith_fixture(self.root)
        # A pre-existing hand-written file with a stale ollama table and an
        # unrelated key that must survive.
        (self.root / self.WS).write_text('name = "lilith"\nseed_packs = ["identity"]\n'
                                         '[ollama_endpoint]\nkind = "local"\nbase_url = "http://x"\n')
        self.render()
        text = self.text(self.WS)
        ws = tomllib.loads(text)
        self.assertEqual(ws["seed_packs"], ["identity"])
        self.assertNotIn("ollama_endpoint", ws)
        first_table = text.index("\n[")
        top_level = [k for k, v in ws.items() if not isinstance(v, dict)
                     and not (isinstance(v, list) and v and all(isinstance(i, dict) for i in v))]
        self.assertEqual(set(top_level), {"name", "model", "backend", "capture_transcripts", "system_prompt",
                                          "seed_packs"})
        for key in top_level:
            self.assertLess(text.index(f"\n{key} = "), first_table, key)
        self.assertLess(text.index("\n[openai_endpoint]"), text.index("\n[chief_of_staff]"))

    def test_prompt_is_her_role_and_the_skill_pack(self):
        lilith_fixture(self.root)
        self.render()
        prompt = self.toml(self.WS)["system_prompt"]
        role = (SOURCE / "roles/lilith.md").read_text().replace("{mcp_servers}", "anima, djehuty, seshat")
        self.assertTrue(prompt.startswith(role))
        self.assertIn((SOURCE.parents[1] / "skills/chief-of-staff/procedures.md").read_text(), prompt)
        self.assertNotIn("{mcp_servers}", prompt)
        self.assertNotIn("Roman-archetype", prompt, "the Roman common text is not hers")
        self.assertNotIn("No filesystem or shell tools are granted", prompt)
        for name in ("Maker", "Truthkeeper", "Seeker", "Challenger", "Guardian", "Sacred Clown",
                     "Steward", "Weaver", "the Center"):
            self.assertIn(name, prompt)

    def test_auto_started_only_with_every_input(self):
        lilith_fixture(self.root)
        warnings = self.render()
        host = self.toml("core-config/akh-medu/config.toml")
        self.assertEqual(host["daemon"]["auto_start"], list(ROLES) + ["lilith"])
        self.assertFalse([w for w in warnings if "lilith" in w], warnings)
        scoped = {p["id"]: p["akhs"] for p in host["acp"]["principals"]}
        self.assertEqual(scoped["operator"], list(ROLES) + ["lilith"])
        self.assertEqual(scoped["t3-janus"], ["janus"])
        core = {p["id"]: p for p in json.loads(self.text("config/core.json"))["principals"]}
        self.assertIn("lilith", core["operator"]["akhs"])
        self.assertIn("lilith", core["events"]["akhs"])
        self.assertNotIn("lilith", core["sessions"]["akhs"])
        # No reactor seeds yet (her wakes are a later slice).
        self.assertFalse((self.root / "config/reactor-seeds/lilith.json").exists())

    def test_missing_secrets_render_her_but_do_not_auto_start(self):
        for missing in list(fa.LILITH_SECRETS) + [fa.LILITH_REPOSITORY]:
            with self.subTest(missing=missing):
                lilith_fixture(self.root, repository=False)
                repository = self.root / fa.LILITH_REPOSITORY
                if missing == fa.LILITH_REPOSITORY:
                    if repository.exists():
                        repository.rmdir()
                else:
                    repository.mkdir(parents=True, exist_ok=True)
                    (self.root / missing).unlink()
                warnings = self.render()
                self.assertTrue((self.root / self.WS).exists())
                self.assertEqual(self.toml("core-config/akh-medu/config.toml")["daemon"]["auto_start"], list(ROLES))
                [warning] = [w for w in warnings if "lilith" in w]
                self.assertIn("NOT auto-started", warning)
                self.assertIn(str(self.root / missing), warning)

    def test_t3_chat_principal_key_and_provider_binary(self):
        lilith_fixture(self.root)
        self.assertFalse((self.root / fa.LILITH_T3_KEY_FILE).exists())
        self.render()
        host = self.toml("core-config/akh-medu/config.toml")
        principals = {p["id"]: p for p in host["acp"]["principals"]}
        self.assertEqual(principals["t3-lilith"], {"id": "t3-lilith", "token_file": str(self.root / fa.LILITH_T3_KEY_FILE),
                                                   "akhs": ["lilith"]})
        self.assertEqual(set(principals["t3-janus"]), set(principals["t3-lilith"]))
        key = self.root / fa.LILITH_T3_KEY_FILE
        self.assertEqual(key.stat().st_mode & 0o777, 0o600)
        secret = key.read_text().strip()
        self.assertGreaterEqual(len(secret), 48)
        # Her key opens the ACP listener, and is never overwritten by a re-render.
        self.assertIn(secret, self.text("secrets/acp-clients.keys").splitlines())
        self.render()
        self.assertEqual(key.read_text().strip(), secret)
        entrypoint = self.root / "bin/roman-lilith"
        self.assertEqual(entrypoint.read_text(), (SOURCE / "t3_entrypoint.py").read_text())
        self.assertEqual(entrypoint.stat().st_mode & 0o777, 0o700)

    def test_existing_t3_key_is_kept(self):
        lilith_fixture(self.root)
        (self.root / fa.LILITH_T3_KEY_FILE).write_text("operator-chosen-" + "x" * 40 + "\n")
        self.render()
        self.assertEqual((self.root / fa.LILITH_T3_KEY_FILE).read_text(), "operator-chosen-" + "x" * 40 + "\n")

    def test_t3_acp_json_lists_the_hosted_akhs_and_her_serve_checkout(self):
        lilith_fixture(self.root)
        self.render()
        t3 = json.loads(self.text("config/t3-acp.json"))
        self.assertEqual(t3["akhs"], list(ROLES) + ["lilith"])
        self.assertEqual(t3["lilith_serve"], fa.LILITH_SERVE_CHECKOUT)
        lilith_fixture(self.root, repository=False, serve_checkout="/srv/lilith-main")
        self.render()
        self.assertEqual(json.loads(self.text("config/t3-acp.json"))["lilith_serve"], "/srv/lilith-main")
        lilith_fixture(self.root, repository=False, serve_checkout="relative/dir")
        with self.assertRaises(fa.RenderError):
            self.render()

    def test_without_lilith_nothing_of_her_t3_surface_is_rendered(self):
        self.render()
        t3 = json.loads(self.text("config/t3-acp.json"))
        self.assertEqual(t3["akhs"], list(ROLES))
        self.assertNotIn("lilith_serve", t3)
        self.assertFalse((self.root / fa.LILITH_T3_KEY_FILE).exists())
        self.assertFalse((self.root / "bin/roman-lilith").exists())
        principals = self.toml("core-config/akh-medu/config.toml")["acp"]["principals"]
        self.assertNotIn("t3-lilith", [p["id"] for p in principals])

    def test_plan_creates_no_key_and_no_binary(self):
        lilith_fixture(self.root)
        configure_core.plan(self.root, self.prov, source=SOURCE)
        self.assertFalse((self.root / fa.LILITH_T3_KEY_FILE).exists())
        self.assertFalse((self.root / "bin/roman-lilith").exists())

    def test_staged_events_carry_her_board_identity(self):
        lilith_fixture(self.root)
        self.render()
        staged = json.loads(self.text("config/events.akh-target.json"))
        self.assertEqual(staged["endpoints"]["board-lilith"],
                         {"url": fa.BOARD_URL, "token_env": "ROMAN_BOARD_LILITH"})
        self.assertEqual(fa.board_identity(staged, "lilith")["auth"], {"token_env": "ROMAN_BOARD_LILITH"})
        self.assertFalse([r for r in staged["routes"] if fa.route_akh(r) == "lilith"])
        # An operator-written board-lilith endpoint is kept as it is.
        events = {"endpoints": {"board-lilith": {"url": "http://other", "token_file": "/t"}}}
        self.assertEqual(fa.with_board_endpoint(events, "lilith", "X")["endpoints"]["board-lilith"],
                         {"url": "http://other", "token_file": "/t"})

    def test_rerender_is_stable(self):
        lilith_fixture(self.root)
        self.render()
        before = {p: p.read_bytes() for p in (self.root / "core-config").rglob("*.toml")}
        self.render()
        self.assertEqual(before, {p: p.read_bytes() for p in (self.root / "core-config").rglob("*.toml")})

    def test_bad_inputs_fail_the_whole_render_closed(self):
        lilith_fixture(self.root)
        self.render()
        snapshot = {p: p.read_bytes() for p in (self.root / "core-config").rglob("*.toml")}
        cases = [
            ({"budget": {"minor_units": -1, "currency": "USD"}}, "budget"),
            ({"budget": {"minor_units": 5, "currency": "usd"}}, "budget"),
            ({"budget": {"minor_units": True, "currency": "USD"}}, "budget"),
            ({"budget": {"minor_units": 5}}, "budget"),
            ({}, "budget"),
            ({"budget": {"minor_units": 5, "currency": "USD"}, "extra": 1}, "unknown keys"),
            ({"budget": {"minor_units": 5, "currency": "USD"}, "context_window": 0}, "context_window"),
            ({"budget": {"minor_units": 5, "currency": "USD"}, "base_url_file": "rel/x"}, "absolute"),
            ({"budget": {"minor_units": 5, "currency": "USD"},
              "base_url_file": str(self.root / "secrets/nope")}, "does not exist"),
        ]
        for settings, message in cases:
            (self.root / fa.LILITH_CONFIG_FILE).write_text(json.dumps(settings))
            with self.subTest(settings=settings), self.assertRaisesRegex(fa.RenderError, message):
                self.render()
        (self.root / fa.LILITH_CONFIG_FILE).write_text(json.dumps({"budget": {"minor_units": 5, "currency": "USD"}}))
        for url in ("http://abc-8000.proxy.runpod.net/v1", "https://abc-8000.proxy.runpod.net",
                    "https://user:pw@abc.proxy.runpod.net/v1", "https://abc.proxy.runpod.net/v1?x=1", ""):
            (self.root / fa.LILITH_BASE_URL_FILE).write_text(url + "\n")
            with self.subTest(url=url), self.assertRaisesRegex(fa.RenderError, "base URL"):
                self.render()
        (self.root / fa.LILITH_CONFIG_FILE).write_text("[not json")
        with self.assertRaisesRegex(fa.RenderError, "not valid JSON"):
            self.render()
        self.assertEqual(snapshot, {p: p.read_bytes() for p in (self.root / "core-config").rglob("*.toml")})

    def test_announcements_inside_a_write_root_are_refused(self):
        lilith_fixture(self.root)
        inside = "core-data/akh-medu/workspaces/lilith/kg/scratch/cos.jsonl"
        with unittest.mock.patch.object(fa, "LILITH_ANNOUNCEMENTS", inside):
            with self.assertRaisesRegex(fa.RenderError, "inside her write root"):
                self.render()

    def test_credential_files_match_daemon_credentials(self):
        import daemon_credentials as dc
        mapped = {f"secrets/{name}": var for var, name in dc.LILITH_SECRET_FILES}
        mapped[f"secrets/{dc.LILITH_BOARD}"] = dc.LILITH_BOARD_ENV
        self.assertEqual(mapped, fa.LILITH_SECRETS)
        self.assertEqual(dc.LILITH_CONFIG, fa.LILITH_CONFIG_FILE)


@NEEDS_TOMLLIB
class LilithBoardAgentIdTests(RenderFixture):
    """`[board] agent_id` (daemon.rs ~2275: without it, board inbox ingest is
    disabled for the akh) is wired for Lilith only. The Romans' board events
    already arrive through akheventd routes, so turning on inbox ingest for
    them is a separate operator decision (full_akh.lilith_board_agent_id)."""

    WS = "core-config/akh-medu/workspaces/lilith.toml"

    def test_agent_id_is_rendered_when_present(self):
        lilith_fixture(self.root)
        self.render()
        ws = self.toml(self.WS)
        self.assertEqual(ws["board"]["agent_id"], LILITH_AGENT_ID)
        # bearer_token is read only by daemon_credentials.py at process start,
        # never by the renderer, and never lands in the rendered TOML.
        self.assertNotIn("b" * 40, self.text(self.WS))

    def test_missing_agent_id_warns_and_does_not_block_auto_start(self):
        lilith_fixture(self.root)
        (self.root / "secrets/board-lilith.json").write_text(json.dumps({"bearer_token": "b" * 40}))
        warnings = self.render()
        ws = self.toml(self.WS)
        self.assertNotIn("agent_id", ws["board"])
        [warning] = [w for w in warnings if "agent.id" in w]
        self.assertIn("board inbox ingest for lilith stays off", warning)
        self.assertIn("GET /v1/agents/me", warning)
        host = self.toml("core-config/akh-medu/config.toml")
        self.assertIn("lilith", host["daemon"]["auto_start"])

    def test_malformed_agent_id_fails_the_render_closed(self):
        lilith_fixture(self.root)
        (self.root / "secrets/board-lilith.json").write_text(
            json.dumps({"bearer_token": "b" * 40, "agent": {"id": "not-a-uuid"}}))
        with self.assertRaisesRegex(fa.RenderError, "not a UUID"):
            self.render()

    def test_romans_board_stays_unchanged(self):
        lilith_fixture(self.root)
        self.render()
        for role in ROLES:
            ws = self.toml(f"core-config/akh-medu/workspaces/{role}.toml")
            self.assertEqual(set(ws["board"]), {"url", "auth"})


@NEEDS_TOMLLIB
def split_evidence(root: Path, **overrides) -> Path:
    """A verified ADR 120 split evidence file for the fixture root, as written by
    `akhomed store split-per-akh`, plus the host database it records."""
    (root / "state/core").mkdir(parents=True, exist_ok=True)
    (root / fa.HOST_DATABASE).write_bytes(b"SQLite format 3\x00")
    evidence = {"schema": fa.SPLIT_EVIDENCE_SCHEMA, "verified": True, "layout": "per_akh",
                "source": {"path": str(root / fa.SHARED_DATABASE)},
                "host": {"path": str(root / fa.HOST_DATABASE), "tables": []},
                "akhs": [{"name": r, "path": str(root / f"state/core/akh.{r}.sqlite3"), "tables": []}
                         for r in ROLES]}
    evidence.update(overrides)
    path = root / "evidence/per-akh-split.json"
    path.write_text(json.dumps(evidence))
    return path


@NEEDS_TOMLLIB
class PerAkhDatabaseTests(RenderFixture):
    """ADR 120: config/per-akh-databases.json opts the core into per-akh SQLite
    files, gated on the split evidence; without it nothing changes."""

    HOST = "core-config/akh-medu/config.toml"

    def plist_env(self):
        plist = plistlib.loads((self.root / "launchd/dev.roman-akhs.core.plist").read_bytes())
        return plist["EnvironmentVariables"]

    def switch(self, evidence):
        (self.root / fa.PER_AKH_DATABASES_FILE).write_text(json.dumps({"evidence": str(evidence)}))

    def test_without_the_switch_the_shared_file_is_rendered(self):
        self.render()
        database = self.toml(self.HOST)["database"]
        self.assertEqual(database["url"], fa.sqlite_url(self.root / fa.SHARED_DATABASE))
        self.assertNotIn("layout", database)
        self.assertEqual(self.plist_env()["DATABASE_URL"], database["url"])

    def test_switch_renders_the_host_file_with_per_akh_layout(self):
        self.switch(split_evidence(self.root))
        warnings = self.render()
        database = self.toml(self.HOST)["database"]
        self.assertEqual(database["url"], fa.sqlite_url(self.root / fa.HOST_DATABASE))
        self.assertEqual(database["layout"], "per_akh")
        self.assertEqual(database["snapshot_interval_secs"], 30)
        # DATABASE_URL overrides [database] url in akhomed, so both must move.
        self.assertEqual(self.plist_env()["DATABASE_URL"], database["url"])
        self.assertIn("ADR 120 per-akh layout", self.text(self.HOST))
        self.assertTrue(any("per-akh databases" in w for w in warnings), warnings)

    def test_switch_accepts_an_evidence_path_relative_to_the_root(self):
        split_evidence(self.root)
        self.switch("evidence/per-akh-split.json")
        self.render()
        self.assertEqual(self.toml(self.HOST)["database"]["layout"], "per_akh")

    def test_switch_changes_only_the_database_table(self):
        self.render()
        shared = self.text(self.HOST)
        self.switch(split_evidence(self.root))
        self.render()
        per_akh = self.text(self.HOST)
        changed = [line for line in per_akh.splitlines() if line not in shared.splitlines()]
        self.assertTrue(all(line.startswith(("url = ", "layout = ", "#")) for line in changed), changed)
        for role in ROLES:
            self.assertEqual(golden(self.root, f"core-config/akh-medu/workspaces/{role}.toml"),
                             (GOLDEN / f"{role}.toml").read_text())

    def test_bad_switches_fail_closed_before_writing(self):
        good = split_evidence(self.root)
        cases = [
            ("not json", "not valid JSON"),
            (json.dumps(["x"]), "JSON object"),
            (json.dumps({"evidence": str(good), "extra": 1}), "unknown keys"),
            (json.dumps({}), "needs `evidence`"),
            (json.dumps({"evidence": str(self.root / "evidence/missing.json")}), "does not exist"),
        ]
        for text, message in cases:
            with self.subTest(message=message):
                (self.root / fa.PER_AKH_DATABASES_FILE).write_text(text)
                with self.assertRaisesRegex(fa.RenderError, re.escape(message)):
                    self.render()
                self.assertFalse((self.root / self.HOST).exists())
        evidence_cases = [
            ({"schema": "other"}, "is not a"),
            ({"verified": False}, "verified split"),
            ({"source": {"path": str(self.root / "state/core/other.sqlite3")}}, "not " + str(self.root)),
            ({"host": {"path": str(self.root / "state/core/elsewhere.sqlite3")}}, "wrote host"),
        ]
        for overrides, message in evidence_cases:
            with self.subTest(message=message):
                self.switch(split_evidence(self.root, **overrides))
                with self.assertRaisesRegex(fa.RenderError, re.escape(message)):
                    self.render()
        self.switch(split_evidence(self.root))
        (self.root / fa.HOST_DATABASE).unlink()
        with self.assertRaisesRegex(fa.RenderError, "is missing"):
            self.render()


class PlanTests(RenderFixture):
    def tree(self):
        return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}

    def test_plan_writes_nothing(self):
        before = self.tree()
        lines, _ = configure_core.plan(self.root, self.prov, source=SOURCE)
        self.assertEqual(before, self.tree())
        self.assertIn("~ core-config/akh-medu/workspaces/janus.toml: changes", lines)
        self.assertIn("+ core-config/akh-medu/workspaces/minerva.toml: new file", lines)

    def test_plan_after_render_is_unchanged_and_shows_a_dropped_hand_hold(self):
        lilith_fixture(self.root)
        self.render()
        lines, _ = configure_core.plan(self.root, self.prov, source=SOURCE)
        for role in ROLES + ("lilith",):
            self.assertIn(f"= core-config/akh-medu/workspaces/{role}.toml: unchanged", lines)
        self.assertIn("= core-config/akh-medu/config.toml: unchanged", lines)
        self.assertEqual(lines[-1], 'daemon.auto_start = ["janus", "minerva", "vesta", "lilith"]')
        # An operator's hand edit (e.g. the 2026-09-27 Janus broker hold) shows as removed.
        janus = self.root / "core-config/akh-medu/workspaces/janus.toml"
        janus.write_text(janus.read_text() + '\n[hold]\nreason = "operator hold"\n')
        before = self.tree()
        lines, _ = configure_core.plan(self.root, self.prov, source=SOURCE)
        self.assertEqual(before, self.tree())
        self.assertIn("~ core-config/akh-medu/workspaces/janus.toml: changes", lines)
        self.assertIn('-reason = "operator hold"', lines)
        self.assertIn("--- a/core-config/akh-medu/workspaces/janus.toml", lines)

    def test_plan_reuses_the_installed_release_provenance(self):
        self.render()
        prov = configure_core.release_provenance(self.root)
        self.assertEqual((prov.commit, prov.short), (self.prov.commit, self.prov.short))
        lines, _ = configure_core.plan(self.root, prov, source=SOURCE)
        self.assertIn("= core-config/akh-medu/config.toml: unchanged", lines)
        (self.root / "config/core-release.json").unlink()
        with self.assertRaisesRegex(fa.RenderError, "--repo"):
            configure_core.release_provenance(self.root)

    def test_plan_fails_closed_on_a_bad_switch_without_writing(self):
        (self.root / fa.LILITH_CONFIG_FILE).write_text("{}")
        before = self.tree()
        with self.assertRaisesRegex(fa.RenderError, "budget"):
            configure_core.plan(self.root, self.prov, source=SOURCE)
        self.assertEqual(before, self.tree())


@NEEDS_TOMLLIB
class TomlWriterTests(unittest.TestCase):
    def test_round_trips_escapes_and_inline_tables(self):
        doc = {"s": 'quote " back \\ tab\t del\x7f nl\n é', "n": 3, "b": False,
               "t": {"x": [1, 2], "auth": fa.Inline(token_env="V")},
               "arr": [{"name": "a", "tools": ["p", "q", "r", "s", "t"]}]}
        self.assertEqual(tomllib.loads(fa.dump_toml(doc)), doc)


if __name__ == "__main__":
    unittest.main()
