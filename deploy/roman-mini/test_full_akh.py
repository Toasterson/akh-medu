"""Rendering tests for the full-akh Roman cutover (Phase 53f, ADR 113).

Renders a synthetic deployment root into a temporary directory and checks the
TOML/JSON shapes akhomed and akheventd will load. No git remote, network or
real credential is used.
"""
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


@NEEDS_TOMLLIB
class TomlWriterTests(unittest.TestCase):
    def test_round_trips_escapes_and_inline_tables(self):
        doc = {"s": 'quote " back \\ tab\t del\x7f nl\n é', "n": 3, "b": False,
               "t": {"x": [1, 2], "auth": fa.Inline(token_env="V")},
               "arr": [{"name": "a", "tools": ["p", "q", "r", "s", "t"]}]}
        self.assertEqual(tomllib.loads(fa.dump_toml(doc)), doc)


if __name__ == "__main__":
    unittest.main()
