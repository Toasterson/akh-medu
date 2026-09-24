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
import tomllib
import unittest

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
            tools.append(mcp_tool("escalate_to_astra", "janus-escalation", "escalate_to_astra"))
        if role == "vesta":
            tools += [mcp_tool("homelab_inspect", "vesta-ops", "homelab_inspect"),
                      mcp_tool("grafana__query_prometheus", "vesta-ops", "grafana__query_prometheus")]
        akhs[role] = {"core_service": "core", "instructions": "", "tools": tools}
        wake = ("You are now on duty. Coordinate with board_send_inbox. " + json.dumps(IDS) +
                "\nReport on thread " + fa.STATUS_THREAD + ".")
        if role == "janus":
            wake += "\n# Janus merge authority\nUse anima__approve_and_merge."
        if role == "vesta":
            wake += "\nVesta has homelab_inspect for live Kubernetes."
        automations[role] = {"id": role, "instructions": wake}
    principals = [{"id": "runtime", "token_file": str(root / "secrets/core.key"), "roles": ["runtime"], "akhs": list(ROLES)}]
    principals += [{"id": f"t3-{r}", "token_file": str(root / f"secrets/t3-{r}.key"), "roles": ["client"], "akhs": [r]} for r in ROLES]
    (root / "config/sessions.json").write_text(json.dumps(
        {"endpoints": endpoints, "principals": principals, "akhs": akhs, "automations": automations}))
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
            expected = {"anima", "seshat", "sokar", "board", "observer", "laya"}
            expected |= {"janus": {"janus-escalation"}, "vesta": {"vesta-ops"}}.get(role, set())
            self.assertEqual(set(servers), expected)
            self.assertEqual(servers["board"]["auth"], {"token_env": f"ROMAN_BOARD_{role.upper()}"})
            self.assertEqual(servers["laya"]["auth"], {"token_file": str(self.root / f"secrets/laya-{role}.key")})
            self.assertEqual(servers["laya"]["tools"], ["record_fixer_result"])
            self.assertTrue(servers["seshat"]["allow_insecure_http"])
            self.assertNotIn("allow_insecure_http", servers["sokar"])
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
        self.assertIn("mcp__janus-escalation__escalate_to_astra", janus)
        self.assertIn("mcp__vesta-ops__homelab_inspect", prompts[2])

    def test_rename_is_single_pass(self):
        renames = {"board__board_agents": "mcp__board__board_agents", "board_agents": "mcp__board__board_agents"}
        self.assertEqual(fa.rename_tools("use board__board_agents or board_agents", renames),
                         "use mcp__board__board_agents or mcp__board__board_agents")

    def test_core_json_scopes_sessions_events_operator(self):
        core = json.loads((self.root / "config/core.json").read_text())
        by_id = {p["id"]: p for p in core["principals"]}
        self.assertEqual(by_id["sessions"]["tools"], {"kind": "legacy_cognitive"})
        self.assertEqual(by_id["sessions"]["grants"], "optional")
        self.assertEqual(by_id["events"]["roles"], ["event"])
        self.assertEqual(by_id["events"]["tools"], {"kind": "named", "tools": []})
        self.assertIn("reactor_manage", by_id["operator"]["tools"]["tools"])
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
                                          "roman-alerts/laya.vesta.intervention"})
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


class TomlWriterTests(unittest.TestCase):
    def test_round_trips_escapes_and_inline_tables(self):
        doc = {"s": 'quote " back \\ tab\t del\x7f nl\n é', "n": 3, "b": False,
               "t": {"x": [1, 2], "auth": fa.Inline(token_env="V")},
               "arr": [{"name": "a", "tools": ["p", "q", "r", "s", "t"]}]}
        self.assertEqual(tomllib.loads(fa.dump_toml(doc)), doc)


if __name__ == "__main__":
    unittest.main()
