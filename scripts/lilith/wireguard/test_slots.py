"""Tests for Variant C (talos k0s hub, 1Password peer slots).

A fake runner stands in for both `op` (the 1Password service-account
wrapper) and `git show` on the talos checkout, so nothing here touches a
vault, a repository or the network.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


render = _load("render")
slots = _load("slots")

OP = "/fake/op-service-account"

# Mirrors talos clusters/archibald/lilith-wg/allowlist.conf (enabled/disabled).
TALOS_ALLOWLIST = """\
# comment
lab-gateway-http          lan  tcp  192.168.1.12  80     on
akh-sessions-home         hub  tcp  10.0.1.2      18240  on
akh-sandbox-akhsbxd       hub  tcp  10.0.1.2      8242   off
openindiana-vm-executor   lan  tcp  192.168.1.114 10115  off
"""


class FakeWorld:
    """Fake `op` + `git` behind the subprocess.run signature."""

    def __init__(self, items: dict[str, dict[str, str]], allowlist: str = TALOS_ALLOWLIST):
        self.items = items
        self.files = {"clusters/archibald/lilith-wg/allowlist.conf": allowlist}
        self.calls: list[tuple[list[str], str | None]] = []
        self.fail_edit_on: str | None = None

    def __call__(self, argv, input=None, capture_output=True, text=True, timeout=None):  # noqa: A002
        argv = list(argv)
        self.calls.append((argv, input))
        ok = lambda out: subprocess.CompletedProcess(argv, 0, out, "")  # noqa: E731
        if argv[0] == "git":
            if argv[3] == "fetch":
                return ok("")
            _ref, path = argv[4].split(":", 1)
            if path not in self.files:
                return subprocess.CompletedProcess(argv, 128, "", "fatal: path does not exist")
            return ok(self.files[path])
        assert argv[0] == OP, argv
        if argv[1] == "read":
            _, _, _vault, item, field = argv[2].split("/")
            return ok(self.items[item][field] + "\n")
        if argv[1:3] == ["item", "get"]:
            item = self.items[argv[3]]
            return ok(json.dumps({"title": argv[3], "fields": [
                {"id": k, "label": k, "value": v} for k, v in item.items()
            ]}))
        if argv[1:3] == ["item", "edit"]:
            if self.fail_edit_on == argv[3]:
                return subprocess.CompletedProcess(argv, 1, "", "[ERROR] permission denied")
            doc = json.loads(input)
            for f in doc["fields"]:
                self.items[argv[3]][f["label"]] = f["value"]
            return ok("{}")
        raise AssertionError(f"unexpected command {argv}")


def _keypair():
    return render.generate_keypair()


T0 = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


class VariantCBase(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        tmp = Path(self.td.name)
        self.talos = tmp / "talos"
        text = (HERE / "wg-config.toml.example").read_text()
        text = text.replace("~/.config/akh-medu/lilith-wireguard/slots", str(tmp / "state"))
        text = text.replace('talos_repo = "~/ws/talos"', f'talos_repo = "{self.talos}"')
        text = text.replace("~/.local/bin/op-service-account", OP)
        cfg = tmp / "wg-config.toml"
        cfg.write_text(text)
        self.config = render.load_config(cfg)
        self.hub = _keypair()
        self.slot_keys = {n: _keypair() for n in range(1, 5)}
        items = {"lilith-wg-hub": {"private_key": self.hub.private, "public_key": self.hub.public}}
        state = {"notesPlain": "public slot state"}
        for n, kp in self.slot_keys.items():
            items[f"lilith-wg-slot-{n}"] = {"private_key": kp.private, "public_key": kp.public, "address": f"10.201.0.{10 + n}/32"}
            state |= {f"slot{n}_public_key": kp.public, f"slot{n}_enabled": "false", f"slot{n}_run": "-",
                      f"slot{n}_assigned_at": "-", f"slot{n}_rotated_at": "-"}
        items["lilith-wg-slots"] = state
        self.world = FakeWorld(items)
        self.op = slots.OnePassword(command=OP, vault="Homelab", runner=self.world)
        self.now = T0

    def tearDown(self):
        self.td.cleanup()

    def clock(self):
        return self.now

    @property
    def state(self):
        return self.world.items["lilith-wg-slots"]

    def assign(self, run, **kw):
        return slots.assign(self.config, self.op, run, runner=self.world, now=self.clock, **kw)

    def release(self, **kw):
        return slots.release(self.config, self.op, now=self.clock, **kw)

    def status(self):
        return {v.slot: v.state for v in slots.slot_status(self.config, self.op, now=self.clock)}


class ConfigTests(VariantCBase):
    def test_example_is_variant_c_with_two_tunnelled_services(self):
        self.assertTrue(self.config.is_variant_c())
        self.assertEqual([s.name for s in self.config.allow_services()], ["lab-gateway-http", "akh-sessions-home"])
        self.assertEqual(self.config.hub.public_endpoint, "95.216.66.53:51822")

    def test_slot_addresses(self):
        self.assertEqual(self.config.slots.address(1), "10.201.0.11/32")
        self.assertEqual(self.config.slots.address(4), "10.201.0.14/32")
        with self.assertRaises(ValueError):
            self.config.slots.address(5)

    def test_pod_routes_only_hub_and_allowlisted_hosts(self):
        self.assertEqual(render.pod_allowed_ips(self.config), "10.0.1.2/32, 10.201.0.1/32, 192.168.1.12/32")

    def test_hostname_service_must_be_an_ip_literal(self):
        text = self.config.source_path.read_text().replace('host = "10.0.1.2"\nport = 18240', 'host = "relay.example"\nport = 18240')
        self.config.source_path.write_text(text)
        with self.assertRaisesRegex(ValueError, "literal IPv4"):
            render.load_config(self.config.source_path)

    def test_state_dir_inside_repo_is_rejected(self):
        text = self.config.source_path.read_text()
        text = text.replace(str(Path(self.td.name) / "state"), str(HERE / "state-should-fail"))
        self.config.source_path.write_text(text)
        with self.assertRaises(ValueError):
            render.load_config(self.config.source_path)


class PodClientRenderTests(VariantCBase):
    def test_onetun_forwards_exactly_the_enabled_services(self):
        cmds = render.render_onetun_commands(self.config, self.hub.public, self.config.slots.address(2))
        joined = " ".join(cmds)
        self.assertIn("127.0.0.1:80:192.168.1.12:80", joined)
        self.assertIn("127.0.0.1:18240:10.0.1.2:18240", joined)
        self.assertNotIn("8242", joined)
        self.assertNotIn("10115", joined)
        self.assertIn("--endpoint-addr 95.216.66.53:51822", joined)
        self.assertIn("--source-peer-ip 10.201.0.12", joined)

    def test_hosts_line_maps_virtual_hosts_to_loopback(self):
        self.assertEqual(render.render_hosts(self.config), "127.0.0.1 seshat.lab.home board.lab.home\n")

    def test_cli_renders_slot_files_without_key_material(self):
        out = Path(self.td.name) / "out"
        rc = render.main(["--config", str(self.config.source_path), "--out-dir", str(out), "--slot", "3"])
        self.assertEqual(rc, 0)
        names = sorted(p.name for p in out.iterdir())
        self.assertEqual(names, [
            "slot-3.env.json", "slot-3.hosts", "slot-3.kernel.conf.template",
            "slot-3.onetun.sh", "slot-3.wireproxy.conf",
        ])
        for path in out.iterdir():
            for kp in [self.hub, *self.slot_keys.values()]:
                self.assertNotIn(kp.private, path.read_text())
        self.assertIn("PrivateKey = $LILITH_WG_PRIVATE_KEY", (out / "slot-3.kernel.conf.template").read_text())
        self.assertIn("Address = 10.201.0.13/32", (out / "slot-3.wireproxy.conf").read_text())

    def test_cli_requires_slot_for_variant_c(self):
        with self.assertRaises(SystemExit):
            render.main(["--config", str(self.config.source_path), "--out-dir", str(Path(self.td.name) / "o")])


class TalosAllowlistTests(VariantCBase):
    def test_allowlist_parity_matches_talos(self):
        self.assertEqual(render.allowlist_drift(self.config, render.parse_talos_allowlist(TALOS_ALLOWLIST)), [])

    def test_allowlist_drift_is_reported_both_ways(self):
        talos = TALOS_ALLOWLIST.replace("8242   off", "8242   on").replace("18240  on", "18240  off")
        drift = render.allowlist_drift(self.config, render.parse_talos_allowlist(talos))
        self.assertEqual(len(drift), 2)
        self.assertTrue(any("akh-sessions-home" in d and "does not allow" in d for d in drift))
        self.assertTrue(any("akh-sandbox-akhsbxd" in d and "no enabled service" in d for d in drift))


class StatusTests(VariantCBase):
    def test_all_free_initially(self):
        self.assertEqual(set(self.status().values()), {slots.SlotState.FREE})

    def test_assigned_rotating_free_and_unknown(self):
        self.assign("run-a")
        self.release(slot=2)
        self.state["slot3_enabled"] = "maybe"
        states = self.status()
        self.assertEqual(states[1], slots.SlotState.ASSIGNED)
        self.assertEqual(states[2], slots.SlotState.ROTATING)
        self.assertEqual(states[3], slots.SlotState.UNKNOWN)
        self.assertEqual(states[4], slots.SlotState.FREE)
        self.now = T0 + timedelta(seconds=self.config.slots.propagation_seconds + 1)
        self.assertEqual(self.status()[2], slots.SlotState.FREE)

    def test_items_disagreeing_reads_rotating(self):
        self.world.items["lilith-wg-slot-4"]["public_key"] = _keypair().public
        self.assertEqual(self.status()[4], slots.SlotState.ROTATING)


class AssignTests(VariantCBase):
    def test_assign_enables_slot_and_returns_bundle(self):
        bundle = self.assign("run-a", runpod_pod_id="pod123")
        self.assertEqual(bundle["slot"], 1)
        self.assertEqual(bundle["address"], "10.201.0.11/32")
        self.assertEqual(bundle["env"]["LILITH_WG_PRIVATE_KEY"], self.slot_keys[1].private)
        self.assertEqual(bundle["hub_public_key"], self.hub.public)
        self.assertEqual(bundle["env"]["LILITH_WG_HOSTS"], "127.0.0.1 seshat.lab.home board.lab.home")
        self.assertEqual(bundle["env"]["LILITH_WG_HEALTHCHECK_RETRIES"], "36")
        self.assertIn("AllowedIPs = 10.0.1.2/32, 10.201.0.1/32, 192.168.1.12/32", bundle["kernel_wg_conf"])
        self.assertEqual(self.state["slot1_enabled"], "true")
        self.assertEqual(self.state["slot1_run"], "run-a")
        self.assertEqual(self.assign("run-b")["slot"], 2)

    def test_no_key_material_on_disk_or_in_argv(self):
        bundle = self.assign("run-a")
        for path in self.config.slots.state_dir.rglob("*"):
            if path.is_file():
                self.assertNotIn(bundle["private_key"], path.read_text())
        for argv, _stdin in self.world.calls:
            self.assertNotIn(bundle["private_key"], " ".join(argv))

    def test_pool_exhausted_fails_with_help(self):
        for run in ("a", "b", "c", "d"):
            self.assign(run)
        with self.assertRaises(slots.SlotError) as ctx:
            self.assign("e")
        self.assertIn("slot-release", ctx.exception.help)

    def test_refuses_private_key_not_matching_published_key(self):
        self.world.items["lilith-wg-slot-1"]["private_key"] = _keypair().private
        with self.assertRaisesRegex(slots.SlotError, "does not derive"):
            self.assign("run-a", slot=1)
        self.assertEqual(self.state["slot1_enabled"], "false")

    def test_refuses_allowlist_drift_unless_overridden(self):
        self.world.files["clusters/archibald/lilith-wg/allowlist.conf"] = TALOS_ALLOWLIST.replace("8242   off", "8242   on")
        with self.assertRaisesRegex(slots.SlotError, "disagree"):
            self.assign("run-a")
        self.assertEqual(self.assign("run-a", ignore_allowlist_drift=True)["slot"], 1)

    def test_stale_hub_key_in_config_warns_and_uses_1password(self):
        warnings = []
        bundle = slots.assign(self.config, self.op, "run-a", runner=self.world, warn=warnings.append, now=self.clock)
        self.assertEqual(bundle["hub_public_key"], self.hub.public)
        self.assertEqual(len(warnings), 1)  # example carries the real archibald key, not this test hub's


class ReleaseTests(VariantCBase):
    def test_release_rotates_disables_and_marks_rotating(self):
        self.assign("run-a")
        result = self.release(run="run-a")
        item = self.world.items["lilith-wg-slot-1"]
        self.assertNotEqual(item["private_key"], self.slot_keys[1].private)
        self.assertEqual(render.derive_public_key(item["private_key"]), item["public_key"])
        self.assertEqual(self.state["slot1_public_key"], item["public_key"])
        self.assertEqual(self.state["slot1_enabled"], "false")
        self.assertEqual(self.state["slot1_run"], "-")
        self.assertEqual(result["new_public_key"], item["public_key"])
        self.assertEqual(result["released_run"], "run-a")
        self.assertEqual(self.status()[1], slots.SlotState.ROTATING)
        for argv, _stdin in self.world.calls:
            self.assertNotIn(item["private_key"], " ".join(argv))

    def test_published_state_is_written_before_the_keypair(self):
        self.assign("run-a")
        self.world.calls.clear()
        self.release(slot=1)
        edits = [argv[3] for argv, _ in self.world.calls if argv[1:3] == ["item", "edit"]]
        self.assertEqual(edits, ["lilith-wg-slots", "lilith-wg-slot-1"])

    def test_failed_keypair_write_still_revokes_and_can_be_repaired(self):
        self.assign("run-a")
        old_pub = self.slot_keys[1].public
        self.world.fail_edit_on = "lilith-wg-slot-1"
        with self.assertRaises(slots.SlotError) as ctx:
            self.release(slot=1)
        self.assertIn("re-run", ctx.exception.help)
        self.assertNotEqual(self.state["slot1_public_key"], old_pub)  # hub drops the old key
        self.assertEqual(self.state["slot1_enabled"], "false")
        self.assertEqual(self.status()[1], slots.SlotState.ROTATING)
        self.world.fail_edit_on = None
        self.release(slot=1)
        self.now = T0 + timedelta(seconds=self.config.slots.propagation_seconds + 1)
        self.assertEqual(self.status()[1], slots.SlotState.FREE)
        self.assertEqual(self.assign("run-b", slot=1)["slot"], 1)

    def test_release_unknown_run_fails_with_help(self):
        with self.assertRaises(slots.SlotError) as ctx:
            self.release(run="nope")
        self.assertIn("--slot", ctx.exception.help)


if __name__ == "__main__":
    unittest.main()
