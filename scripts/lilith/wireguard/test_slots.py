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


def _peers_conf(pubkeys: dict[int, str], off: set[int] = frozenset()) -> str:
    lines = ["# slot public_key allowed_ip on|off"]
    for n, key in sorted(pubkeys.items()):
        lines.append(f"slot-{n}  {key}  10.201.0.{10 + n}/32  {'off' if n in off else 'on'}")
    return "\n".join(lines) + "\n"


class FakeWorld:
    """Fake `op` + `git` behind the subprocess.run signature."""

    def __init__(self, items: dict[str, dict[str, str]], peers_conf: str, allowlist: str = TALOS_ALLOWLIST):
        self.items = items
        self.files = {
            "clusters/archibald/lilith-wg/peers.conf": peers_conf,
            "clusters/archibald/lilith-wg/allowlist.conf": allowlist,
        }
        self.calls: list[tuple[list[str], str | None]] = []
        self.fail_edit = False

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
            if self.fail_edit:
                return subprocess.CompletedProcess(argv, 1, "", "[ERROR] permission denied")
            doc = json.loads(input)
            for f in doc["fields"]:
                self.items[argv[3]][f["label"]] = f["value"]
            return ok("{}")
        raise AssertionError(f"unexpected command {argv}")


def _keypair():
    return render.generate_keypair()


class VariantCBase(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        tmp = Path(self.td.name)
        self.talos = tmp / "talos"
        (self.talos / "clusters/archibald/lilith-wg").mkdir(parents=True)
        text = (HERE / "wg-config.toml.example").read_text()
        text = text.replace("~/.config/akh-medu/lilith-wireguard/slots", str(tmp / "leases"))
        text = text.replace('talos_repo = "~/ws/talos"', f'talos_repo = "{self.talos}"')
        text = text.replace("~/.local/bin/op-service-account", OP)
        cfg = tmp / "wg-config.toml"
        cfg.write_text(text)
        self.config = render.load_config(cfg)
        self.hub = _keypair()
        self.slot_keys = {n: _keypair() for n in range(1, 5)}
        items = {"lilith-wg-hub": {"private_key": self.hub.private, "public_key": self.hub.public}}
        for n, kp in self.slot_keys.items():
            items[f"lilith-wg-slot-{n}"] = {"private_key": kp.private, "public_key": kp.public, "address": f"10.201.0.{10 + n}/32"}
        self.world = FakeWorld(items, _peers_conf({n: kp.public for n, kp in self.slot_keys.items()}))
        self.op = slots.OnePassword(command=OP, vault="Homelab", runner=self.world)

    def tearDown(self):
        self.td.cleanup()

    def assign(self, run, **kw):
        return slots.assign(self.config, self.op, run, runner=self.world, **kw)

    def status(self):
        return {v.slot: v.state for v in slots.slot_status(self.config, self.op, runner=self.world)}


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

    def test_lease_dir_inside_repo_is_rejected(self):
        text = self.config.source_path.read_text()
        text = text.replace(str(Path(self.td.name) / "leases"), str(HERE / "leases-should-fail"))
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


class TalosStateTests(VariantCBase):
    def test_allowlist_parity_matches_talos(self):
        self.assertEqual(render.allowlist_drift(self.config, render.parse_talos_allowlist(TALOS_ALLOWLIST)), [])

    def test_allowlist_drift_is_reported_both_ways(self):
        talos = TALOS_ALLOWLIST.replace("8242   off", "8242   on").replace("18240  on", "18240  off")
        drift = render.allowlist_drift(self.config, render.parse_talos_allowlist(talos))
        self.assertEqual(len(drift), 2)
        self.assertTrue(any("akh-sessions-home" in d and "does not allow" in d for d in drift))
        self.assertTrue(any("akh-sandbox-akhsbxd" in d and "no enabled service" in d for d in drift))

    def test_status_states(self):
        self.assertEqual(set(self.status().values()), {slots.SlotState.FREE})
        self.world.files["clusters/archibald/lilith-wg/peers.conf"] = _peers_conf(
            {n: kp.public for n, kp in self.slot_keys.items()}, off={4},
        )
        self.world.items["lilith-wg-slot-3"]["public_key"] = _keypair().public
        self.assign("run-a")
        states = self.status()
        self.assertEqual(states[1], slots.SlotState.LEASED)
        self.assertEqual(states[2], slots.SlotState.FREE)
        self.assertEqual(states[3], slots.SlotState.PENDING_ROTATION)
        self.assertEqual(states[4], slots.SlotState.DISABLED)


class AssignReleaseTests(VariantCBase):
    def test_assign_returns_bundle_and_leases_without_key_on_disk(self):
        bundle = self.assign("run-a", runpod_pod_id="pod123")
        self.assertEqual(bundle["slot"], 1)
        self.assertEqual(bundle["address"], "10.201.0.11/32")
        self.assertEqual(bundle["env"]["LILITH_WG_PRIVATE_KEY"], self.slot_keys[1].private)
        self.assertEqual(bundle["hub_public_key"], self.hub.public)
        self.assertEqual(bundle["env"]["LILITH_WG_HOSTS"], "127.0.0.1 seshat.lab.home board.lab.home")
        self.assertIn("AllowedIPs = 10.0.1.2/32, 10.201.0.1/32, 192.168.1.12/32", bundle["kernel_wg_conf"])
        lease_files = list(self.config.slots.lease_dir.iterdir())
        self.assertEqual([p.name for p in lease_files], ["slot-1.json"])
        self.assertEqual(lease_files[0].stat().st_mode & 0o777, 0o600)
        for path in lease_files:
            self.assertNotIn(self.slot_keys[1].private, path.read_text())
        self.assertEqual(self.assign("run-b")["slot"], 2)

    def test_assign_when_pool_exhausted_fails_with_help(self):
        for run in ("a", "b", "c", "d"):
            self.assign(run)
        with self.assertRaises(slots.SlotError) as ctx:
            self.assign("e")
        self.assertIn("slot-release", ctx.exception.help)

    def test_assign_refuses_key_that_does_not_match_merged_peer(self):
        # public_key field still matches talos, but the private key was swapped.
        self.world.items["lilith-wg-slot-1"]["private_key"] = _keypair().private
        with self.assertRaisesRegex(slots.SlotError, "does not match"):
            self.assign("run-a", slot=1)
        self.assertFalse((self.config.slots.lease_dir / "slot-1.json").exists())

    def test_assign_refuses_allowlist_drift_unless_overridden(self):
        self.world.files["clusters/archibald/lilith-wg/allowlist.conf"] = TALOS_ALLOWLIST.replace("8242   off", "8242   on")
        with self.assertRaisesRegex(slots.SlotError, "disagree"):
            self.assign("run-a")
        self.assertEqual(self.assign("run-a", ignore_allowlist_drift=True)["slot"], 1)

    def test_stale_hub_key_in_config_warns_and_uses_1password(self):
        warnings = []
        bundle = slots.assign(self.config, self.op, "run-a", runner=self.world, warn=warnings.append)
        self.assertEqual(bundle["hub_public_key"], self.hub.public)
        self.assertEqual(len(warnings), 1)  # example carries the real archibald key, not this test hub's

    def test_release_rotates_via_stdin_and_marks_pending(self):
        self.assign("run-a")
        peers_path = self.talos / "clusters/archibald/lilith-wg/peers.conf"
        peers_path.write_text(self.world.files["clusters/archibald/lilith-wg/peers.conf"])
        result = slots.release(self.config, self.op, run="run-a", write_talos=True)
        new_item = self.world.items["lilith-wg-slot-1"]
        self.assertNotEqual(new_item["private_key"], self.slot_keys[1].private)
        self.assertEqual(render.derive_public_key(new_item["private_key"]), new_item["public_key"])
        self.assertEqual(result["new_public_key"], new_item["public_key"])
        for argv, _stdin in self.world.calls:
            self.assertNotIn(new_item["private_key"], " ".join(argv))
        written = peers_path.read_text()
        self.assertIn(f"slot-1  {new_item['public_key']}  10.201.0.11/32  on", written)
        self.assertIn(self.slot_keys[2].public, written)
        self.assertFalse((self.config.slots.lease_dir / "slot-1.json").exists())
        # talos main still has the old key -> not reassignable until merged
        self.assertEqual(self.status()[1], slots.SlotState.PENDING_ROTATION)
        self.world.files["clusters/archibald/lilith-wg/peers.conf"] = written
        self.assertEqual(self.status()[1], slots.SlotState.FREE)

    def test_failed_rotation_keeps_the_lease(self):
        self.assign("run-a")
        self.world.fail_edit = True
        with self.assertRaises(slots.SlotError):
            slots.release(self.config, self.op, slot=1)
        self.assertTrue((self.config.slots.lease_dir / "slot-1.json").exists())

    def test_release_unknown_run_fails_with_help(self):
        with self.assertRaises(slots.SlotError) as ctx:
            slots.release(self.config, self.op, run="nope")
        self.assertIn("--slot", ctx.exception.help)


if __name__ == "__main__":
    unittest.main()
