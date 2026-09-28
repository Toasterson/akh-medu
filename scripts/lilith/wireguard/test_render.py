"""Tests for the Lilith C6a WireGuard artifact renderer and peer manager.

Pure unittest + stdlib, matching scripts/session-platform/test_lilith_vm_executor.py's
module-loading pattern (these two scripts are not an importable package).
Nothing here opens a network connection or touches a real host: peer.py is
only ever exercised with --dry-run / dry_run=True.
"""

from __future__ import annotations

import importlib.util
import ipaddress
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    # dataclasses (e.g. its KW_ONLY sentinel resolution) looks the module up
    # via sys.modules by __module__ name, so register it before executing —
    # otherwise Python 3.13+ raises AttributeError deep inside `dataclass()`.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


render = _load("render")
peer = _load("peer")


def make_config(rm, keys_dir: Path, *, variant: str = "A", extra_services: list[dict] | None = None) -> "render.Config":
    services = [
        {"name": "akhsessd", "host": "192.168.1.114", "port": 7401, "protocol": "tcp", "bypass_gateway": False},
        {"name": "self-svc", "host": "192.168.1.114", "port": 7402, "protocol": "tcp", "bypass_gateway": False},
        {"name": "remote-svc", "host": "192.168.1.115", "port": 8080, "protocol": "tcp", "bypass_gateway": False},
        {"name": "public-svc", "host": "example.org", "port": 443, "protocol": "tcp", "bypass_gateway": True},
    ]
    if extra_services:
        services += extra_services
    return rm.Config(
        tunnel=rm.TunnelSection(variant=variant, subnet="10.73.0.0/24", listen_port=51820, dns=[]),
        hub=rm.HubSection(
            host="archibald.example.net", public_endpoint="archibald.example.net:51820",
            tunnel_address="10.73.0.1/32", ssh_user="root", ssh_target="root@archibald.example.net",
            wg_interface="wg0", config_path="/etc/wireguard/wg0.conf",
        ),
        gateway=rm.GatewaySection(
            host="192.168.1.114", tunnel_address="10.73.0.2/32", persistent_keepalive=25,
            lan_cidr="192.168.1.0/24", lan_interface="eth0", wg_interface="wg0",
            config_path="/etc/wireguard/wg0.conf",
        ),
        pods=rm.PodsSection(address_range="10.73.0.16/28"),
        services=tuple(rm.Service.from_dict(s) for s in services),
        keys_dir=keys_dir, nft_table="lilith_wg",
    )


class ConfigLoadingTests(unittest.TestCase):
    def test_example_config_loads_and_is_variant_a(self):
        with tempfile.TemporaryDirectory() as td:
            keys_dir = Path(td) / "keys"
            text = (HERE / "wg-config.variant-a.toml.example").read_text().replace(
                "~/.config/akh-medu/lilith-wireguard/keys", str(keys_dir)
            )
            cfg_path = Path(td) / "wg-config.toml"
            cfg_path.write_text(text)
            config = render.load_config(cfg_path)
            self.assertFalse(config.is_variant_b())
            self.assertGreaterEqual(len(config.allow_services()), 6)
            self.assertIn("anima-mcp", [s.name for s in config.bypass_services()])

    def test_local_port_collision_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            keys_dir = Path(td) / "keys"
            cfg = make_config(render, keys_dir, extra_services=[
                {"name": "dup", "host": "192.168.1.200", "port": 7401, "protocol": "tcp", "bypass_gateway": False},
            ])
            # Force the collision through the real validator (make_config's
            # services already satisfy from_dict; re-run the collision check
            # directly since Config is normally built by load_config).
            with self.assertRaisesRegex(ValueError, "local_port collision"):
                render._validate_local_port_collisions(cfg.services)

    def test_keys_dir_inside_repo_is_rejected(self):
        repo_root = HERE.parent.parent.parent  # scripts/lilith/wireguard -> repo root
        self.assertTrue((repo_root / ".git").exists() or (repo_root / "Cargo.toml").exists())
        with tempfile.TemporaryDirectory() as td:
            text = (HERE / "wg-config.variant-a.toml.example").read_text().replace(
                "~/.config/akh-medu/lilith-wireguard/keys", str(repo_root / "scripts" / "lilith-wg-keys-should-fail")
            )
            cfg_path = Path(td) / "wg-config.toml"
            cfg_path.write_text(text)
            with self.assertRaises(ValueError):
                render.load_config(cfg_path)


class RenderingDeterminismTests(unittest.TestCase):
    def test_hub_and_gateway_render_deterministically(self):
        with tempfile.TemporaryDirectory() as td:
            keys_dir = Path(td) / "keys"
            config = make_config(render, keys_dir)
            peers_state = Path(td) / "peers.json"
            peers = [render.PodPeer(name="pod-a", address="10.73.0.17/32", public_key="A" * 43 + "=", created_utc="t")]
            render.save_peers(peers_state, peers)

            out1 = Path(td) / "out1"
            out2 = Path(td) / "out2"
            rendered1 = render.render_all(config, out1, render.load_peers(peers_state))
            rendered2 = render.render_all(config, out2, render.load_peers(peers_state))

            for name in ("hub.conf", "hub.nft", "gateway.conf", "gateway.nft"):
                self.assertEqual(rendered1[name], rendered2[name], f"{name} was not deterministic")

    def test_rerendering_without_config_change_does_not_rotate_hub_gateway_keys(self):
        with tempfile.TemporaryDirectory() as td:
            keys_dir = Path(td) / "keys"
            config = make_config(render, keys_dir)
            first = render.load_or_create_keypair(config.keys_dir, "hub")
            second = render.load_or_create_keypair(config.keys_dir, "hub")
            self.assertEqual(first, second)


class NftAllowlistTests(unittest.TestCase):
    def test_hub_nft_only_allows_active_pods_to_the_gateway(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(render, Path(td) / "keys")
            peers = [
                render.PodPeer(name="pod-a", address="10.73.0.17/32", public_key="k1", created_utc="t"),
                render.PodPeer(name="pod-b", address="10.73.0.18/32", public_key="k2", created_utc="t"),
            ]
            nft = render.render_hub_nft(config, peers)
            self.assertIn("10.73.0.17/32", nft)
            self.assertIn("10.73.0.18/32", nft)
            self.assertIn("ip daddr 10.73.0.2/32", nft)  # gateway tunnel address, the only allowed destination
            self.assertIn('iifname "wg0" drop', nft)  # explicit deny-rest
            # A dedicated table, never a bare "forward" hook takeover of
            # unrelated traffic: policy is accept, so only iifname-wg0
            # traffic is ever touched by this table.
            self.assertIn("policy accept", nft)
            self.assertNotIn("policy drop", nft)

    def test_hub_nft_with_no_pods_still_denies_everything_from_the_tunnel(self):
        config = make_config(render, Path(tempfile.mkdtemp()) / "keys")
        nft = render.render_hub_nft(config, [])
        self.assertIn('iifname "wg0" drop', nft)
        self.assertNotIn("AllowedIPs", nft)  # this is nft, not a wg conf

    def test_gateway_nft_matches_exactly_the_allowlist_nothing_else(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(render, Path(td) / "keys")
            nft = render.render_gateway_nft(config)
            # Allowed, self-hosted (gateway.host == service.host) -> input chain
            self.assertIn("tcp dport 7401", nft)
            self.assertIn("tcp dport 7402", nft)
            # Allowed, remote LAN host -> forward chain + masquerade
            self.assertIn("ip daddr 192.168.1.115", nft)
            self.assertIn("tcp dport 8080", nft)
            self.assertIn("masquerade", nft)
            # bypass_gateway service must NEVER appear in the gateway's
            # nft rules at all — it's not supposed to transit the gateway.
            self.assertNotIn("example.org", nft)
            self.assertNotIn("443", nft)
            # Explicit deny-rest on both chains that touch the tunnel.
            self.assertEqual(nft.count('drop comment "deny: not in the service allowlist"'), 2)

    def test_gateway_nft_self_vs_remote_split_requires_ip_match(self):
        # Regression guard: if gateway.host is a hostname while services use
        # the gateway's IP, the self/remote classification silently breaks
        # (a self-hosted service is misclassified as remote). This is
        # exactly the bug fixed in wg-config.toml.example — keep it fixed.
        with tempfile.TemporaryDirectory() as td:
            config = make_config(render, Path(td) / "keys")
            mismatched = render.GatewaySection(
                host="hephaistos.local",  # WRONG: services use the IP, not this hostname
                tunnel_address=config.gateway.tunnel_address,
                persistent_keepalive=config.gateway.persistent_keepalive,
                lan_cidr=config.gateway.lan_cidr, lan_interface=config.gateway.lan_interface,
                wg_interface=config.gateway.wg_interface, config_path=config.gateway.config_path,
            )
            broken = render.Config(
                tunnel=config.tunnel, hub=config.hub, gateway=mismatched, pods=config.pods,
                services=config.services, keys_dir=config.keys_dir, nft_table=config.nft_table,
            )
            correct_nft = render.render_gateway_nft(config)
            broken_nft = render.render_gateway_nft(broken)
            # With the correct (IP) host, the self-hosted service reaches
            # the input chain. With the mismatched hostname, it does not.
            self.assertIn("tcp dport 7401", correct_nft.split("chain forward")[0])
            self.assertNotIn("tcp dport 7401", broken_nft.split("chain forward")[0])


class PeerAllocationTests(unittest.TestCase):
    def test_addresses_are_allocated_sequentially_and_avoid_reserved(self):
        config = make_config(render, Path(tempfile.mkdtemp()) / "keys")
        first = render.allocate_pod_address(config, [])
        self.assertNotEqual(ipaddress.ip_interface(first).ip, ipaddress.ip_interface(config.hub.tunnel_address).ip)
        self.assertNotEqual(ipaddress.ip_interface(first).ip, ipaddress.ip_interface(config.gateway.tunnel_address).ip)

        existing = [render.PodPeer(name="p1", address=first, public_key="k", created_utc="t")]
        second = render.allocate_pod_address(config, existing)
        self.assertNotEqual(first, second)

    def test_removed_peer_address_becomes_available_again(self):
        config = make_config(render, Path(tempfile.mkdtemp()) / "keys")
        a = render.allocate_pod_address(config, [])
        peers = [render.PodPeer(name="p1", address=a, public_key="k", created_utc="t")]
        b = render.allocate_pod_address(config, peers)
        self.assertNotEqual(a, b)
        # p1 removed -> "a" is free again and is the first candidate.
        peers_after_removal = [p for p in peers if p.name != "p1"]
        reallocated = render.allocate_pod_address(config, peers_after_removal)
        self.assertEqual(reallocated, a)

    def test_pool_exhaustion_raises(self):
        config = make_config(render, Path(tempfile.mkdtemp()) / "keys")
        network = ipaddress.ip_network(config.pods.address_range, strict=False)
        reserved = {
            ipaddress.ip_interface(config.hub.tunnel_address).ip,
            ipaddress.ip_interface(config.gateway.tunnel_address).ip,
        }
        peers = [
            render.PodPeer(name=f"p{i}", address=f"{h}/32", public_key="k", created_utc="t")
            for i, h in enumerate(network.hosts())
            if h not in reserved
        ]
        with self.assertRaises(RuntimeError):
            render.allocate_pod_address(config, peers)

    def test_peers_round_trip_through_disk(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "peers.json"
            peers = [
                render.PodPeer(name="p1", address="10.73.0.17/32", public_key="k1", created_utc="2026-09-28T00:00:00+00:00", runpod_pod_id="rp-1"),
            ]
            render.save_peers(state, peers)
            loaded = render.load_peers(state)
            self.assertEqual(loaded, peers)
            self.assertEqual(oct(state.stat().st_mode)[-3:], "600")


class PeerCliDryRunTests(unittest.TestCase):
    def test_add_dry_run_does_not_persist_or_touch_network_state(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(render, Path(td) / "keys")
            peers_state = Path(td) / "peers.json"
            rc = peer.cmd_add(config, peers_state, "pod-x", None, dry_run=True)
            self.assertEqual(rc, 0)
            self.assertFalse(peers_state.exists(), "dry-run add must not persist peer state")

    def test_remove_of_unknown_peer_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(render, Path(td) / "keys")
            peers_state = Path(td) / "peers.json"
            rc = peer.cmd_remove(config, peers_state, "does-not-exist", dry_run=True)
            self.assertEqual(rc, 1)

    def test_add_same_name_twice_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(render, Path(td) / "keys")
            peers_state = Path(td) / "peers.json"
            render.save_peers(peers_state, [render.PodPeer(name="pod-x", address="10.73.0.17/32", public_key="k", created_utc="t")])
            rc = peer.cmd_add(config, peers_state, "pod-x", None, dry_run=True)
            self.assertEqual(rc, 1)


class OnetunBatchingTests(unittest.TestCase):
    def test_batches_never_exceed_the_smoltcp_limit(self):
        with tempfile.TemporaryDirectory() as td:
            extra = [
                {"name": f"svc{i}", "host": "192.168.1.200", "port": 9000 + i, "protocol": "tcp", "bypass_gateway": False}
                for i in range(10)
            ]
            config = make_config(render, Path(td) / "keys", extra_services=extra)
            commands = render.render_onetun_commands(config, "hubpub", "10.73.0.17/32")
            total_forwards = sum(cmd.count("127.0.0.1:") for cmd in commands)
            self.assertEqual(total_forwards, len(config.allow_services()))
            for cmd in commands:
                self.assertLessEqual(cmd.count("127.0.0.1:"), render.ONETUN_MAX_FORWARDS_PER_PROCESS)
            self.assertGreater(len(commands), 1, "expected the allowlist to be split across processes")

    def test_udp_service_gets_protocol_suffix(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(render, Path(td) / "keys", extra_services=[
                {"name": "udp-svc", "host": "192.168.1.201", "port": 5000, "protocol": "udp", "bypass_gateway": False},
            ])
            commands = render.render_onetun_commands(config, "hubpub", "10.73.0.17/32")
            joined = " ".join(commands)
            self.assertIn("192.168.1.201:5000:UDP", joined)


class NoPrivateKeyMaterialInRepoTests(unittest.TestCase):
    def test_repo_tree_has_no_generated_key_files(self):
        for pattern in ("*.private", "*.public", "peers.json"):
            hits = list(HERE.glob(pattern))
            self.assertEqual(hits, [], f"found generated key/state files matching {pattern} inside the repo: {hits}")

    def test_example_config_contains_no_key_material(self):
        text = (HERE / "wg-config.variant-a.toml.example").read_text()
        self.assertNotIn("PrivateKey", text)
        self.assertNotIn("public_key =", text)
        # Variant C carries only the hub's PUBLIC key; never a private one.
        text_c = (HERE / "wg-config.toml.example").read_text()
        self.assertNotIn("PrivateKey", text_c)
        self.assertNotRegex(text_c, r"(?m)^\s*private_key\s*=")

    def test_kernel_conf_is_never_written_under_out_dir_without_explicit_out_dir(self):
        # render_all only ever writes under the caller-supplied --out-dir; it
        # must never default to a path under this repo.
        with tempfile.TemporaryDirectory() as td:
            config = make_config(render, Path(td) / "keys")
            out_dir = Path(td) / "out"
            render.render_all(config, out_dir, [])
            for path in out_dir.rglob("*"):
                self.assertFalse(str(path).startswith(str(HERE)))


class ShellcheckTests(unittest.TestCase):
    def test_pod_client_passes_shellcheck_if_available(self):
        shellcheck = shutil.which("shellcheck")
        if not shellcheck:
            self.skipTest("shellcheck not installed")
        result = subprocess.run(
            [shellcheck, str(HERE / "pod-client.sh")], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_pod_client_is_at_least_valid_bash_syntax(self):
        bash = shutil.which("bash")
        if not bash:
            self.skipTest("bash not installed")
        result = subprocess.run([bash, "-n", str(HERE / "pod-client.sh")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
