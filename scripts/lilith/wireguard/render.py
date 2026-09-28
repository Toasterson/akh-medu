#!/usr/bin/env python3
"""Render WireGuard + nftables + userspace-client artifacts for Lilith C6a.

Pure stdlib except for the optional `cryptography` fallback used only when
the `wg` binary is unavailable on the machine running this script. Nothing in
this module talks to a network or a remote host — it is pure rendering over
a `wg-config.toml` (see `wg-config.toml.example`) plus a persisted peer-state
JSON file. `peer.py` is the only piece that opens an SSH connection.

Never applied automatically: this script only writes files under an
explicit `--out-dir`, plus key material under `[keys].dir`. Both refuse to
resolve inside this repository checkout.

ADR: docs/ai/decisions/116-lilith-chief-of-staff-guided-rl.md (Decision 3)
Plan: docs/ai/plans/2026-09-27-lilith-chief-of-staff-rl-runpod.md (C6a row)
Runbook: docs/ops/lilith-wireguard.md
"""

from __future__ import annotations

import argparse
import base64
import ipaddress
import json
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# onetun's smoltcp interface backend caps the number of IP addresses per
# protocol at 7 (see README "Maximum number of tunnels"); above that a
# single onetun process must be split into several, each with its own
# --source-peer-ip is NOT required (all forwards share one peer identity),
# so we simply run multiple onetun invocations on the same identity.
ONETUN_MAX_FORWARDS_PER_PROCESS = 7

REPO_MARKERS = (".git", "Cargo.toml")


# --------------------------------------------------------------------------
# Config model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Service:
    name: str
    host: str
    port: int
    protocol: str
    bypass_gateway: bool
    description: str = ""
    local_port: int | None = None

    def effective_local_port(self) -> int:
        return self.local_port if self.local_port is not None else self.port

    @staticmethod
    def from_dict(d: dict) -> "Service":
        protocol = str(d["protocol"]).lower()
        if protocol not in ("tcp", "udp"):
            raise ValueError(f"service {d.get('name')!r}: protocol must be tcp or udp, got {protocol!r}")
        return Service(
            name=str(d["name"]),
            host=str(d["host"]),
            port=int(d["port"]),
            protocol=protocol,
            bypass_gateway=bool(d.get("bypass_gateway", False)),
            description=str(d.get("description", "")),
            local_port=int(d["local_port"]) if "local_port" in d else None,
        )


@dataclass(frozen=True)
class TunnelSection:
    variant: str
    subnet: str
    listen_port: int
    dns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HubSection:
    host: str
    public_endpoint: str
    tunnel_address: str
    ssh_user: str
    ssh_target: str
    wg_interface: str
    config_path: str
    ssh_port: int = 22


@dataclass(frozen=True)
class GatewaySection:
    host: str
    tunnel_address: str
    persistent_keepalive: int
    lan_cidr: str
    lan_interface: str
    wg_interface: str
    config_path: str


@dataclass(frozen=True)
class PodsSection:
    address_range: str


@dataclass(frozen=True)
class Config:
    tunnel: TunnelSection
    hub: HubSection
    gateway: GatewaySection
    pods: PodsSection
    services: tuple[Service, ...]
    keys_dir: Path
    nft_table: str
    source_path: Path | None = None

    def allow_services(self) -> list[Service]:
        return [s for s in self.services if not s.bypass_gateway]

    def bypass_services(self) -> list[Service]:
        return [s for s in self.services if s.bypass_gateway]

    def is_variant_b(self) -> bool:
        return self.tunnel.variant.strip().upper() == "B"


def _assert_outside_repo(path: Path, *, what: str) -> Path:
    resolved = path.expanduser().resolve()
    for parent in (resolved, *resolved.parents):
        if any((parent / marker).exists() for marker in REPO_MARKERS):
            raise ValueError(
                f"{what} ({resolved}) resolves inside a repository checkout "
                f"(found {parent}); it must live outside the repo"
            )
    return resolved


def load_config(path: Path) -> Config:
    raw = tomllib.loads(Path(path).read_text())
    try:
        tunnel = TunnelSection(
            variant=str(raw["tunnel"]["variant"]),
            subnet=str(raw["tunnel"]["subnet"]),
            listen_port=int(raw["tunnel"]["listen_port"]),
            dns=list(raw["tunnel"].get("dns", [])),
        )
        hub = HubSection(
            host=str(raw["hub"]["host"]),
            public_endpoint=str(raw["hub"]["public_endpoint"]),
            tunnel_address=str(raw["hub"]["tunnel_address"]),
            ssh_user=str(raw["hub"]["ssh_user"]),
            ssh_target=str(raw["hub"]["ssh_target"]),
            wg_interface=str(raw["hub"]["wg_interface"]),
            config_path=str(raw["hub"]["config_path"]),
            ssh_port=int(raw["hub"].get("ssh_port", 22)),
        )
        gateway = GatewaySection(
            host=str(raw["gateway"]["host"]),
            tunnel_address=str(raw["gateway"]["tunnel_address"]),
            persistent_keepalive=int(raw["gateway"]["persistent_keepalive"]),
            lan_cidr=str(raw["gateway"]["lan_cidr"]),
            lan_interface=str(raw["gateway"]["lan_interface"]),
            wg_interface=str(raw["gateway"]["wg_interface"]),
            config_path=str(raw["gateway"]["config_path"]),
        )
        pods = PodsSection(address_range=str(raw["pods"]["address_range"]))
        services = tuple(Service.from_dict(s) for s in raw.get("services", []))
        keys_dir = _assert_outside_repo(Path(raw["keys"]["dir"]), what="[keys].dir")
        nft_table = str(raw["nftables"]["table"])
    except KeyError as exc:
        raise ValueError(f"wg-config.toml missing required key: {exc}") from exc

    if tunnel.variant.strip().upper() not in ("A", "B"):
        raise ValueError(f"tunnel.variant must be 'A' or 'B', got {tunnel.variant!r}")

    _validate_local_port_collisions(services)

    return Config(
        tunnel=tunnel, hub=hub, gateway=gateway, pods=pods, services=services,
        keys_dir=keys_dir, nft_table=nft_table, source_path=Path(path),
    )


def _validate_local_port_collisions(services: tuple[Service, ...]) -> None:
    seen: dict[tuple[int, str], str] = {}
    for svc in services:
        if svc.bypass_gateway:
            continue
        key = (svc.effective_local_port(), svc.protocol)
        if key in seen:
            raise ValueError(
                f"local_port collision: services {seen[key]!r} and {svc.name!r} "
                f"both bind 127.0.0.1:{key[0]}/{key[1]} on the pod — set an explicit "
                f"distinct local_port for one of them"
            )
        seen[key] = svc.name


# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyPair:
    private: str
    public: str


def _wg_binary() -> str | None:
    return shutil.which("wg")


def generate_keypair() -> KeyPair:
    """Generate a fresh WireGuard (Curve25519) keypair.

    Prefers the real `wg` CLI (wireguard-tools) when present. Falls back to
    the `cryptography` package's X25519 primitive, which produces raw
    32-byte keys in the same format WireGuard uses (base64 of the raw
    scalar / point) when neither is available the call fails loudly instead
    of silently degrading.
    """
    wg = _wg_binary()
    if wg:
        priv = subprocess.run([wg, "genkey"], capture_output=True, check=True, text=True).stdout.strip()
        pub = subprocess.run(
            [wg, "pubkey"], input=priv, capture_output=True, check=True, text=True
        ).stdout.strip()
        return KeyPair(private=priv, public=pub)

    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    except ImportError as exc:
        raise RuntimeError(
            "cannot generate a WireGuard keypair: neither the `wg` binary "
            "(wireguard-tools) nor the `cryptography` Python package is "
            "available. Install one of the two."
        ) from exc

    sk = X25519PrivateKey.generate()
    priv_bytes = sk.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_bytes = sk.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    return KeyPair(private=base64.b64encode(priv_bytes).decode(), public=base64.b64encode(pub_bytes).decode())


def load_or_create_keypair(keys_dir: Path, name: str) -> KeyPair:
    """Persistent keypair for the hub / gateway (long-lived identities).

    Pod keypairs are deliberately NOT handled by this function: they are
    ephemeral, minted fresh per pod session by peer.py, and never persisted
    to disk anywhere (see peer.py's `add` command).
    """
    keys_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    priv_path = keys_dir / f"{name}.private"
    pub_path = keys_dir / f"{name}.public"
    if priv_path.exists() and pub_path.exists():
        return KeyPair(private=priv_path.read_text().strip(), public=pub_path.read_text().strip())
    kp = generate_keypair()
    priv_path.write_text(kp.private + "\n")
    priv_path.chmod(0o600)
    pub_path.write_text(kp.public + "\n")
    pub_path.chmod(0o600)
    return kp


# --------------------------------------------------------------------------
# Peer state (pod peers persisted outside the repo)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PodPeer:
    name: str
    address: str  # e.g. "10.73.0.17/32"
    public_key: str
    created_utc: str
    runpod_pod_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "address": self.address,
            "public_key": self.public_key,
            "created_utc": self.created_utc,
            "runpod_pod_id": self.runpod_pod_id,
        }

    @staticmethod
    def from_dict(d: dict) -> "PodPeer":
        return PodPeer(
            name=d["name"], address=d["address"], public_key=d["public_key"],
            created_utc=d["created_utc"], runpod_pod_id=d.get("runpod_pod_id"),
        )


def load_peers(state_path: Path) -> list[PodPeer]:
    if not state_path.exists():
        return []
    raw = json.loads(state_path.read_text())
    return [PodPeer.from_dict(d) for d in raw.get("peers", [])]


def save_peers(state_path: Path, peers: list[PodPeer]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps({"peers": [p.to_dict() for p in peers]}, indent=2, sort_keys=True) + "\n")
    tmp.chmod(0o600)
    tmp.replace(state_path)


def allocate_pod_address(config: Config, existing: list[PodPeer]) -> str:
    network = ipaddress.ip_network(config.pods.address_range, strict=False)
    used = {ipaddress.ip_interface(p.address).ip for p in existing}
    reserved = {
        ipaddress.ip_interface(config.hub.tunnel_address).ip,
        ipaddress.ip_interface(config.gateway.tunnel_address).ip,
    }
    for host in network.hosts():
        if host in used or host in reserved:
            continue
        return f"{host}/32"
    raise RuntimeError(f"no free pod address remains in {config.pods.address_range}")


# --------------------------------------------------------------------------
# wg-quick style config rendering
# --------------------------------------------------------------------------

_HEADER = "# Generated by scripts/lilith/wireguard/render.py — do not hand-edit; re-render instead.\n"


def render_hub_conf(config: Config, hub_keys: KeyPair, gateway_pubkey: str, pods: list[PodPeer]) -> str:
    lines = [_HEADER, "[Interface]"]
    lines += [
        f"PrivateKey = {hub_keys.private}",
        f"Address = {config.hub.tunnel_address}",
        f"ListenPort = {config.tunnel.listen_port}",
        "",
    ]
    if not config.is_variant_b():
        lines += [
            "# Gateway peer (LAN gateway — dials in, e.g. from hephaistos in Variant A)",
            "[Peer]",
            f"PublicKey = {gateway_pubkey}",
            f"AllowedIPs = {config.gateway.tunnel_address}",
            "",
        ]
    for pod in pods:
        tag = f"added {pod.created_utc}"
        if pod.runpod_pod_id:
            tag += f", runpod_pod_id={pod.runpod_pod_id}"
        lines += [
            f"# Pod peer: {pod.name} ({tag})",
            "[Peer]",
            f"PublicKey = {pod.public_key}",
            f"AllowedIPs = {pod.address}",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def render_gateway_conf(config: Config, gateway_keys: KeyPair, hub_pubkey: str) -> str:
    lines = [_HEADER, "[Interface]"]
    lines += [f"PrivateKey = {gateway_keys.private}", f"Address = {config.gateway.tunnel_address}", ""]
    lines += ["[Peer]", f"PublicKey = {hub_pubkey}", f"AllowedIPs = {config.hub.tunnel_address}, {config.tunnel.subnet}"]
    if not config.is_variant_b():
        # Variant A: the gateway dials OUT to the hub's public endpoint.
        lines.append(f"Endpoint = {config.hub.public_endpoint}")
    lines.append(f"PersistentKeepalive = {config.gateway.persistent_keepalive}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_pod_conf(config: Config, pod_keys: KeyPair, hub_pubkey: str, pod_address: str) -> str:
    """Kernel-WireGuard fallback config for a pod, used only when /dev/net/tun
    and NET_ADMIN are actually available inside the pod (rare on Runpod).
    AllowedIPs is scoped to the tunnel subnet only — this is a service
    tunnel, never a default-route VPN.
    """
    lines = [_HEADER, "[Interface]"]
    lines += [f"PrivateKey = {pod_keys.private}", f"Address = {pod_address}"]
    if config.tunnel.dns:
        lines.append(f"DNS = {', '.join(config.tunnel.dns)}")
    lines.append("")
    lines += [
        "[Peer]",
        f"PublicKey = {hub_pubkey}",
        f"AllowedIPs = {config.tunnel.subnet}",
        f"Endpoint = {config.hub.public_endpoint}",
        "PersistentKeepalive = 25",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------
# Userspace pod-client rendering (onetun / wireproxy)
# --------------------------------------------------------------------------


def _chunk(items: list[Service], size: int) -> list[list[Service]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def render_onetun_commands(config: Config, hub_pubkey: str, pod_address: str) -> list[str]:
    """One or more `onetun` invocations covering every allowlisted service.

    Split into batches of at most ONETUN_MAX_FORWARDS_PER_PROCESS per
    process, per onetun's own documented smoltcp interface-address limit.

    Deliberately takes no pod private key: the command reads it from
    `$LILITH_WG_PRIVATE_KEY` at run time via ordinary shell expansion.
    Caveat (documented in docs/ops/lilith-wireguard.md): shell expansion
    means the expanded value ends up in this process's own argv, so it is
    visible to `ps`/`/proc/<pid>/cmdline` from *within the same pod
    container* (not from outside it — Runpod pods are single-tenant
    containers). onetun's `--help` documents environment-variable support
    for every flag; if onetun accepts the key via its own env var without
    ever taking it as an argument, prefer that instead once confirmed
    against the deployed onetun version — this is called out as a
    pre-flight check in the runbook rather than guessed here.
    """
    addr_only = pod_address.split("/")[0]
    services = config.allow_services()
    commands = []
    for batch in _chunk(services, ONETUN_MAX_FORWARDS_PER_PROCESS):
        forwards = " ".join(
            f"127.0.0.1:{s.effective_local_port()}:{s.host}:{s.port}"
            + ("" if s.protocol == "tcp" else f":{s.protocol.upper()}")
            for s in batch
        )
        commands.append(
            "onetun " + forwards
            + f" --endpoint-addr {config.hub.public_endpoint}"
            + f" --endpoint-public-key '{hub_pubkey}'"
            + " --private-key \"$LILITH_WG_PRIVATE_KEY\""
            + f" --source-peer-ip {addr_only}"
            + " --keep-alive 25"
            + " --log info"
        )
    return commands


def render_wireproxy_conf(config: Config, hub_pubkey: str, pod_address: str) -> str:
    lines = [_HEADER, "[Interface]"]
    lines += [f"Address = {pod_address}", "PrivateKey = $LILITH_WG_PRIVATE_KEY"]
    if config.tunnel.dns:
        lines.append(f"DNS = {', '.join(config.tunnel.dns)}")
    lines.append("")
    lines += [
        "[Peer]",
        f"PublicKey = {hub_pubkey}",
        f"Endpoint = {config.hub.public_endpoint}",
        f"AllowedIPs = {config.tunnel.subnet}",
        "PersistentKeepalive = 25",
        "",
    ]
    for svc in config.allow_services():
        if svc.protocol != "tcp":
            lines.append(f"# {svc.name}: UDP not supported by wireproxy TCPClientTunnel — use onetun instead")
            continue
        lines += [
            "[TCPClientTunnel]",
            f"# {svc.description or svc.name}",
            f"BindAddress = 127.0.0.1:{svc.effective_local_port()}",
            f"Target = {svc.host}:{svc.port}",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------
# nftables rendering — everything lives in one dedicated table.
# --------------------------------------------------------------------------


def render_hub_nft(config: Config, pods: list[PodPeer]) -> str:
    """Hub: forward only wg<->wg between an active pod /32 and the gateway's
    tunnel address. No forwarding to any other interface, ever (the hub
    never NATs pod traffic to the open internet). Defense in depth on top
    of WireGuard's own cryptokey routing (AllowedIPs already prevents a
    pod from presenting any other peer's source address), scoped so this
    table never touches unrelated (docker/ufw) forwarding: it only ever
    actively matches iifname wg0, and its policy is accept, so anything
    that isn't wg0-sourced simply falls through untouched.
    """
    wg_if = config.hub.wg_interface
    gw_ip = ipaddress.ip_interface(config.gateway.tunnel_address).ip
    lines = [
        _HEADER.strip(),
        f"table inet {config.nft_table} {{",
        "    chain forward {",
        "        type filter hook forward priority filter + 10; policy accept;",
        "        ct state established,related accept",
    ]
    for pod in pods:
        pod_ip = ipaddress.ip_interface(pod.address).ip
        lines.append(
            f'        iifname "{wg_if}" oifname "{wg_if}" ip saddr {pod_ip}/32 ip daddr {gw_ip}/32 '
            f"accept comment \"pod:{pod.name}\""
        )
    lines += [
        f'        iifname "{wg_if}" drop comment "deny: anything else arriving from the tunnel"',
        "    }",
        "}",
    ]
    return "\n".join(lines) + "\n"


def render_gateway_nft(config: Config) -> str:
    """Gateway: forward (and masquerade) only the allowlisted LAN
    IP:port/protocol pairs, sourced only from the tunnel subnet. Any service
    whose host is the gateway's own LAN address is additionally allowed
    through the input chain (it terminates locally, never reaching forward —
    the classification is a plain string match against config.gateway.host,
    so that field must be the literal LAN IP, not a hostname; see the
    warning in wg-config.toml.example). Both chains use policy accept and
    end in an explicit drop scoped to `iifname wg0` — so, like the hub
    table, this never touches traffic that doesn't come from the tunnel.
    """
    wg_if = config.gateway.wg_interface
    lan_if = config.gateway.lan_interface
    subnet = config.tunnel.subnet
    gw_host = config.gateway.host
    allow = config.allow_services()
    self_services = [s for s in allow if s.host == gw_host]
    remote_services = [s for s in allow if s.host != gw_host]

    lines = [_HEADER.strip(), f"table inet {config.nft_table} {{"]

    lines.append("    chain input {")
    lines.append("        type filter hook input priority filter + 10; policy accept;")
    for svc in self_services:
        lines.append(
            f'        iifname "{wg_if}" ip saddr {subnet} {svc.protocol} dport {svc.port} '
            f"accept comment \"svc:{svc.name}\""
        )
    lines.append(f'        iifname "{wg_if}" ip saddr {subnet} drop comment "deny: not in the service allowlist"')
    lines.append("    }")

    lines.append("    chain forward {")
    lines.append("        type filter hook forward priority filter + 10; policy accept;")
    lines.append("        ct state established,related accept")
    for svc in remote_services:
        lines.append(
            f'        iifname "{wg_if}" oifname "{lan_if}" ip saddr {subnet} ip daddr {svc.host} '
            f"{svc.protocol} dport {svc.port} accept comment \"svc:{svc.name}\""
        )
    lines.append(f'        iifname "{wg_if}" ip saddr {subnet} drop comment "deny: not in the service allowlist"')
    lines.append("    }")

    lines.append("    chain postrouting {")
    lines.append("        type nat hook postrouting priority srcnat; policy accept;")
    if remote_services:
        lines.append(
            f'        iifname "{wg_if}" oifname "{lan_if}" ip saddr {subnet} masquerade '
            'comment "so LAN services see hephaistos, not the tunnel subnet"'
        )
    lines.append("    }")

    lines.append("}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def default_peers_state(config: Config) -> Path:
    return config.keys_dir.parent / "peers.json"


def render_all(config: Config, out_dir: Path, peers: list[PodPeer]) -> dict[str, str]:
    hub_keys = load_or_create_keypair(config.keys_dir, "hub")
    gateway_keys = load_or_create_keypair(config.keys_dir, "gateway")

    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = {
        "hub.conf": render_hub_conf(config, hub_keys, gateway_keys.public, peers),
        "hub.nft": render_hub_nft(config, peers),
        "gateway.conf": render_gateway_conf(config, gateway_keys, hub_keys.public),
        "gateway.nft": render_gateway_nft(config),
    }
    # A template pod-client bundle, for docs / dry runs only — real pod
    # peers are minted by `peer.py add`, never by this bulk render.
    template_pod_address = allocate_pod_address(config, peers)
    template_keys = generate_keypair()
    onetun_cmds = render_onetun_commands(config, hub_keys.public, template_pod_address)
    rendered["pod-client.conf"] = render_pod_conf(config, template_keys, hub_keys.public, template_pod_address)
    rendered["pod-client.wireproxy.conf"] = render_wireproxy_conf(config, hub_keys.public, template_pod_address)
    rendered["pod-client.onetun.sh"] = "#!/bin/sh\n" + "\n".join(onetun_cmds) + "\n"

    for name, text in rendered.items():
        path = out_dir / name
        path.write_text(text)
        if name.endswith(".sh"):
            path.chmod(0o755)
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="path to wg-config.toml")
    parser.add_argument("--out-dir", type=Path, required=True, help="directory to write rendered artifacts into")
    parser.add_argument(
        "--peers-state", type=Path, default=None,
        help="peers.json to include already-added pod peers (default: sibling of [keys].dir)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    peers_state = args.peers_state or default_peers_state(config)
    peers = load_peers(peers_state)
    rendered = render_all(config, args.out_dir, peers)
    for name in rendered:
        print(f"wrote {args.out_dir / name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
