#!/usr/bin/env python3
"""Add / remove / list Runpod pod peers on the Lilith C6a WireGuard hub.

Used by the supervisor: `peer.py add` at pod start, `peer.py remove` at pod
stop (ADR 116 Decision 3: "the peer added at pod start and removed at
stop"). Never writes a pod's private key anywhere — it is generated
in-memory, pushed live to the hub as a public key only, and the *pod-side*
config (which does need the private key) is printed to stdout once, for the
caller to inject as a Runpod secret. Nothing this script does is applied to
any host unless the caller actually runs it with network access to the hub;
`--dry-run` prints the exact commands instead of running them, which is the
mode used for testing and for review.

Usage (Variant C — chosen; hub = talos clusters/archibald/lilith-wg):
    peer.py slot-status [--fetch]
    peer.py slot-assign <run> [--runpod-pod-id ID] [--slot N] [--fetch]
    peer.py slot-release (--slot N | --run RUN)

Usage (Variants A/B — superseded; hub managed over SSH):
    peer.py add <name> [--runpod-pod-id ID] [--dry-run]
    peer.py remove <name> [--dry-run]
    peer.py list [--live]

Variant C slot keys are read from / rotated in 1Password at call time via
the op-service-account wrapper and never written to disk or logs. The hub
hot-reloads the published peer set (1Password -> ESO), so assign/release
need no git change; see slots.py.

ADR: docs/ai/decisions/116-lilith-chief-of-staff-guided-rl.md (Decision 3)
Plan: docs/ai/plans/2026-09-27-lilith-chief-of-staff-rl-runpod.md (C6a row)
Runbook: docs/ops/lilith-wireguard.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render  # noqa: E402  (local module, path inserted above)
import slots  # noqa: E402


def _ssh_argv(config: render.Config) -> list[str]:
    return [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "ConnectTimeout=8",
        "-p", str(config.hub.ssh_port),
        config.hub.ssh_target,
    ]


def _run_remote(config: render.Config, remote_command: str, *, dry_run: bool, input_text: str | None = None) -> str:
    argv = _ssh_argv(config) + [remote_command]
    if dry_run:
        print(f"DRY-RUN ssh: {' '.join(argv)}", file=sys.stderr)
        if input_text is not None:
            print(f"DRY-RUN stdin ({len(input_text)} bytes, not shown — may contain key material)", file=sys.stderr)
        return ""
    result = subprocess.run(
        argv, input=input_text, capture_output=True, text=True, timeout=20,
    )
    if result.returncode != 0:
        raise RuntimeError(f"remote command failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


def _push_hub_state(config: render.Config, peers: list[render.PodPeer], *, dry_run: bool) -> None:
    """Push the regenerated hub wg conf + nft ruleset to the hub: a live
    `wg set`/`wg syncconf` for immediate effect, plus a persisted config
    file so a reboot keeps the same peer set. Both are scoped to exactly
    the dedicated interface and the dedicated nft table.
    """
    hub_keys = render.load_or_create_keypair(config.keys_dir, "hub")
    gateway_keys = render.load_or_create_keypair(config.keys_dir, "gateway")

    hub_conf = render.render_hub_conf(config, hub_keys, gateway_keys.public, peers)
    hub_nft = render.render_hub_nft(config, peers)

    strip_cmd = f"wg-quick strip {config.hub.wg_interface}"
    # Write the persisted config, then apply live via `wg syncconf` (which
    # diffs and only changes what's different — it does not tear the
    # interface down, so existing sessions are not disrupted).
    _run_remote(
        config,
        f"cat > {config.hub.config_path}.new && install -m 0600 {config.hub.config_path}.new {config.hub.config_path}"
        f" && wg syncconf {config.hub.wg_interface} <({strip_cmd})",
        dry_run=dry_run, input_text=hub_conf,
    )
    _run_remote(
        config,
        f"cat > /etc/nftables.d/{config.nft_table}.nft.new"
        f" && mv /etc/nftables.d/{config.nft_table}.nft.new /etc/nftables.d/{config.nft_table}.nft"
        f" && nft -f /etc/nftables.d/{config.nft_table}.nft",
        dry_run=dry_run, input_text=hub_nft,
    )


def _live_remove_peer(config: render.Config, public_key: str, *, dry_run: bool) -> None:
    """Immediate revocation, independent of the config-file push above:
    remove the peer from the live interface right away so it goes dead even
    if the config push races with something else.
    """
    _run_remote(
        config, f"wg set {config.hub.wg_interface} peer {public_key} remove", dry_run=dry_run,
    )


def cmd_add(config: render.Config, peers_state: Path, name: str, runpod_pod_id: str | None, *, dry_run: bool) -> int:
    peers = render.load_peers(peers_state)
    if any(p.name == name for p in peers):
        print(f"error: a peer named {name!r} already exists — remove it first", file=sys.stderr)
        return 1

    address = render.allocate_pod_address(config, peers)
    pod_keys = render.generate_keypair()  # never written to disk
    peer = render.PodPeer(
        name=name, address=address, public_key=pod_keys.public,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        runpod_pod_id=runpod_pod_id,
    )
    new_peers = [*peers, peer]

    _push_hub_state(config, new_peers, dry_run=dry_run)
    # Only persist public info (name/address/public_key) — the private key
    # was never written anywhere and is discarded once this process exits,
    # except for the copy printed below for the caller to capture.
    if not dry_run:
        render.save_peers(peers_state, new_peers)

    hub_keys = render.load_or_create_keypair(config.keys_dir, "hub")
    onetun_cmds = render.render_onetun_commands(config, hub_keys.public, address)
    wireproxy_conf = render.render_wireproxy_conf(config, hub_keys.public, address)
    kernel_conf = render.render_pod_conf(config, pod_keys, hub_keys.public, address)
    services_env = "\n".join(
        f"{s.name} {s.effective_local_port()} {s.protocol}" for s in config.allow_services()
    )

    bundle = {
        "schema": "lilith/wg-pod-client-bundle@1",
        "name": name,
        "address": address,
        "private_key": pod_keys.private,
        "public_key": pod_keys.public,
        "hub_public_key": hub_keys.public,
        "hub_endpoint": config.hub.public_endpoint,
        "tunnel_subnet": config.tunnel.subnet,
        "kernel_wg_conf": kernel_conf,
        "wireproxy_conf": wireproxy_conf,
        "onetun_commands": onetun_cmds,
        "env": {
            "LILITH_WG_PRIVATE_KEY": pod_keys.private,
            "LILITH_WG_ADDRESS": address,
            "LILITH_WG_HUB_PUBKEY": hub_keys.public,
            "LILITH_WG_HUB_ENDPOINT": config.hub.public_endpoint,
            "LILITH_WG_TUNNEL_SUBNET": config.tunnel.subnet,
            "LILITH_WG_ONETUN_CMDS": "\n".join(onetun_cmds),
            "LILITH_WG_WIREPROXY_CONF": wireproxy_conf,
            "LILITH_WG_KERNEL_CONF": kernel_conf,
            "LILITH_WG_SERVICES": services_env,
        },
        "note": (
            "Inject `env` as Runpod secrets for pod-client.sh (see "
            "docs/ops/lilith-wireguard.md). This bundle is printed once, to "
            "stdout only, and is not stored anywhere by this script — the "
            "private key never touches disk or the repository."
        ),
    }
    print(json.dumps(bundle, indent=2))
    print(f"peer {name!r} added at {address} (dry_run={dry_run})", file=sys.stderr)
    return 0


def cmd_remove(config: render.Config, peers_state: Path, name: str, *, dry_run: bool) -> int:
    peers = render.load_peers(peers_state)
    match = next((p for p in peers if p.name == name), None)
    if match is None:
        print(f"error: no peer named {name!r}", file=sys.stderr)
        return 1

    remaining = [p for p in peers if p.name != name]
    # Revoke live first (fastest path to "dead"), then converge the
    # persisted config + nft ruleset to match.
    _live_remove_peer(config, match.public_key, dry_run=dry_run)
    _push_hub_state(config, remaining, dry_run=dry_run)
    if not dry_run:
        render.save_peers(peers_state, remaining)
    print(f"peer {name!r} removed (dry_run={dry_run})", file=sys.stderr)
    return 0


def cmd_list(config: render.Config, peers_state: Path, *, live: bool, dry_run: bool) -> int:
    peers = render.load_peers(peers_state)
    print(json.dumps([p.to_dict() for p in peers], indent=2, sort_keys=True))
    if live:
        output = _run_remote(config, f"wg show {config.hub.wg_interface} dump", dry_run=dry_run)
        print("--- live `wg show ... dump` (hub) ---", file=sys.stderr)
        print(output, file=sys.stderr)
    return 0


def _op(config: render.Config) -> "slots.OnePassword":
    assert config.slots is not None
    if config.slots.backend == "openbao":
        from openbao import OpenBao
        section = config.slots
        return OpenBao(section.openbao_config, [section.hub_item, section.state_item,
                                              *[section.item(n) for n in range(1, section.count + 1)]])
    return slots.OnePassword(command=config.slots.op_command, vault=config.slots.vault)


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def cmd_slot_status(config: render.Config) -> int:
    views = slots.slot_status(config, _op(config))
    print(json.dumps([v.to_dict() for v in views], indent=2))
    return 0


def cmd_slot_assign(config: render.Config, run: str, *, runpod_pod_id: str | None, slot: int | None,
                    fetch: bool, ignore_allowlist_drift: bool) -> int:
    bundle = slots.assign(
        config, _op(config), run, runpod_pod_id=runpod_pod_id, slot=slot, fetch=fetch,
        ignore_allowlist_drift=ignore_allowlist_drift, warn=_warn,
    )
    print(json.dumps(bundle, indent=2))
    print(f"slot {bundle['slot']} ({bundle['address']}) assigned to {run!r}", file=sys.stderr)
    return 0


def cmd_slot_release(config: render.Config, *, slot: int | None, run: str | None) -> int:
    result = slots.release(config, _op(config), slot=slot, run=run)
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="path to wg-config.toml")
    parser.add_argument("--peers-state", type=Path, default=None, help="peers.json (default: sibling of [keys].dir)")
    parser.add_argument("--dry-run", action="store_true", help="print intended SSH/nft commands, do not execute")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="add one pod peer")
    p_add.add_argument("name")
    p_add.add_argument("--runpod-pod-id", default=None)

    p_remove = sub.add_parser("remove", help="remove one pod peer")
    p_remove.add_argument("name")

    p_list = sub.add_parser("list", help="list persisted (and optionally live) peers")
    p_list.add_argument("--live", action="store_true", help="also query the hub with `wg show ... dump`")

    sub.add_parser("slot-status", help="variant C: assigned / free / rotating per slot")

    p_assign = sub.add_parser("slot-assign", help="variant C: claim a free slot, print the pod bundle")
    p_assign.add_argument("run")
    p_assign.add_argument("--runpod-pod-id", default=None)
    p_assign.add_argument("--slot", type=int, default=None, help="only this slot")
    p_assign.add_argument("--fetch", action="store_true", help="git fetch the talos checkout first")
    p_assign.add_argument("--ignore-allowlist-drift", action="store_true")

    p_release = sub.add_parser("slot-release", help="variant C: disable + rotate a slot (after a run, or to revoke)")
    target = p_release.add_mutually_exclusive_group(required=True)
    target.add_argument("--slot", type=int, default=None)
    target.add_argument("--run", default=None)

    args = parser.parse_args(argv)
    config = render.load_config(args.config)

    if args.command.startswith("slot-"):
        try:
            if args.command == "slot-status":
                return cmd_slot_status(config)
            if args.command == "slot-assign":
                return cmd_slot_assign(
                    config, args.run, runpod_pod_id=args.runpod_pod_id, slot=args.slot,
                    fetch=args.fetch, ignore_allowlist_drift=args.ignore_allowlist_drift,
                )
            return cmd_slot_release(config, slot=args.slot, run=args.run)
        except slots.SlotError as exc:
            print(f"error: {exc}\n  help: {exc.help}", file=sys.stderr)
            return 1
    if config.is_variant_c():
        parser.error(f"{args.command!r} manages an SSH hub (variants A/B); variant C uses slot-assign/slot-release")

    peers_state = args.peers_state or render.default_peers_state(config)

    if args.command == "add":
        return cmd_add(config, peers_state, args.name, args.runpod_pod_id, dry_run=args.dry_run)
    if args.command == "remove":
        return cmd_remove(config, peers_state, args.name, dry_run=args.dry_run)
    if args.command == "list":
        return cmd_list(config, peers_state, live=args.live, dry_run=args.dry_run)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
