"""Variant C peer slots for the Lilith WireGuard hub on archibald.

The hub (talos `clusters/archibald/lilith-wg`) has a fixed pool of peer
slots. Slot public keys are GitOps state in talos `peers.conf`; slot private
keys live ONLY in 1Password (items `lilith-wg-slot-<n>`). This module is the
supervisor side:

- `status`: derive each slot's state from three sources — the merged talos
  `peers.conf` (via `git show <ref>:<path>`), the 1Password `public_key`,
  and the local lease records.
- `assign`: claim a free slot (atomic lease file), read its private key from
  1Password at call time, verify it matches the merged public key, and
  return the pod-client bundle (to stdout only; never to disk).
- `release`: rotate the slot's keypair in 1Password (piped JSON, the key
  never appears in argv), optionally write the new PUBLIC key into a talos
  work tree, and drop the lease. The slot then reads `pending-rotation`
  until the talos change is merged.

Nothing here writes private key material to disk. External commands (op,
git) run through an injectable runner so tests never touch a vault or repo.

Runbook: docs/ops/lilith-wireguard.md; hub runbook: talos
clusters/archibald/lilith-wg/README.md; ADR 116 Decision 3.
"""

from __future__ import annotations

import enum
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import render

Runner = Callable[..., subprocess.CompletedProcess]


class SlotError(RuntimeError):
    """An error with a concrete next step for the caller."""

    def __init__(self, message: str, *, help: str) -> None:  # noqa: A002 (mirrors miette's help)
        super().__init__(message)
        self.help = help


class SlotState(str, enum.Enum):
    FREE = "free"
    LEASED = "leased"
    PENDING_ROTATION = "pending-rotation"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Lease:
    slot: int
    run: str
    runpod_pod_id: str | None
    assigned_utc: str
    public_key: str

    def to_dict(self) -> dict:
        return {
            "slot": self.slot, "run": self.run, "runpod_pod_id": self.runpod_pod_id,
            "assigned_utc": self.assigned_utc, "public_key": self.public_key,
        }

    @staticmethod
    def from_dict(d: dict) -> "Lease":
        return Lease(int(d["slot"]), str(d["run"]), d.get("runpod_pod_id"), str(d["assigned_utc"]), str(d["public_key"]))


@dataclass(frozen=True)
class SlotView:
    slot: int
    item: str
    address: str
    state: SlotState
    detail: str
    lease: Lease | None

    def to_dict(self) -> dict:
        return {
            "slot": self.slot, "item": self.item, "address": self.address, "state": self.state.value,
            "detail": self.detail, "lease": self.lease.to_dict() if self.lease else None,
        }


# --------------------------------------------------------------------------
# 1Password (service account) — values captured in memory only
# --------------------------------------------------------------------------


@dataclass
class OnePassword:
    command: str
    vault: str
    runner: Runner = subprocess.run

    def _run(self, args: Sequence[str], *, input_text: str | None = None) -> str:
        try:
            result = self.runner(
                [self.command, *args], input=input_text, capture_output=True, text=True, timeout=60,
            )
        except FileNotFoundError as exc:
            raise SlotError(
                f"1Password CLI wrapper not found: {self.command}",
                help="set [slots].op_command to the op-service-account wrapper (see the onepassword-service-account skill)",
            ) from exc
        if result.returncode != 0:
            # stderr from op never contains field values; keep it short anyway.
            raise SlotError(
                f"1Password command `{args[0]} {args[1] if len(args) > 1 else ''}` failed: {result.stderr.strip()[:300]}",
                help="check that the service account can read (and, for rotation, write) the "
                f"{self.vault} vault; on a permission error ask the operator to rotate the item by hand",
            )
        return result.stdout

    def read(self, item: str, field: str) -> str:
        return self._run(["read", f"op://{self.vault}/{item}/{field}"]).strip()

    def rotate_keypair(self, item: str, keys: render.KeyPair) -> None:
        """Replace private_key/public_key on an existing item via piped JSON."""
        doc = json.loads(self._run(["item", "get", item, "--vault", self.vault, "--format", "json"]))
        seen = set()
        for f in doc.get("fields", []):
            if f.get("label") == "private_key":
                f["value"] = keys.private
                seen.add("private_key")
            elif f.get("label") == "public_key":
                f["value"] = keys.public
                seen.add("public_key")
        if seen != {"private_key", "public_key"}:
            raise SlotError(
                f"1Password item {item} lacks a private_key/public_key field",
                help="recreate the item with fields private_key (concealed), public_key, address",
            )
        self._run(["item", "edit", item, "--vault", self.vault, "--format", "json"], input_text=json.dumps(doc))


# --------------------------------------------------------------------------
# talos GitOps state (what is merged, not the work tree)
# --------------------------------------------------------------------------


def talos_show(slots: render.SlotsSection, path: str, *, runner: Runner = subprocess.run, fetch: bool = False) -> str:
    if slots.talos_repo is None:
        raise SlotError(
            "no talos checkout configured",
            help="set [slots].talos_repo to a clone of Toasterson/talos (only `git show` is used)",
        )
    repo = str(slots.talos_repo)
    if fetch:
        runner(["git", "-C", repo, "fetch", "--quiet", "origin"], capture_output=True, text=True, timeout=60)
    result = runner(["git", "-C", repo, "show", f"{slots.talos_ref}:{path}"], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise SlotError(
            f"cannot read {slots.talos_ref}:{path} from {repo}: {result.stderr.strip()[:200]}",
            help="run with --fetch, or check [slots].talos_ref / the path (the hub PR may not be merged yet)",
        )
    return result.stdout


# --------------------------------------------------------------------------
# Leases (public data only)
# --------------------------------------------------------------------------


def _lease_path(slots: render.SlotsSection, n: int) -> Path:
    return slots.lease_dir / f"slot-{n}.json"


def load_lease(slots: render.SlotsSection, n: int) -> Lease | None:
    path = _lease_path(slots, n)
    if not path.exists():
        return None
    return Lease.from_dict(json.loads(path.read_text()))


def _claim_lease(slots: render.SlotsSection, lease: Lease) -> bool:
    """Atomically create the lease file; False if the slot is already leased."""
    slots.lease_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        fd = os.open(_lease_path(slots, lease.slot), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(lease.to_dict(), indent=2, sort_keys=True) + "\n")
    return True


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


def slot_status(config: render.Config, op: OnePassword, *, runner: Runner = subprocess.run, fetch: bool = False) -> list[SlotView]:
    slots = _slots(config)
    committed = render.parse_talos_peers(talos_show(slots, slots.talos_peers_path, runner=runner, fetch=fetch))
    views = []
    for n in range(1, slots.count + 1):
        item, address = slots.item(n), slots.address(n)
        lease = load_lease(slots, n)
        entry = committed.get(f"slot-{n}")
        if entry is None:
            state, detail = SlotState.UNKNOWN, f"slot-{n} is not in talos {slots.talos_ref} peers.conf"
        elif entry.address != address:
            state, detail = SlotState.UNKNOWN, f"talos has {entry.address}, config expects {address}"
        elif not entry.enabled:
            state, detail = SlotState.DISABLED, "off in talos peers.conf"
        elif lease is not None:
            state, detail = SlotState.LEASED, f"run {lease.run} since {lease.assigned_utc}"
        elif op.read(item, "public_key") != entry.public_key:
            state, detail = SlotState.PENDING_ROTATION, (
                f"1Password key differs from talos {slots.talos_ref}; merge the rotation PR first"
            )
        else:
            state, detail = SlotState.FREE, "1Password key matches the merged hub peer"
        views.append(SlotView(n, item, address, state, detail, lease))
    return views


def _slots(config: render.Config) -> render.SlotsSection:
    if not config.is_variant_c() or config.slots is None:
        raise SlotError("slot commands need a variant C config", help="use wg-config.toml.example (variant C)")
    return config.slots


# --------------------------------------------------------------------------
# Assign / release
# --------------------------------------------------------------------------


def assign(
    config: render.Config, op: OnePassword, run: str, *, runpod_pod_id: str | None = None,
    slot: int | None = None, runner: Runner = subprocess.run, fetch: bool = False,
    ignore_allowlist_drift: bool = False, warn: Callable[[str], None] = lambda _m: None,
) -> dict:
    """Claim a slot and return the pod-client bundle (contains the private key)."""
    slots = _slots(config)
    drift = render.allowlist_drift(
        config, render.parse_talos_allowlist(talos_show(slots, slots.talos_allowlist_path, runner=runner, fetch=fetch))
    )
    if drift and not ignore_allowlist_drift:
        raise SlotError(
            "pod forwards and the hub allowlist disagree: " + "; ".join(drift),
            help="align [[services]] in wg-config.toml with talos lilith-wg/allowlist.conf "
            "(or pass --ignore-allowlist-drift; mismatched forwards will fail the pod health check)",
        )
    views = slot_status(config, op, runner=runner)
    candidates = [v for v in views if v.state is SlotState.FREE and (slot is None or v.slot == slot)]
    if not candidates:
        wanted = f"slot {slot}" if slot is not None else "any slot"
        summary = ", ".join(f"{v.slot}={v.state.value}" for v in views)
        raise SlotError(
            f"no free slot for {wanted} ({summary})",
            help="release a finished run (`peer.py slot-release`), merge pending rotation PRs in talos, "
            "or add slots to talos peers.conf + 1Password",
        )
    view = candidates[0]

    private = op.read(view.item, "private_key")
    public = render.derive_public_key(private)
    committed = render.parse_talos_peers(talos_show(slots, slots.talos_peers_path, runner=runner))
    if public != committed[f"slot-{view.slot}"].public_key:
        raise SlotError(
            f"slot {view.slot}: the 1Password private key does not match the merged hub peer",
            help="the item was rotated without merging talos peers.conf (or edited by hand); "
            "merge the rotation or run `peer.py slot-release --slot N` to rotate cleanly",
        )
    hub_public = op.read(slots.hub_item, "public_key")
    if config.hub.public_key and config.hub.public_key != hub_public:
        warn(f"[hub].public_key in wg-config.toml is stale; using 1Password {slots.hub_item} ({hub_public})")

    lease = Lease(view.slot, run, runpod_pod_id, datetime.now(timezone.utc).isoformat(timespec="seconds"), public)
    if not _claim_lease(slots, lease):
        raise SlotError(f"slot {view.slot} was leased concurrently", help="retry: the next free slot will be chosen")

    keys = render.KeyPair(private=private, public=public)
    address = view.address
    onetun = render.render_onetun_commands(config, hub_public, address)
    wireproxy = render.render_wireproxy_conf(config, hub_public, address)
    kernel = render.render_pod_conf(config, keys, hub_public, address)
    hosts = render.render_hosts(config).strip()
    return {
        "schema": "lilith/wg-pod-client-bundle@1",
        "variant": "C",
        "name": run,
        "slot": view.slot,
        "address": address,
        "private_key": private,
        "public_key": public,
        "hub_public_key": hub_public,
        "hub_endpoint": config.hub.public_endpoint,
        "tunnel_subnet": config.tunnel.subnet,
        "kernel_wg_conf": kernel,
        "wireproxy_conf": wireproxy,
        "onetun_commands": onetun,
        "env": {
            "LILITH_WG_PRIVATE_KEY": private,
            "LILITH_WG_ADDRESS": address,
            "LILITH_WG_HUB_PUBKEY": hub_public,
            "LILITH_WG_HUB_ENDPOINT": config.hub.public_endpoint,
            "LILITH_WG_TUNNEL_SUBNET": config.tunnel.subnet,
            "LILITH_WG_ONETUN_CMDS": "\n".join(onetun),
            "LILITH_WG_WIREPROXY_CONF": wireproxy,
            "LILITH_WG_KERNEL_CONF": kernel,
            "LILITH_WG_SERVICES": render.services_env(config),
            "LILITH_WG_HOSTS": hosts,
        },
        "note": (
            "Inject `env` as Runpod secrets for pod-client.sh. Printed once to stdout; the "
            "private key is not stored anywhere by peer.py. Run `peer.py slot-release` after the run."
        ),
    }


def _find_slot(slots: render.SlotsSection, slot: int | None, run: str | None) -> int:
    if slot is not None:
        slots.address(slot)  # range check
        return slot
    for n in range(1, slots.count + 1):
        lease = load_lease(slots, n)
        if lease is not None and lease.run == run:
            return n
    raise SlotError(f"no leased slot for run {run!r}", help="pass --slot N, or `peer.py slot-status` to list leases")


def replace_peer_key(text: str, slot_name: str, new_public_key: str) -> str:
    """Rewrite one slot's public key in talos peers.conf, keeping the layout."""
    out, hit = [], False
    for line in text.splitlines(keepends=True):
        cols = line.split()
        if cols and cols[0] == slot_name and not line.lstrip().startswith("#"):
            line = line.replace(cols[1], new_public_key, 1)
            hit = True
        out.append(line)
    if not hit:
        raise SlotError(f"{slot_name} not found in peers.conf", help="add the slot line to talos peers.conf first")
    return "".join(out)


def release(
    config: render.Config, op: OnePassword, *, slot: int | None = None, run: str | None = None,
    rotate: bool = True, write_talos: bool = False,
) -> dict:
    """Rotate a finished slot's key in 1Password and drop its lease."""
    slots = _slots(config)
    n = _find_slot(slots, slot, run)
    lease = load_lease(slots, n)
    result: dict = {"slot": n, "item": slots.item(n), "released_run": lease.run if lease else None}
    if rotate:
        keys = render.generate_keypair()
        op.rotate_keypair(slots.item(n), keys)  # before dropping the lease: a failure keeps the slot leased
        line = f"slot-{n}  {keys.public}  {slots.address(n)}  on"
        result |= {"rotated": True, "new_public_key": keys.public, "peers_conf_line": line}
        if write_talos:
            if slots.talos_repo is None:
                raise SlotError("--write-talos needs [slots].talos_repo", help="set it, or apply peers_conf_line by hand")
            path = slots.talos_repo / slots.talos_peers_path
            path.write_text(replace_peer_key(path.read_text(), f"slot-{n}", keys.public))
            result["talos_file_written"] = str(path)
        result["next"] = (
            "commit the new public key in talos clusters/archibald/lilith-wg/peers.conf, open a PR and "
            "merge it; Flux applies it and the hub drops the old key. The slot reads pending-rotation "
            "until then."
        )
    else:
        result |= {"rotated": False, "next": "key NOT rotated: the previous pod's key stays valid on the hub"}
    lease_path = _lease_path(slots, n)
    if lease_path.exists():
        lease_path.unlink()
    return result
