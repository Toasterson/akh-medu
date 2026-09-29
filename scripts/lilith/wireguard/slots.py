"""Variant C peer slots for the Lilith WireGuard hub on archibald.

The hub (talos `clusters/archibald/lilith-wg`) has a fixed pool of peer
slots. Two kinds of 1Password items (vault Homelab) hold the state:

- `lilith-wg-slot-<n>`: the slot keypair (`private_key`, `public_key`,
  `address`). Only this supervisor reads the private key, at call time.
- `lilith-wg-slots`: PUBLIC state per slot (`slotN_public_key`,
  `slotN_enabled`, `slotN_run`, `slotN_assigned_at`, `slotN_rotated_at`).
  ESO renders the hub's peer set from it every minute; the hub hot-applies
  the diff with `wg set`. No git change per run.

Commands:
- `status`: assigned / free / rotating per slot.
- `assign`: pick a free slot, read its private key, verify it derives to the
  published public key, set `slotN_enabled=true`, return the pod bundle (to
  stdout only; never to disk or logs).
- `release`: fresh keypair; publish the new public key with
  `slotN_enabled=false` first (this is what kills the old key on the hub),
  then store the keypair in the slot item. All 1Password writes are piped
  JSON — key material never appears in argv.

External commands (op, git) run through an injectable runner so tests never
touch a vault or a repository. A local flock serialises concurrent
assign/release calls on one supervisor host.

Runbook: docs/ops/lilith-wireguard.md; hub runbook: talos
clusters/archibald/lilith-wg/README.md; ADR 116 Decision 3.
"""

from __future__ import annotations

import contextlib
import enum
import fcntl
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterator, Sequence

import render

Runner = Callable[..., subprocess.CompletedProcess]
Clock = Callable[[], datetime]
UNSET = "-"  # 1Password drops empty fields; "-" marks "no value"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(ts: datetime) -> str:
    return ts.isoformat(timespec="seconds")


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


class SlotError(RuntimeError):
    """An error with a concrete next step for the caller."""

    def __init__(self, message: str, *, help: str) -> None:  # noqa: A002 (mirrors miette's help)
        super().__init__(message)
        self.help = help


class SlotState(str, enum.Enum):
    FREE = "free"
    ASSIGNED = "assigned"
    ROTATING = "rotating"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PublishedSlot:
    """One slot's public state from the `lilith-wg-slots` item."""

    public_key: str
    enabled: bool
    run: str | None
    assigned_at: str | None
    rotated_at: str | None


@dataclass(frozen=True)
class SlotView:
    slot: int
    item: str
    address: str
    state: SlotState
    detail: str
    run: str | None
    assigned_at: str | None
    rotated_at: str | None

    def to_dict(self) -> dict:
        return {
            "slot": self.slot, "item": self.item, "address": self.address, "state": self.state.value,
            "detail": self.detail, "run": self.run, "assigned_at": self.assigned_at, "rotated_at": self.rotated_at,
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
            # op's stderr never echoes field values; keep it short anyway.
            raise SlotError(
                f"1Password `{' '.join(args[:2])}` failed: {result.stderr.strip()[:300]}",
                help=f"check that the service account can read (and, for assign/release, write) the {self.vault} "
                "vault; on a permission error, ask the operator to edit the items by hand (runbook: slot lifecycle)",
            )
        return result.stdout

    def read(self, item: str, field: str) -> str:
        return self._run(["read", f"op://{self.vault}/{item}/{field}"]).strip()

    def fields(self, item: str) -> dict[str, str]:
        doc = json.loads(self._run(["item", "get", item, "--vault", self.vault, "--format", "json"]))
        return {f["label"]: f.get("value", "") for f in doc.get("fields", []) if "label" in f}

    def set_fields(self, item: str, values: dict[str, str]) -> None:
        """Update existing fields via piped JSON (never argv)."""
        doc = json.loads(self._run(["item", "get", item, "--vault", self.vault, "--format", "json"]))
        missing = set(values)
        for f in doc.get("fields", []):
            if f.get("label") in values:
                f["value"] = values[f["label"]]
                missing.discard(f["label"])
        if missing:
            raise SlotError(
                f"1Password item {item} lacks field(s) {sorted(missing)}",
                help="recreate the item with the documented fields (runbook: slot lifecycle)",
            )
        self._run(["item", "edit", item, "--vault", self.vault, "--format", "json"], input_text=json.dumps(doc))


# --------------------------------------------------------------------------
# Published state
# --------------------------------------------------------------------------


def _field(n: int, name: str) -> str:
    return f"slot{n}_{name}"


def _opt(value: str | None) -> str | None:
    return None if value in (None, "", UNSET) else value


def published(fields: dict[str, str], n: int) -> PublishedSlot | None:
    pub, enabled = fields.get(_field(n, "public_key")), fields.get(_field(n, "enabled"))
    if not pub or enabled not in ("true", "false"):
        return None
    return PublishedSlot(
        public_key=pub, enabled=enabled == "true", run=_opt(fields.get(_field(n, "run"))),
        assigned_at=_opt(fields.get(_field(n, "assigned_at"))), rotated_at=_opt(fields.get(_field(n, "rotated_at"))),
    )


def _slots(config: render.Config) -> render.SlotsSection:
    if not config.is_variant_c() or config.slots is None:
        raise SlotError("slot commands need a variant C config", help="use wg-config.toml.example (variant C)")
    return config.slots


def slot_status(config: render.Config, op: OnePassword, *, now: Clock = _utcnow) -> list[SlotView]:
    slots = _slots(config)
    state = op.fields(slots.state_item)
    window = timedelta(seconds=slots.propagation_seconds)
    views = []
    for n in range(1, slots.count + 1):
        item, address = slots.item(n), slots.address(n)
        pub = published(state, n)
        if pub is None:
            views.append(SlotView(n, item, address, SlotState.UNKNOWN,
                                  f"{slots.state_item} lacks valid slot{n}_public_key/slot{n}_enabled", None, None, None))
            continue
        item_pub = op.read(item, "public_key")
        if item_pub != pub.public_key:
            state_, detail = SlotState.ROTATING, (
                f"{item} and {slots.state_item} disagree (interrupted release); re-run slot-release --slot {n}"
            )
        elif pub.enabled:
            state_, detail = SlotState.ASSIGNED, f"run {pub.run or '?'} since {pub.assigned_at or '?'}"
        elif (released := _parse_ts(pub.rotated_at)) is not None and now() - released < window:
            state_, detail = SlotState.ROTATING, (
                f"released {pub.rotated_at}; hub has dropped the old key by {_iso(released + window)}"
            )
        elif pub.rotated_at and _parse_ts(pub.rotated_at) is None:
            state_, detail = SlotState.UNKNOWN, f"slot{n}_rotated_at is not an ISO timestamp: {pub.rotated_at!r}"
        else:
            state_, detail = SlotState.FREE, "disabled on the hub, key published"
        views.append(SlotView(n, item, address, state_, detail, pub.run, pub.assigned_at, pub.rotated_at))
    return views


# --------------------------------------------------------------------------
# talos allowlist (what is merged, not the work tree)
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


@contextlib.contextmanager
def _locked(slots: render.SlotsSection) -> Iterator[None]:
    slots.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(slots.state_dir / "slots.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# --------------------------------------------------------------------------
# Assign / release
# --------------------------------------------------------------------------


def assign(
    config: render.Config, op: OnePassword, run: str, *, runpod_pod_id: str | None = None,
    slot: int | None = None, runner: Runner = subprocess.run, fetch: bool = False,
    ignore_allowlist_drift: bool = False, warn: Callable[[str], None] = lambda _m: None, now: Clock = _utcnow,
) -> dict:
    """Enable a free slot and return the pod-client bundle (contains the private key)."""
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
    with _locked(slots):
        views = slot_status(config, op, now=now)
        candidates = [v for v in views if v.state is SlotState.FREE and (slot is None or v.slot == slot)]
        if not candidates:
            wanted = f"slot {slot}" if slot is not None else "any slot"
            summary = ", ".join(f"{v.slot}={v.state.value}" for v in views)
            raise SlotError(
                f"no free slot for {wanted} ({summary})",
                help="release a finished run (`peer.py slot-release --run RUN`), wait for rotating slots "
                f"({slots.propagation_seconds}s after release), or add slots (talos template + 1Password)",
            )
        view = candidates[0]
        private = op.read(view.item, "private_key")
        public = render.derive_public_key(private)
        pub = published(op.fields(slots.state_item), view.slot)
        if pub is None or public != pub.public_key:
            raise SlotError(
                f"slot {view.slot}: the private key in {view.item} does not derive to the published public key",
                help=f"the item was edited by hand; run `peer.py slot-release --slot {view.slot}` to rotate cleanly",
            )
        hub_public = op.read(slots.hub_item, "public_key")
        if config.hub.public_key and config.hub.public_key != hub_public:
            warn(f"[hub].public_key in wg-config.toml is stale; using 1Password {slots.hub_item} ({hub_public})")
        assigned_at = _iso(now())
        op.set_fields(slots.state_item, {
            _field(view.slot, "enabled"): "true",
            _field(view.slot, "run"): run,
            _field(view.slot, "assigned_at"): assigned_at,
        })

    keys = render.KeyPair(private=private, public=public)
    address = view.address
    onetun = render.render_onetun_commands(config, hub_public, address)
    wireproxy = render.render_wireproxy_conf(config, hub_public, address)
    kernel = render.render_pod_conf(config, keys, hub_public, address)
    # The hub applies the enable after propagation; let the pod's health
    # check wait that long (5 s connect timeout + 1 s sleep per attempt).
    retries = max(6, slots.propagation_seconds // 5)
    return {
        "schema": "lilith/wg-pod-client-bundle@1",
        "variant": "C",
        "name": run,
        "runpod_pod_id": runpod_pod_id,
        "slot": view.slot,
        "address": address,
        "assigned_at": assigned_at,
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
            "LILITH_WG_HOSTS": render.render_hosts(config).strip(),
            "LILITH_WG_HEALTHCHECK_RETRIES": str(retries),
        },
        "note": (
            "Inject `env` as Runpod secrets for pod-client.sh. Printed once to stdout; the private key "
            f"is not stored anywhere by peer.py. The hub enables the slot within ~{slots.propagation_seconds}s. "
            "Run `peer.py slot-release --run <run>` after the run."
        ),
    }


def _find_slot(config: render.Config, op: OnePassword, slot: int | None, run: str | None) -> int:
    slots = _slots(config)
    if slot is not None:
        slots.address(slot)  # range check
        return slot
    state = op.fields(slots.state_item)
    for n in range(1, slots.count + 1):
        pub = published(state, n)
        if pub is not None and pub.enabled and pub.run == run:
            return n
    raise SlotError(f"no assigned slot for run {run!r}", help="pass --slot N, or `peer.py slot-status` to list runs")


def release(config: render.Config, op: OnePassword, *, slot: int | None = None, run: str | None = None,
            now: Clock = _utcnow) -> dict:
    """Rotate a slot: publish a fresh public key disabled, then store the keypair."""
    slots = _slots(config)
    with _locked(slots):
        n = _find_slot(config, op, slot, run)
        before = published(op.fields(slots.state_item), n)
        # Observe both documents before publishing revocation. OpenBao CAS must
        # not lazily adopt a competing writer's new keypair version afterward.
        op.fields(slots.item(n))
        keys = render.generate_keypair()
        rotated_at = _iso(now())
        # 1) Publish first: the hub drops the old key on its next pass after
        #    propagation, regardless of what happens next.
        op.set_fields(slots.state_item, {
            _field(n, "public_key"): keys.public,
            _field(n, "enabled"): "false",
            _field(n, "run"): UNSET,
            _field(n, "assigned_at"): UNSET,
            _field(n, "rotated_at"): rotated_at,
        })
        # 2) Store the keypair. A failure here leaves the slot `rotating`
        #    (items disagree) and disabled; re-running release repairs it.
        try:
            op.set_fields(slots.item(n), {"private_key": keys.private, "public_key": keys.public})
        except SlotError as exc:
            raise SlotError(
                f"slot {n}: published the new key disabled, but storing the keypair failed: {exc}",
                help=f"the old key is revoked anyway; re-run `peer.py slot-release --slot {n}` to finish",
            ) from exc
    return {
        "slot": n,
        "item": slots.item(n),
        "released_run": before.run if before else None,
        "new_public_key": keys.public,
        "rotated_at": rotated_at,
        "old_key_dead_by": _iso(now() + timedelta(seconds=slots.propagation_seconds)),
    }
