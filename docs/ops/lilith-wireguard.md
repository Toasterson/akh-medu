# Runbook — Lilith C6a: pure WireGuard tunnel and scoped identity

> Anima WI `01a0e2a0-5e4d-7fd2-8864-f06460218ec4` ("Lilith C6a — pure WireGuard
> tunnel and scoped identity")
> ADR: [116 — Lilith as Chief of Staff](../ai/decisions/116-lilith-chief-of-staff-guided-rl.md)
> (Decision 3: the full live system, in training too, over a tunnel; Decision 7:
> training surface and Runpod placement)
> Plan: [2026-09-27 Lilith Chief of Staff RL](../ai/plans/2026-09-27-lilith-chief-of-staff-rl-runpod.md)
> (C6a row)
> Scripts: `scripts/lilith/wireguard/` (`wg-config.toml.example`, `render.py`,
> `peer.py`, `pod-client.sh`, `test_render.py`)

**Status of this runbook as written: prepared, not applied.** Nothing in
`scripts/lilith/wireguard/` has been run against archibald or hephaistos. The
operator (or the supervisor, once the operator has provisioned the hub) picks
a variant, fills in `wg-config.toml`, and runs the one-time setup below.

## What this buys

Lilith's Runpod pods (`akh-medu-lilith-serve`, `akh-medu-lilith-train-<run>`)
need the full live akh runtime, Anima, the Agent Board, Seshat, the repo,
fleet status and the OpenIndiana validation VM over a tunnel — without opening
an inbound port at home, without Tailscale/headscale, and without ever handing
a pod anything beyond its own scoped, per-session identity (ADR 116 Decision
3: "Only Lilith's own scoped identity... injected at pod start as Runpod
secrets and expire or are revoked at stop"). This resolves the Runpod
handoff's "localhost:10115 is unreachable" blocker for the illumos validation
VM (`lilith_vm_executor`).

## Variant choice

| | Variant A (recommended) | Variant B |
|---|---|---|
| Hub | archibald (public VPS, root) | hephaistos (home LAN, needs a router port-forward) |
| Gateway | hephaistos dials **out** to the hub, `PersistentKeepalive` | gateway == hub (same host) |
| Home router changes | none | one UDP port-forward |
| Blast radius if the hub is compromised | a VPS with no LAN presence of its own | a box already on the home LAN |

Variant A is recommended because it needs zero home-router changes and keeps
the internet-facing surface off the LAN entirely. Pick it unless archibald is
unavailable. Set `tunnel.variant = "A"` (or `"B"`) in `wg-config.toml`
accordingly — every render function branches on this field.

## One-time hub + gateway setup (operator, Variant A)

These are the commands to run by hand once, on each host. They are **not**
run by any script in this repo — `render.py`/`peer.py` only render text and
(for `peer.py`, non-dry-run) push it over SSH to a hub that's already been
bootstrapped this way.

On **archibald** (the hub):

```sh
apt-get update && apt-get install -y wireguard-tools nftables
mkdir -p /etc/nftables.d
# Scope IPv4 forwarding to the wg0 interface only — do not flip the global
# net.ipv4.ip_forward=1 sysctl on a box that also does other things; a
# per-interface forwarding policy is expressed through the nft rules
# render.py generates (hub.nft), which already default-deny everything that
# isn't iifname wg0. If archibald's kernel needs the sysctl at all for
# forwarding to work, set it and rely on hub.nft (not the sysctl) to scope
# what's actually allowed — see render_hub_nft in scripts/lilith/wireguard/render.py.
sysctl -w net.ipv4.ip_forward=1
echo 'net.ipv4.ip_forward=1' >> /etc/sysctl.d/99-lilith-wg.conf
systemctl enable --now nftables
```

On **hephaistos** (the gateway, Variant A):

```sh
sudo apt-get update && sudo apt-get install -y wireguard-tools nftables
# hephaistos already has wireguard-tools per the C6a brief; nftables may be new.
sudo mkdir -p /etc/nftables.d
sudo sysctl -w net.ipv4.ip_forward=1
echo 'net.ipv4.ip_forward=1' | sudo tee -a /etc/sysctl.d/99-lilith-wg.conf
sudo systemctl enable --now nftables
```

Then, from the operator's own machine (or wherever `scripts/lilith/wireguard/`
is checked out), fill in `wg-config.toml` from the example and do the first
render:

```sh
cd scripts/lilith/wireguard
cp wg-config.toml.example wg-config.toml
# Edit wg-config.toml: real archibald hostname/IP, real hephaistos LAN IP,
# real Agent Board / Seshat / fleet-status IPs — replace every 192.168.1.XXX
# placeholder. Keep [gateway].host as the literal LAN IP, not a hostname
# (see the warning inside the file — this is load-bearing for the nft split).
python3 render.py --config wg-config.toml --out-dir /tmp/lilith-wg-initial
```

This writes `hub.conf`, `hub.nft`, `gateway.conf`, `gateway.nft` (plus a
template pod-client bundle, useful only to sanity-check the format — real pod
peers always come from `peer.py add`, never from this bulk render) and, as a
side effect, mints the hub's and gateway's **persistent** keypairs under
`[keys].dir` (outside the repo; `render.py`/`peer.py` refuse to write inside
the akh-medu checkout — see `test_render.py`'s
`test_keys_dir_inside_repo_is_rejected`).

Install the rendered files:

```sh
# On archibald:
scp /tmp/lilith-wg-initial/hub.conf root@archibald:/etc/wireguard/wg0.conf
scp /tmp/lilith-wg-initial/hub.nft  root@archibald:/etc/nftables.d/lilith_wg.nft
ssh root@archibald 'chmod 600 /etc/wireguard/wg0.conf && nft -f /etc/nftables.d/lilith_wg.nft && systemctl enable --now wg-quick@wg0'

# On hephaistos (Variant A gateway):
scp /tmp/lilith-wg-initial/gateway.conf toasty@hephaistos:/tmp/wg0.conf
scp /tmp/lilith-wg-initial/gateway.nft  toasty@hephaistos:/tmp/lilith_wg.nft
ssh toasty@hephaistos 'sudo install -m 0600 /tmp/wg0.conf /etc/wireguard/wg0.conf && sudo nft -f /tmp/lilith_wg.nft && sudo systemctl enable --now wg-quick@wg0'
```

Bring both interfaces up (`wg-quick@wg0` above does this via systemd; `wg
show wg0` on either host should show the other as a peer with a recent
handshake once hephaistos has dialed out).

**Variant B** differs only in: hub == gateway (run the archibald steps on
hephaistos instead), and add a UDP port-forward on the home router,
`external:51820 -> hephaistos:51820` (adjust `tunnel.listen_port` if a
different external port is used). Everything downstream (peer lifecycle,
allowlist, nftables) is identical.

## Peer lifecycle (the supervisor)

The **supervisor is the architect session (Claude)** per ADR 116 Decision 7
("the architect session... starts and stops her pods"). It owns the peer
lifecycle:

**At pod start**, before the pod needs any allowlisted service:

```sh
python3 scripts/lilith/wireguard/peer.py --config wg-config.toml \
  add lilith-serve-<date> --runpod-pod-id <runpod pod id> > /tmp/lilith-pod-bundle.json
```

This allocates the next free `/32`, mints an ephemeral keypair **that is
never written to disk anywhere** (not in this repo, not in `[keys].dir`, not
on the network volume), pushes the new peer to the hub live (`wg syncconf`,
no disruption to the gateway peer or other pods) and persists the
config-file + nftables update so a hub reboot doesn't lose the peer set.
`peer add` then prints the pod-side bundle to stdout exactly once — read
`.env` out of it and set those as **Runpod secrets** for the pod (never bake
them into the image, never write `/tmp/lilith-pod-bundle.json` anywhere
persistent — delete it after injecting the secrets).

Inside the pod, `pod-client.sh` (see below) reads those secrets and brings
the tunnel up.

**At pod stop**, always, even on a crash/timeout:

```sh
python3 scripts/lilith/wireguard/peer.py --config wg-config.toml remove lilith-serve-<date>
```

This revokes the peer on the hub **immediately** (`wg set wg0 peer <key>
remove`, independent of the config-file push) and then converges the
persisted hub config + nftables to match. A removed peer's `/32` becomes
available for reuse by the next `add`.

`peer.py list [--live]` shows the persisted peer set, and with `--live` cross
-checks it against `wg show wg0 dump` on the hub.

Every `peer.py` invocation supports `--dry-run`, which prints the exact SSH
commands and payload sizes instead of running them — this is how
`test_render.py`'s `PeerCliDryRunTests` exercise `add`/`remove` without a
network, and it is the safe way to review what an invocation *would* do
before running it for real.

## Inside the pod: `pod-client.sh`

Runpod pods normally lack `NET_ADMIN` and `/dev/net/tun`, so the default path
is a **userspace** WireGuard client:

- **onetun** (preferred): per-service TCP/UDP port forwards to
  `127.0.0.1:<port>` on the pod, fully transparent to whatever inside the pod
  connects to that port. `render.py` batches forwards into groups of at most
  7 per `onetun` process (`onetun`'s smoltcp backend caps IP addresses per
  protocol at 7 — see its README) and starts one process per batch.
- **wireproxy** (alternative): a single process exposing the same forwards as
  `[TCPClientTunnel]` blocks (TCP only — UDP services fall back to onetun).
- **kernel WireGuard / wireguard-go**: used automatically when
  `pod-client.sh`'s `detect_kernel_capable` check finds a real `/dev/net/tun`
  **and** confirms `NET_ADMIN` with a harmless dummy-interface probe (a
  device node alone is not proof of the capability on Runpod).

`pod-client.sh --client auto` (the default) picks in that order: kernel,
then onetun, then wireproxy, then **fails closed** — it never silently
proceeds without a tunnel. After starting the client it health-checks every
TCP entry in `LILITH_WG_SERVICES` (name/port/protocol triples, from the
`peer add` bundle) and exits non-zero if any is unreachable, so the pod never
reports itself ready with a broken tunnel.

Required env (all come straight out of the `peer.py add` bundle's `.env`
object — inject them as Runpod secrets):

```
LILITH_WG_PRIVATE_KEY   LILITH_WG_ADDRESS        LILITH_WG_HUB_PUBKEY
LILITH_WG_HUB_ENDPOINT  LILITH_WG_TUNNEL_SUBNET  LILITH_WG_ONETUN_CMDS
LILITH_WG_SERVICES
```

Optional: `LILITH_WG_WIREPROXY_CONF`, `LILITH_WG_KERNEL_CONF` (needed only if
`LILITH_WG_CLIENT` forces those modes, or `auto` falls into them).

**Pre-flight check before first real use:** `pod-client.sh`'s onetun
invocations pass the private key as `--private-key "$LILITH_WG_PRIVATE_KEY"`,
which shell-expands into this process's own argv (visible via `ps` only
*inside the pod's own single-tenant container*, not from outside it). Before
running this against a real pod, check `onetun --help` on the deployed
version for an environment-variable-only path that avoids even that — the
script doesn't guess at an unconfirmed flag name.

## Service allowlist

Everything reachable through the tunnel is **data**, in
`wg-config.toml`'s `[[services]]` array — never hand-added to a config file.
Each entry has `name`, `host`, `port`, `protocol`, `bypass_gateway`, and an
optional `local_port`. `bypass_gateway = true` (Anima MCP, Forgejo) means the
service is public HTTPS and the pod may just reach it directly over the
normal internet — it's listed for completeness but never appears in any
rendered WireGuard peer, onetun forward, or nftables rule.

nftables lives in **one dedicated table**, `table inet lilith_wg` (name
configurable via `[nftables].table`), on both the hub and the gateway:

- **Hub** (`render_hub_nft`): forwards only `wg0 -> wg0` traffic from an
  active pod's `/32` to the gateway's tunnel address — nothing else, ever
  (no NAT to the open internet, no pod-to-pod). Defense in depth on top of
  WireGuard's own cryptokey routing (a pod literally cannot present another
  peer's source address).
- **Gateway** (`render_gateway_nft`): an `input`-chain rule per allowlisted
  service whose `host` is the gateway's own LAN IP (Linux never routes
  same-host traffic through `forward`), and a `forward` + `masquerade` rule
  per allowlisted service on a different LAN host. Everything else arriving
  from the tunnel subnet is dropped.

Both tables use `policy accept` at their hook and end in an explicit
`iifname "wg0" ... drop`, scoped only to tunnel-subnet traffic — so, unlike a
`policy drop` table would, this **never touches unrelated docker/ufw
forwarding** on either host. `test_render.py`'s `NftAllowlistTests` assert
exactly this (no `policy drop`, an explicit deny-rest line, and that
`bypass_gateway` services never appear in generated nft text at all).

## Verification checklist (C6a exit condition)

From a real test pod, once its peer has been added and `pod-client.sh` has
run:

- [ ] Every non-bypass service in the allowlist answers on its
      `127.0.0.1:<local_port>` inside the pod (this is exactly what
      `pod-client.sh`'s health-check loop verifies and fails closed on).
- [ ] Nothing outside the allowlist is reachable — spot-check with `nc -zv
      127.0.0.1 <port>` for a port that was never forwarded (must fail), and
      from the pod's normal network path confirm no other `192.168.1.0/24`
      address is reachable at all (the gateway's `forward`/`input` chains
      only ever match the specific allowlisted `host:port`).
- [ ] `peer.py list --live` shows the test pod's peer on the hub.
- [ ] `peer.py remove <name>` is run, then: `wg show wg0` on the hub no
      longer lists the peer, and every previously-reachable
      `127.0.0.1:<port>` in the pod stops answering (the dead-peer check).
- [ ] The runbook (this file) matches what was actually run — update it if a
      step diverged.

## Rollback

Tearing down the tunnel touches only this feature's own state:

```sh
# On the hub and/or gateway:
sudo systemctl disable --now wg-quick@wg0
sudo nft delete table inet lilith_wg     # flushes only the dedicated table —
                                          # docker/ufw's own tables are untouched
```

To remove a single pod's access without touching the tunnel as a whole, use
`peer.py remove <name>` (see above) — that is the normal, non-destructive
path and should be preferred over any table-level rollback.

## Files in this slice

| File | Purpose |
|---|---|
| `scripts/lilith/wireguard/wg-config.toml.example` | Template: tunnel subnet, hub/gateway endpoints, service allowlist as data. Copy to `wg-config.toml` (gitignored by convention) and fill in real values — never commit a filled-in copy. |
| `scripts/lilith/wireguard/render.py` | Pure rendering: wg-quick configs, nftables rulesets, onetun/wireproxy pod-client artifacts. Generates/caches the hub and gateway's persistent keypairs under `[keys].dir` (asserted outside the repo). No network I/O. |
| `scripts/lilith/wireguard/peer.py` | `add` / `remove` / `list` pod peers on the hub over SSH (`wg set` live + persisted config + nft reload). Never writes a pod private key to disk; prints the pod-side bundle to stdout once. `--dry-run` everywhere. |
| `scripts/lilith/wireguard/pod-client.sh` | Runs inside the pod: picks kernel/onetun/wireproxy, brings the tunnel up, health-checks the allowlist, fails closed. |
| `scripts/lilith/wireguard/test_render.py` | Rendering determinism, allowlist→nft correctness (including a regression test for the hostname-vs-IP self/remote bug caught while building this), peer `/32` allocation/removal, and "no private key material ends up under the repo" checks. Runs `shellcheck` on `pod-client.sh` when available. |
