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

**Status (2026-09-28): Variant C chosen** — the hub is the Flux-managed
`wg-lilith` pod on archibald's k0s cluster, defined in the talos repo
(GitHub `Toasterson/talos`, `clusters/archibald/lilith-wg`, PR
[#51](https://github.com/Toasterson/talos/pull/51)). Operator instruction:
"archibald uses talos repo to configure k0s — use that to set up the
WireGuard endpoint". Variants A/B below (SSH-managed hub + hephaistos
gateway) are **superseded** and kept for the record. Nothing is live until
the talos PR merges and the operator runs `k0sctl apply` (below). Slot
assignment and rotation afterwards are 1Password writes only; no git change
is needed per run.

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

| | **Variant C (chosen)** | Variant A (superseded) | Variant B (superseded) |
|---|---|---|---|
| Hub | `wg-lilith` pod on archibald k0s (hostNetwork, Flux, talos repo) | archibald host, wg-quick over SSH | hephaistos, router port-forward |
| Path to the LAN | archibald's existing `wg-home` (masquerade) | hephaistos gateway dials out | gateway == hub |
| Home router changes | none (its 3 WireGuard credentials are all allocated) | none | one UDP port-forward |
| Who changes the hub | talos PR for the hub itself and its allowlist; the peer set is 1Password state, hot-reloaded via ESO (no runtime kubectl/SSH) | `peer.py` over SSH | `peer.py` over SSH |
| Pod identity | fixed pool of 4 slots, private keys only in 1Password, rotated after every run | fresh keypair per pod | fresh keypair per pod |

Variant C needs no new host: archibald already reaches the LAN through
`wg-home` (LAN to archibald times out, so a hephaistos gateway could not be
dialled). There is no runtime kubectl access, so peers cannot be pushed to the
hub per pod. Instead there is a fixed slot pool whose public state lives in
1Password. ESO delivers it to the hub every minute, and the hub hot-applies
changes. Assign and release are therefore 1Password writes, with no git change
per run.

## Variant C: archibald k0s hub (talos repo)

The hub side is **talos** `clusters/archibald/lilith-wg/` — its README is
the hub runbook (architecture, allowlist, rotation, verification, rollback).
Summary:

| | |
|---|---|
| Endpoint | `95.216.66.53:51822` (`archibald.solstice-ci.org`), udp |
| Hub | `wg-lilith` `10.201.0.1/24`, pubkey `L4EYWLhJwTI8JKLtikrdLw2+5iRaSI2xfT7N47+Pdio=` |
| Slots | `slot-1..4` = `10.201.0.11..14/32`; keypairs in 1Password Homelab `lilith-wg-slot-<n>` (`private_key`, `public_key`, `address`) |
| Peer set | 1Password `lilith-wg-slots` (public only: `slotN_public_key`, `slotN_enabled`, `slotN_run`, `slotN_assigned_at`, `slotN_rotated_at`) → ESO `lilith-wg-peers` (1 min) → hub `wg set` diff (10 s), about 2 min end to end |
| Hub key | 1Password `lilith-wg-hub`, delivered by ESO |
| Filter | dedicated `inet lilith_wg` table; conntrack-original-destination allowlist; masquerade into `wg-home` only |
| Image | `procustodibus/wireguard@sha256:e20f6b4b6451fc14b78fb92440d2fbc7d1be2ec603bb51ac72919bdaacc79ebd` |

**Allowlist** (talos `allowlist.conf`; mirrored by the enabled `[[services]]`
in `wg-config.toml.example`, and `slot-assign` refuses to proceed if they
drift apart):

| Entry | Target | Pod forward | Source |
|---|---|---|---|
| `lab-gateway-http` | `192.168.1.12:80` | `127.0.0.1:80`, `/etc/hosts`: `seshat.lab.home board.lab.home` | talos `akh-medu/httproute.yaml` (seshd on the internal gateway HTTP listener), `anima-runners/runners.yaml` + `akh-platform/coordination-directory.json` (`http://board.lab.home`) |
| `akh-sessions-home` | `10.0.1.2:18240` | `127.0.0.1:18240` | talos `akh-platform/README.md` "Home placement and bidirectional relay" (relay reverse forward to home `sessions:8080`) |
| `akh-sandbox-akhsbxd` | `10.0.1.2:8242` | — (off) | same README; off because it would bypass session tool grants |
| `openindiana-vm-executor` | `192.168.1.114:10115` | — (off, placeholder) | future hephaistos VM; port unconfirmed |

Anima MCP, Forgejo and the public ACP endpoint
(`https://anima.wegmueller.it/akh`) are public and **not** tunnelled.
Seshat and the Board sit on the internal gateway's plain-HTTP `:80`
listener; `:443` there carries only unrelated TLS listeners, so it is not
allowlisted.

### One-time operator steps

1. Merge the talos PR. Flux creates the namespace, both ExternalSecrets and
   the hub Deployment. All slots start disabled, so the hub has 0 peers.
2. From a clean talos `main`, run `k0sctl apply --config k0sctl/archibald.yaml`.
   This opens udp/51822 in archibald's `fw_public` perimeter table. Until then
   the hub runs, but pods cannot reach it.
3. Supervisor host: copy `scripts/lilith/wireguard/wg-config.toml.example` to
   `wg-config.toml`. Set `[slots].talos_repo` to a talos clone; it is used only
   for the allowlist parity check, via `git show origin/main:…`. Confirm that
   `op_command` points to the op-service-account wrapper. That service account
   needs write access to the `lilith-wg-slots` and `lilith-wg-slot-<n>` items.

### Slot lifecycle (supervisor)

```sh
cd scripts/lilith/wireguard
python3 peer.py --config wg-config.toml slot-status                 # assigned / free / rotating per slot
python3 peer.py --config wg-config.toml slot-assign lilith-serve-<date> \
    --runpod-pod-id <id> --fetch                                    # bundle JSON on stdout: inject .env as Runpod secrets
# ... run ...
python3 peer.py --config wg-config.toml slot-release --run lilith-serve-<date>
```

- **`slot-assign`**
  - Checks allowlist parity against talos `origin/main`.
  - Picks a **free** slot: not enabled, released more than
    `propagation_seconds` (180) ago, and its keypair item agrees with
    `lilith-wg-slots`.
  - Reads the slot's private key from 1Password at call time, verifies that
    it derives to the published public key, and sets `slotN_enabled=true`
    (plus `slotN_run` and `slotN_assigned_at`).
  - The hub adds the peer within about 2 min. The bundle sets
    `LILITH_WG_HEALTHCHECK_RETRIES` so `pod-client.sh` waits out that delay.
  - The bundle (`lilith/wg-pod-client-bundle@1`) goes to stdout once. Nothing
    secret is written to disk, logs or argv. Never capture the bundle into a
    file; pipe it straight into the Runpod secret creation.
- **`slot-release`** runs after every run, and is also the **emergency
  revoke**. It generates a fresh keypair in memory, then:
  1. It writes the new public key to `lilith-wg-slots` with
     `slotN_enabled=false` and `slotN_rotated_at`. This write kills the old
     key: the hub removes it within about 2 min without restarting the
     interface, and the other slots are untouched.
  2. It stores the keypair in `lilith-wg-slot-N`.

  Both writes are piped JSON, so no key appears in argv. The slot then reads
  **rotating** until the propagation window has passed. If step 2 fails, the
  old key is still revoked, and the slot stays **rotating**, because the two
  items disagree, until `slot-release --slot N` is re-run.
- **`slot-status`** lists each slot's state, run and timestamps.
- A local flock serialises concurrent assign/release on one supervisor host.
  Run only one supervisor host at a time: 1Password has no compare-and-swap.
- On failure, errors print `error:` plus a `help:` line naming the next step.
  Examples: 1Password permission denied, pool exhausted, allowlist drift, or a
  key that does not match the published one.

Pod side: `pod-client.sh` is unchanged apart from `LILITH_WG_HOSTS` (it
appends the `/etc/hosts` line). Onetun forwards `127.0.0.1:80 →
192.168.1.12:80` and `127.0.0.1:18240 → 10.0.1.2:18240`. The pod's
`AllowedIPs` for the hub peer are exactly `10.0.1.2/32, 10.201.0.1/32,
192.168.1.12/32`, never a default route. `render.py --config wg-config.toml
--out-dir <dir> --slot N` renders the same artifacts for review with a
`$LILITH_WG_PRIVATE_KEY` placeholder.

### Variant C verification (C6a exit condition)

Use the checklist in talos `clusters/archibald/lilith-wg/README.md`. In
short: each allowlisted service answers from a test pod; everything else
(`192.168.1.12:443`, other LAN hosts, `10.0.1.2:22`, the internet) is dropped;
within about 2 min of `slot-release`, the old key can no longer handshake and
the other slots stay up. The hermetic regression
`sh clusters/archibald/lilith-wg/test/run.sh` (talos) must print `PASS`. It
covers live key rotation from the mounted secret: the old peer is removed,
the new one added, and the interface is not restarted.

---

The sections below describe the **superseded** variants A/B (hub and
gateway managed over SSH with `peer.py add/remove/list`). They are kept for
the record. `peer.py` refuses these commands on a variant C config.

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

**The key never reaches argv.** `pod-client.sh`'s onetun invocations set
`ONETUN_PRIVATE_KEY="$LILITH_WG_PRIVATE_KEY"` in onetun's environment instead of
passing `--private-key`. onetun v0.3.10 reads that variable and warns when the key
comes on the command line (seen in the first live run, 2026-09-28).

**Run one WireGuard client per slot key at a time.** onetun and wireproxy (or a
kernel interface) sharing a slot's key fight over the hub's endpoint for that peer:
the last handshake wins, and the other client silently loses its path. The
2026-09-28 live check therefore stops onetun before it starts wireproxy for the
SOCKS negative tests, and counts a "blocked" result only when a SOCKS positive
control in the same session passed.

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
| `scripts/lilith/wireguard/wg-config.toml.example` | **Variant C** template: endpoint, subnet, slot pool, 1Password items, talos paths, service allowlist (mirrors talos `allowlist.conf`). Copy to `wg-config.toml` (gitignored). |
| `scripts/lilith/wireguard/wg-config.variant-a.toml.example` | Superseded A/B template (hub + hephaistos gateway over SSH). |
| `scripts/lilith/wireguard/slots.py` | Variant C slot state machine over 1Password (`lilith-wg-slots` + `lilith-wg-slot-<n>`): status (assigned/free/rotating); assign (private key read at call time, derive-check, enable, allowlist parity against talos); release (publish-disabled-first rotation, piped JSON only). Injectable runner and clock. |
| `scripts/lilith/wireguard/test_slots.py` | Variant C tests with a fake `op`/`git` and clock: states, assignment, exhaustion, tamper and drift refusals, publish-before-keypair ordering, interrupted-release repair, no key material on disk or in argv. |
| `scripts/lilith/wireguard/render.py` | Variant C: `--slot N` renders one slot's pod client (no key material) plus talos `allowlist.conf` parsing and the drift check. A/B: pure rendering: wg-quick configs, nftables rulesets, onetun/wireproxy pod-client artifacts. Generates/caches the hub and gateway's persistent keypairs under `[keys].dir` (asserted outside the repo). No network I/O. |
| `scripts/lilith/wireguard/peer.py` | Variant C: `slot-status` / `slot-assign` / `slot-release` (see above). Superseded A/B: `add` / `remove` / `list` pod peers on the hub over SSH (`wg set` live + persisted config + nft reload). Never writes a pod private key to disk; prints the pod-side bundle to stdout once. `--dry-run` everywhere. |
| `scripts/lilith/wireguard/pod-client.sh` | Runs inside the pod: picks kernel/onetun/wireproxy, brings the tunnel up, health-checks the allowlist, fails closed. |
| `scripts/lilith/wireguard/test_render.py` | Rendering determinism, allowlist→nft correctness (including a regression test for the hostname-vs-IP self/remote bug caught while building this), peer `/32` allocation/removal, and "no private key material ends up under the repo" checks. Runs `shellcheck` on `pod-client.sh` when available. |
