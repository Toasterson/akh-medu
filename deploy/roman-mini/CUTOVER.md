# Roman cutover to full akhs (Phase 53f runbook)

- **ADR:** [113](../../docs/ai/decisions/113-full-akh-always.md)
- **Briefs:** [docs/ai/phases/phase-53/README.md](../../docs/ai/phases/phase-53/README.md) ("53d implementation notes" and "53f")
- **Anima:** 53f `01a0d46a-18db`, umbrella `01a0d43b-bf90`
- **Deploy rule:** [.claude/rules/deploy-from-main.md](../../.claude/rules/deploy-from-main.md)

This runbook moves janus, minerva and vesta from the stripped
`akhomed-d563ba34a247` plus akhsessd automations plus akhrtd `ollama-*` turns
onto one full akhomed built from `origin/main`. After the cutover, each Roman
runs its own daemon loop, sleep, harvest and in-process turn (GLM-5.3 on Ollama
Cloud). Events arrive through its own reactor, and it calls external tools
through its own `[[mcp.servers]]`. Identities, facts and the 598 captured turns
stay in `state/core/akh.sqlite3`.

The repository renders everything; nothing in this directory touches the live
deployment until an operator runs the commands below.

## Paths

```sh
ROOT="$HOME/Library/Application Support/roman-akhs"   # live deployment
REPO=/tmp/akh-main-cutover                            # clean origin/main checkout (created below)
TS=$(date -u +%Y%m%dT%H%M%SZ)
BK="$ROOT/backups/phase53-$TS"
UID_=$(id -u)                                         # 501
```

| What | Where |
|---|---|
| Core store (SQLite, WAL) | `$ROOT/state/core/akh.sqlite3` (+ `-wal`, `-shm`, `.lock` files — never unlink locks) |
| Host config | `$ROOT/core-config/akh-medu/config.toml` |
| Workspace TOMLs | `$ROOT/core-config/akh-medu/workspaces/{janus,minerva,vesta}.toml` |
| Core API config | `$ROOT/config/core.json` |
| Event routes | `$ROOT/config/events.json` (staged: `config/events.akh-target.json`) |
| Reactor seeds | `$ROOT/config/reactor-seeds/{janus,minerva,vesta}.json` |
| Core release record | `$ROOT/config/core-release.json`, `$ROOT/evidence/core-binary.json` |
| T3 transport switch | `$ROOT/config/t3-acp.json` |
| System LaunchDaemons | `/Library/LaunchDaemons/dev.roman-akhs.{core,events,runtime,runtime-minerva,runtime-vesta,sandbox,executor,board-tunnel,sokar-tunnel,metrics-tunnel,otlp-tunnel}.plist` |
| User LaunchAgents | `~/Library/LaunchAgents/dev.roman-akhs.{sessions,observer,laya,laya-controller,janus-escalation,vesta-ops,t3-activity,grafana-tunnel,anima-activity}.plist` |
| Old core binary (kept) | `$ROOT/bin/akhomed-d563ba34a247` |

## Scripted run

Steps 2–8 are scripted in `cutover.sh`. After step 1 (build from main), run
`sudo DRY_RUN=1 deploy/roman-mini/cutover.sh "$REPO"` to review the plan, then
`sudo deploy/roman-mini/cutover.sh "$REPO"`. It stops at the first failure and
prints the backup directory for Rollback. Root is needed only for the system
LaunchDaemons; every other step runs as the deployment owner.

## Core secrets as files

The core is a system LaunchDaemon, so it cannot read the login keychain that
`anima-pat` uses. Write the token once, from a login session, before the
cutover (and again whenever the PAT rotates):

```sh
umask 077
/Users/toasty/.local/bin/anima-pat get-auth-header | python3 -c \
  'import json,sys;print(json.load(sys.stdin)["Authorization"].removeprefix("Bearer "))' \
  > "$ROOT/secrets/anima-token"
```

Do the same for `secrets/ollama-key` and `secrets/archive-token`
(`op-service-account read` of the references in `daemon_credentials.py`). The
core reads these files first and calls 1Password only as a fallback: a
KeepAlive crash loop that reads 1Password on every start exhausts the service
account's rate limit (2026-09-25). A missing credential is logged and the core
starts without it.

## 0. Preconditions (stop if any fails)

1. The Phase 53 branches are merged to `main` by PR, **including
   `phase53/fixups`**: per-akh ACP scoping (`[acp] principals`) and the per-akh
   board identity (`[board]`). Without fixups, akhomed ignores both sections
   with an unknown-key warning. In that case any T3 key opens any Roman
   (`AKHOMED_SHARED_SECRET_FILE` is not scoped), and reactor `report_thread`
   posts have no board identity. Check the rendered shapes in `full_akh.py`
   against the merged code before rendering.
2. CI is green on the `main` commit you will build.
3. The operator has decided the open questions at the end of this file.
4. It is a quiet period: no incident is open, and Janus has no Astra
   escalation or mergiraf job in flight (`mcp__janus-escalation__astra_escalation_status`).

## 1. Build from main, with a provenance check

```sh
git -C ~/ws/akh-medu fetch origin
rm -rf "$REPO" && git -C ~/ws/akh-medu worktree add --detach "$REPO" origin/main
cd "$REPO"
test -z "$(git status --porcelain)" && echo clean
git merge-base --is-ancestor HEAD origin/main && echo "HEAD $(git rev-parse --short=12 HEAD) is on origin/main"
export CARGO_TARGET_DIR="$REPO/target"
cargo build --release --locked --features server --bin akhomed --bin akh
cargo build --release --locked -p akh-session-services \
  --features session,runtime,executor,sandbox,events,client,trigger \
  --bin akheventd --bin akh-acp --bin akhsessd --bin akhrtd --bin akhsbxd --bin akhexecd --bin akh-trigger
SHA=$(git rev-parse --short=12 HEAD)
shasum -a 256 target/release/{akhomed,akh,akheventd,akh-acp,akhsessd,akhrtd,akhsbxd,akhexecd,akh-trigger} \
  | tee "/tmp/phase53-binaries-$SHA.sha256"
```

Quote both provenance lines (`clean` and the `merge-base` line) in the deploy
report. `configure_core.py` repeats the check and refuses a dirty or off-main
checkout.

## 2. Pause admission and wait for quiet

```sh
sudo launchctl bootout system/dev.roman-akhs.events
launchctl bootout gui/$UID_/dev.roman-akhs.observer
launchctl bootout gui/$UID_/dev.roman-akhs.laya-controller
```

Wait until no Roman turn is running. The akhsessd compact session view must
show no `running` turn:

```sh
curl -s -H "Authorization: Bearer $(cat "$ROOT/secrets/operator.key")" \
  'http://127.0.0.1:18431/v1/sessions?compact=true' | python3 -c \
  'import json,sys; s=json.load(sys.stdin); print([x.get("id") for x in s if x.get("status")=="running"] or "quiet")'
```

The akheventd outbox keeps undelivered events. They are delivered after
re-targeting (step 8).

## 3. Stop the core and back it up

```sh
sudo launchctl bootout system/dev.roman-akhs.core
mkdir -p "$BK" && chmod 700 "$BK"
sqlite3 "$ROOT/state/core/akh.sqlite3" ".backup '$BK/akh.sqlite3'"
sqlite3 "$BK/akh.sqlite3" 'pragma integrity_check;'            # must print ok
shasum -a 256 "$ROOT/state/core/akh.sqlite3" "$BK/akh.sqlite3" | tee "$BK/SHA256SUMS"
cp -Rp "$ROOT/core-config" "$ROOT/config" "$BK/"
cp -p /Library/LaunchDaemons/dev.roman-akhs.*.plist ~/Library/LaunchAgents/dev.roman-akhs.*.plist "$BK/"
cp -p "$ROOT/bin/t3_entrypoint.py" "$ROOT/bin/roman-janus" "$ROOT/bin/daemon_credentials.py" "$ROOT/bin/observer.py" "$BK/"
```

Do not delete or move any `*.lock` file. The OS releases the lock when the
owning process exits.

## 4. Install binaries and render the full-akh configuration

```sh
cd "$REPO"
python3 deploy/roman-mini/configure_core.py --root "$ROOT" --binaries "$CARGO_TARGET_DIR/release" --repo "$REPO"
```

This does the following:

- installs the core at the stable path `bin/akhomed` (copy-to-temp +
  codesign + atomic rename — the running inode is never touched) and keeps a
  versioned `bin/akhomed-main-$SHA` alongside it for rollback and audit; only
  `bin/akhomed` is ever exec'd. Also installs `bin/akh-acp-main-$SHA` (unchanged);
  the old binaries stay in place;
- signs `bin/akhomed` with the identity in `config/codesign.json`
  (`{"identity": "...", "identifier": "dev.akh-medu.akhomed"}`) if that file
  exists, or ad-hoc (`codesign --sign -`) with a printed warning if it does
  not. **A stable signing identity is what makes the macOS Local Network
  privacy (TCC) approval survive a redeploy** — see "Local Network privacy
  (TCC) and the stable core path" below; without `config/codesign.json` every
  deploy is again a new TCC identity;
- writes `config/core-release.json` and `evidence/core-binary.json`
  (commit, `build_sha256` — the unsigned `target/release` artifact — `sha256`
  — the installed signed `bin/akhomed` — `codesign`
  `{identity, identifier, designated_requirement}`, `build_command`,
  `binary` = `bin/akhomed`, `versioned_binary` = the versioned copy);
- renders `config.toml`, the three workspace TOMLs, `core.json` (with a new
  `secrets/core-events.key`), `secrets/acp-clients.keys`, `t3-acp.json`, the
  reactor seeds, the staged `events.akh-target.json` and
  `launchd/dev.roman-akhs.core.plist`;
- copies the new `daemon_credentials.py` and `t3_entrypoint.py` into `bin/`.

`daemon_credentials.py` role `core` refuses to exec anything whose sha256
does not match `config/core-release.json`'s `sha256` — this, not the
filename, is the real provenance gate (the filename check only constrains
the file to `bin/akhomed` or the legacy `bin/akhomed-main-<sha>` shape inside
`bin/`).

### Local Network privacy (TCC) and the stable core path

macOS Local Network privacy blocks a *new* binary from the LAN (`No route to
host`, os error 65) until the operator approves it once in *System Settings
→ Privacy & Security → Local Network*. TCC keys that approval off the
binary's installed path and code signature together, so the old
`bin/akhomed-main-<sha>`-per-deploy scheme meant a fresh TCC identity — and a
fresh manual approval — on every deploy. Installing at the stable path
`bin/akhomed` and signing it with a stable identity (`config/codesign.json`)
keeps the same TCC identity across deploys, so **the approval should need to
happen only once, the first time `config/codesign.json` is in place.**

`config/codesign.json` is operator-written, not repository-rendered (it
names a real signing identity, so it stays out of the repo). Today the
keychain on this Mac has only `Apple Development: toasterson@gmail.com
(QPJC8WM3MN)` (team 84CM4GHWH4); write:

```sh
cat > "$ROOT/config/codesign.json" <<'JSON'
{"identity": "Apple Development: toasterson@gmail.com (QPJC8WM3MN)", "identifier": "dev.akh-medu.akhomed"}
JSON
chmod 600 "$ROOT/config/codesign.json"
```

A Developer ID Application certificate should replace this identity later
(same JSON shape, no code change); an `Apple Development` identity is
machine-scoped to the keychain that created it, so it still works for a
LaunchDaemon on this Mac but would not travel to another machine. `--options
runtime` (the hardened runtime) is deliberately not used: it is unneeded
without notarization and could break library loading.

**Scope:** only `bin/akhomed` (the Roman core, one per-Roman LAN-facing
process, hence the only one that hits Local Network TCC) gets the
stable-path-plus-signing treatment. The session-platform binaries
(`akheventd`, `akhsessd`, `akhrtd`, `akhsbxd`, `akhexecd`, `akh-trigger`) and
`akh-acp` are installed with a plain `install -m 755`/`install_binary` copy
per deploy, as before, and are not code-signed by `configure_core.py`. They
either only listen on loopback or are short-lived CLI invocations, so they do
not hit the same TCC gate; generalizing the stable-path-plus-signing scheme
to them would mean rewriting each of their operator install/rollback steps
throughout this file and INVENTORY.md, which is a bigger change than this
pass covers. Revisit if one of them starts making outbound LAN connections
that trigger a TCC prompt.

It prints warnings, for example a server that has no granted tools. Read
every warning. Then check the rendered configuration:

```sh
python3 -c 'import tomllib,sys;[tomllib.load(open(p,"rb")) for p in sys.argv[1:]];print("toml ok")' \
  "$ROOT"/core-config/akh-medu/config.toml "$ROOT"/core-config/akh-medu/workspaces/*.toml
grep -n 'deployment\|auto_start\|tcp_port' "$ROOT/core-config/akh-medu/config.toml"
```

Install the session-platform binaries from the same SHA. **akheventd is
required**, because the akh route target is new in 53d. Replace akhsessd,
akhrtd, akhsbxd and akhexecd together or leave all four alone: 53c added
`OperationReceipt.principal` under `deny_unknown_fields`, so mixed versions
reject each other's receipts.

```sh
for b in akheventd akhsessd akhrtd akhsbxd akhexecd akh-trigger; do
  cp -p "$ROOT/bin/$b" "$BK/$b.pre-phase53"
  install -m 755 "$CARGO_TARGET_DIR/release/$b" "$ROOT/bin/$b"
done
shasum -a 256 "$ROOT"/bin/{akheventd,akhsessd,akhrtd,akhsbxd,akhexecd,akh-trigger} \
  > "$ROOT/evidence/services-main-$SHA.sha256"
for r in janus minerva vesta; do install -m 700 deploy/roman-mini/t3_entrypoint.py "$ROOT/bin/roman-$r"; done
cp -p deploy/roman-mini/observer.py "$ROOT/bin/observer.py"
```

The observer update records akh admissions as `admitted` and a
`no_subscriber` rejection as a workflow failure. The old copy would mark every
akh-routed delivery `rejected`.

### Wire compatibility: restart readers before writers

Phase 53 adds two things that older binaries cannot read. Both types keep
`deny_unknown_fields` on purpose; the fix is the restart order, not a looser
reader.

| New on the wire | Written by | Read by | An old reader does this |
|---|---|---|---|
| `OperationReceipt.principal` (53c; only serialized when set) | akhexecd | akhsessd (broker, snapshots, publication, repository bootstrap; also stored in its redb), akhsbxd, akhomed (`[exec] kind = "execd"`) | rejects the receipt, so the tool call fails |
| `DerivationKind::DelegatedCall`, provenance tag 108 (53e) | akhomed (every `/v1` delegate write) | akhomed on every host that can wake or relocate the akh (redb ledger, Postgres snapshot and WAL) | cannot decode the provenance ledger |

Order:

1. **Readers first.** Restart akhsessd, akhsbxd and akhomed on the new binaries
   before akhexecd. Until akhexecd restarts, the old akhexecd writes receipts
   without `principal`, and the new readers accept those.
2. **akhexecd last.** From now on receipts may carry `principal`.
3. **Every akhomed that can host a given akh moves together.** This covers the
   Mac core and any relocation target (ADR 049) that could wake janus, minerva
   or vesta from the same store. Upgrade all of them before the first delegated
   `/v1` call, which is the first `DelegatedCall` write. On this Mac there is
   one core, so step 5 covers it. If a second host ever joins, upgrade it first.

Rollback carries the same constraint in reverse. Once akhexecd has written
receipts with `principal`, or akhomed has written a `DelegatedCall`, an old
akhsessd or old akhomed cannot read that state. Either keep the new readers,
or restore their stores from `$BK` (see Rollback). Downgrade akhexecd before
the readers, never after.

## 5. Start the full core

```sh
sudo cp "$ROOT/launchd/dev.roman-akhs.core.plist" /Library/LaunchDaemons/
sudo chown root:wheel /Library/LaunchDaemons/dev.roman-akhs.core.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/dev.roman-akhs.core.plist
```

Before calling it started, confirm the SHA again and quote the result in the
deploy report:

```sh
python3 -c 'import json;r=json.load(open("'"$ROOT"'/config/core-release.json"));print(r["binary"],r["commit"])'
git -C "$REPO" merge-base --is-ancestor "$(python3 -c 'import json;print(json.load(open("'"$ROOT"'/config/core-release.json"))["commit"])')" origin/main && echo on-main
ps -axo pid,command | grep -c "[a]khomed-main-$SHA"                # 1
```

## 6. Health checks

```sh
curl -s http://127.0.0.1:18430/healthz                                  # {"service":"akhomed",...}
nc -z 127.0.0.1 18444 && echo acp-tcp-up
OP="Authorization: Bearer $(cat "$ROOT/secrets/operator.key")"
for r in janus minerva vesta; do
  curl -s -H "$OP" "http://127.0.0.1:18430/v1/akhs/$r/tools" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)))'
done                                                                    # 2 each: kg_query, reactor_manage
grep -E 'auto-start|akh::autonomy|akh::mcp|in_process_forbidden|ERROR' "$ROOT/logs/core.err.log" | tail -50
```

Expected in the log:

- all three akhs auto-started;
- `akh::mcp` discovery for each `[[mcp.servers]]`. An unreachable server is a
  warning and is retried lazily;
- no `akh::config::in_process_forbidden`;
- autonomy task reports, with the librarian still off because `AKH_LIBRARIAN`
  is unset.

Then run one direct turn per Roman through T3. `config/t3-acp.json` now routes
the Roman providers through `akh-acp --akhomed`. Pick the `roman-<role>`
provider in T3 and ask a trivial question that requires one tool call (for
example "what is on your ledger?"). A reply plus a `mcp__…` or `memory_recall`
tool receipt shows that the in-process turn, the tools and ACP work.

## 7. Apply and enable reactor seeds

```sh
cd "$REPO"
python3 deploy/roman-mini/apply_reactor_seeds.py "$ROOT" --dry-run | head -40   # review
python3 deploy/roman-mini/apply_reactor_seeds.py "$ROOT"                       # add (staged disabled)
python3 deploy/roman-mini/apply_reactor_seeds.py "$ROOT" --enable              # needs AKH_REACTOR_ENABLE=1 (daemon_credentials.py core sets it)
```

The seeds are one `event` schedule per akheventd route. Each is an `llm_turn`
with `allow_tools`, the akhsessd wake instructions (with 53i tool names), a
route-specific ADR 105 `exit_condition`, `coalesce` from the route's
`coalesce_pending`, and `report_thread` on inbox and laya schedules:

| akh | schedule source (`<source>/<event_type>`) | coalesce |
|---|---|---|
| janus | `roman-inbox-janus/djehuty.inbox`, `roman-anima/anima.janus.changed`, `roman-anima/laya.janus.intervention` | no |
| minerva | `roman-inbox-minerva/djehuty.inbox`, `roman-anima/anima.minerva.changed`, `roman-anima/laya.minerva.intervention` | no |
| vesta | `roman-inbox-vesta/djehuty.inbox`, `roman-alerts/laya.vesta.intervention` | no |
| vesta | `roman-alerts/monitoring.changed` | yes |

## 8. Re-target events and resume admission

Do this only after step 7 enabled every schedule. An event that no enabled
schedule subscribes to gets `no_subscriber`, which is terminal and never
retried.

```sh
python3 deploy/roman-mini/retarget_events.py "$ROOT"     # backs up to evidence/events-before-phase53.json
sudo launchctl bootstrap system /Library/LaunchDaemons/dev.roman-akhs.events.plist
launchctl bootstrap gui/$UID_ ~/Library/LaunchAgents/dev.roman-akhs.observer.plist
launchctl bootstrap gui/$UID_ ~/Library/LaunchAgents/dev.roman-akhs.laya-controller.plist
```

Now prove the 53d acceptance: send minerva an addressed
`coordination/delegation@v1` on the Board (for example "record a quiet check
and reply with your ledger size"). Within about 30 s:

- akheventd shows a delivery with `event_admission.schedules_woken` containing
  `roman-minerva-djehuty.inbox`;
- the core log shows a reactor `LlmTurn` run for minerva;
- no new akhsessd automation session exists for it.

## 9. Verify the lifeform capabilities

Right after the first sleep tick, and again within 24 h:

```sh
DB="$ROOT/state/core/akh.sqlite3"
sqlite3 -readonly "$DB" 'select hex(akh_id),harvest_watermark,cycle_count from akhomed_akh_state;'   # watermark > 0
sqlite3 -readonly "$DB" 'select count(*) from akhomed_episodes;'                                      # > 0
sqlite3 -readonly "$DB" 'select count(*) from akhomed_sleep_metrics;'                                 # > 0
sqlite3 -readonly "$DB" 'select kind,count(*) from akhomed_mutation_log group by kind;'
```

Acceptance for each Roman (ADR 113, 53f): within 24 h the sleep-metrics,
consolidation and harvest counters are non-zero, one live turn shows psyche
and communicative-intelligence telemetry, and Anima and Board work continues
without operator relay. An exhausted budget shows up as `SkippedBudget` task
reports, not as errors.

## 10. Memory hygiene

Phase 53j (ADR 113) made memory hygiene part of the akh. Nothing in this step
edits triples by hand:

- **Single-valued predicates.** A schema pack can declare a relation
  `"cardinality": "single_valued"` (optionally `"temporal": true`). The engine
  enforces it on every write (`memory_write`, `kg_mutate`, …): the newest value
  for a `(subject, predicate)` stays live and the older one is retracted
  (the TMS cascades only if no other subject still holds that value), with
  `memory_hygiene` provenance naming its successor and,
  for temporal predicates, when it stopped holding. These predicates no longer
  log `TemporalConflict` warnings.
- **Aliases.** A pack's `"aliases"` are other spellings of a relation. A write
  under an alias is stored under the canonical relation.
- **Daemon task `memory_hygiene`** (free, on by default, bounded by
  `[memory] hygiene_batch`, default 1000 mutations per tick). It runs on every
  sleep tick after harvest and does not depend on the sleep cycle succeeding.
  Each pass does three things:
  1. It merges exact spelling collisions (`next_check` / `next-check` /
     `next check`: same case-folded key once whitespace, `-` and `_` are
     unified) onto one canonical spelling. The canonical spelling is the one a
     pack declares; if no pack declares one, the most frequent spelling wins,
     with ties broken lexicographically.
  2. It retracts the older values of every single-valued predicate.
  3. It repairs contradictions on functional predicates per
     `[memory] contradiction_policy`: `warn` (the default) only reports them,
     `replace` keeps the newest value and `reject` keeps the oldest.

  Everything else is only reported: declared aliases whose spelling does not
  collide, ambiguous families, and undeclared predicates holding several
  values. Each pass logs `memory hygiene pass` under target `akh::autonomy`.
- **`kg_delete`** is now registered in the live tool registry, as is
  **`kg_merge_predicates`**. Both are `Dangerous`, so `/v1` scopes do not
  expose them unless a scope or grant names them.

`deploy/roman-mini/schemas/roman-memory.json` declares 11 state predicates the
Romans write (`next_check`, `next_action`, `planned_action`, `current_phase`,
`current_state`, `has_state`, `observed_state`, `verified_state`,
`boundary_state`, `state`, `status`) as single-valued and temporal, with their
observed alias spellings. The evidence is in each relation's `description`.

Use the `akh` CLI built in step 1:

```sh
akhc() { env AKH_SYSTEM_PATHS=0 XDG_CONFIG_HOME="$ROOT/core-config" XDG_DATA_HOME="$ROOT/core-data" \
    XDG_STATE_HOME="$ROOT/core-state" XDG_CACHE_HOME="$ROOT/core-cache" XDG_RUNTIME_DIR="$ROOT/core-runtime" \
    "$CARGO_TARGET_DIR/release/akh" "$@"; }
```

1. **Compile, review and admit the pack** once. It lands in the core store's
   registry table, and admission is a human confirm (ADR 081 decision 8):

   ```sh
   akhc schema compile "$REPO/deploy/roman-mini/schemas/roman-memory.json" --out "$BK/roman-memory.akhschema"
   akhc schema show "$BK/roman-memory.akhschema"            # review the triples
   akhc schema admit "$BK/roman-memory.akhschema" --yes
   ```

2. **Install it into each Roman.** The activation is recorded, so a later
   wake converges it again. From this point, new writes supersede and
   canonicalize:

   ```sh
   for r in janus minerva vesta; do akhc -w $r schema install roman-memory@1.0.0; done
   ```

3. **Dry-run, then run the first hygiene pass now** instead of waiting for the
   sleep tick. Repeat the real pass while its report says `"truncated": true`:

   ```sh
   for r in janus minerva vesta; do
     akhc -w $r schema hygiene --dry-run > "$BK/hygiene-$r-dry.json"
     akhc -w $r schema hygiene           > "$BK/hygiene-$r.json"
   done
   ```

4. **Fold in the declared aliases the daemon only reports**, for example
   `has_next_check` → `next_check`. Their spellings do not collide, so the
   merge is an explicit operator action:

   ```sh
   for r in janus minerva vesta; do
     akhc -w $r schema merge-predicates --declared --dry-run
     akhc -w $r schema merge-predicates --declared > "$BK/merge-declared-$r.json"
   done
   ```

   To merge a pair that is neither declared nor colliding, use
   `akhc -w <akh> schema merge-predicates --canonical <keep> --alias <fold> [--alias …]`
   (the `kg_merge_predicates` tool and `_akh/memory/merge_predicates` ACP method
   do the same).

5. **Verify.**
   - The pass is idempotent. `akhc -w $r schema hygiene --dry-run` reports
     `"mutations": 0`, `"truncated": false`, `"merged": []` and
     `"pending_declared_aliases": []`.
   - Each subject has exactly one live `next_check`. Get the relation id, then
     look for subjects that still hold more than one value. Expect no rows:

     ```sh
     for r in janus minerva vesta; do
       id=$(akhc -w $r acp create-symbol next_check --kind relation)    # resolves the existing id
       akhc -w $r acp query "SELECT ?s (COUNT(?o) AS ?n) WHERE { ?s <https://akh-medu.dev/sym/$id> ?o } GROUP BY ?s HAVING (COUNT(?o) > 1)"
     done
     ```

   - The WAL shows the retractions:
     `sqlite3 "$ROOT/state/core/akh.sqlite3" "select kind,count(*) from akhomed_mutation_log where applied_at > '<step-10 start, UTC>' group by kind;"`
     returns `assert` and `retract_batch` rows.
   - After the next sleep tick, `grep 'memory hygiene pass' "$ROOT"/logs/core.*.log`
     shows one line per Roman with `mutations=0`, and the `TemporalConflict`
     count for the declared predicates stops growing:
     `grep TemporalConflict "$ROOT"/logs/core.*.log | grep -c next_check`.

   Undeclared predicates that still hold several values per subject appear in
   the report's `unresolved` list. If one of them is state, add it to the pack
   under a new version, then admit and install it again.

Expected result, from `tests/hygiene_accept.rs` run on a private copy of the
2026-09-24 store (`AKH_HYGIENE_ACCEPT_DB=… cargo test --features server --test hygiene_accept -- --ignored --nocapture`):

| akh | `next_check` family before | after | retracted | `memory_hygiene` provenance |
|---|---|---|---|---|
| janus | 262 triples, 5 spellings, 127 subjects (29 with 2–46 values) | 127, one per subject | 366 | 403 |
| minerva | 125 triples, 4 spellings, 56 subjects (15 with 2–39 values) | 56, one per subject | 248 | 257 |
| vesta | 103 triples, 6 spellings, 54 subjects (11 with 2–32 values) | 54, one per subject | 191 | 221 |

A second pass is a no-op on all three akhs. A hygiene pass takes 1–3 s per akh.
The `memory_hygiene` daemon task can be switched off per akh with
`[autonomy] tasks = { memory_hygiene = false }`.

## 11. Retire the old paths (after 24 h and acceptance)

- Boot out and remove `dev.roman-akhs.runtime`, `runtime-minerva` and
  `runtime-vesta` (the `ollama-*` akhrtd profiles), but only after each Roman
  has passed step 9.
- In `config/sessions.json`, set `automations.<role>.enabled=false` (akheventd
  no longer routes to them), then restart `dev.roman-akhs.sessions`. akhsessd
  keeps brokering native-runtime delegation sessions.
- `t3-activity` and `anima-activity` read akhsessd sessions and do not see
  reactor turns. **Retired 2026-09-25** (ADR 113 / 53g plan PR 1 — see
  INVENTORY.md follow-up 5 and its "Mac-side steps for PR 1"): boot both out
  and remove the `t3-activity` plist and `bin/roman-activity-*` rather than
  leaving them for historical views.
- Keep `bin/akhomed-d563ba34a247` and `$BK` until the next successful full
  deploy.

## 12. Add Lilith (Lilith C2, ADR 116)

Lilith is the operator's Chief of Staff. She runs as a fourth workspace in this
core (supervisor placement decision, 2026-09-28). Her model is the Runpod vLLM
endpoint (`--served-model-name lilith`), reached over its HTTPS proxy URL.
`configure_core.py` renders her only once `config/lilith.json` exists. She
auto-starts only once every input below exists. A missing input leaves her
rendered but out of `daemon.auto_start`, with a printed warning, and the Romans
are unaffected.

Render inputs (all under `$ROOT`, mode 0600, written from a login session):

| Input | Content | Source |
|---|---|---|
| `config/lilith.json` | `{"budget": {"minor_units": N, "currency": "USD"}}`; optional `context_window` (default 32768) and `base_url_file` | operator decision (her spend ceiling) |
| `secrets/lilith-base-url` | the vLLM base URL, `https://…proxy.runpod.net/v1` | 1Password item `lilith-serve-endpoint`, field `base_url` |
| `secrets/lilith-openai-key` | the vLLM `--api-key` → env `LILITH_OPENAI_KEY` | the serve pod's key |
| `secrets/anima-lilith-pat` | her own Anima PAT → env `LILITH_ANIMA_TOKEN` (MCP server `anima`) | minted for Lilith, never Toasty's |
| `secrets/board-lilith.json` | `{"bearer_token": …}` of her own Board agent → env `ROMAN_BOARD_LILITH` (MCP server `djehuty`, `[board]`) | Board registration for `lilith` |
| `lilith/ancestry/akh-medu.git` | her dedicated clone for `git merge-base --is-ancestor <sha> origin/main` | see below |

```sh
umask 077
op-service-account read 'op://Homelab/lilith-serve-endpoint/base_url' > "$ROOT/secrets/lilith-base-url"
printf '{"budget": {"minor_units": %s, "currency": "USD"}}\n' "$BUDGET_CENTS" > "$ROOT/config/lilith.json"
git clone --bare https://code.aopc.cloud/toasterson/akh-medu.git "$ROOT/lilith/ancestry/akh-medu.git"
git -C "$ROOT/lilith/ancestry/akh-medu.git" config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
git -C "$ROOT/lilith/ancestry/akh-medu.git" fetch origin
```

The clone has to stay fresh, or her ancestry checks refuse new main SHAs
(that is the safe direction). Re-run the `fetch` before you rely on a new main
commit.

Her workspace uses `backend = "openai"`, `model = "lilith"` and
`[openai_endpoint]` (`api_key = "env:LILITH_OPENAI_KEY"`). Her
`[chief_of_staff]` table points at:

- the installed `core-data/akh-medu/workspaces/lilith/kg/skills/chief-of-staff/permission-profile.json`;
- the clone above;
- `state/lilith/cos-announcements.jsonl`, which lies outside her write roots
  (the renderer refuses otherwise);
- the budget.

She has no reactor seeds yet. Her wakes are a later slice.

Before every full render, review the plan. A full re-render has dropped hand
holds before, for example Janus's broker hold:

```sh
python3 deploy/roman-mini/configure_core.py --plan --root "$ROOT"   # writes nothing
```

To prove that the rendered `lilith.toml` loads in Rust with no key dropped
(through `AkhConfig::load_checked`, the loader akhomed uses, and with
`GuardInstall` active), run this from the checkout after the render:

```sh
AKH_LILITH_WORKSPACE_TOML="$ROOT/core-config/akh-medu/workspaces/lilith.toml" \
  cargo test --lib roman_fixture_tests::lilith
```

CI runs the same test on `fixtures/golden/lilith.toml`, and
`test_full_akh.py` requires the renderer to reproduce that file byte for byte.

`--plan` prints one line per file: `=` unchanged, `~` changed with a unified
diff, or `+` new. Then it prints the resulting `daemon.auto_start`. Without
`--repo`, it takes the host-config provenance from the installed
`config/core-release.json`. Then render as in step 4, and restart the core
(KeepAlive: `kill` the core pid). Verify:

```sh
OP="Authorization: Bearer $(cat "$ROOT/secrets/operator.key")"
curl -s http://127.0.0.1:18430/healthz
grep -E 'workspace="?lilith|Chief-of-Staff guard' "$ROOT"/logs/core.*.log | tail -20
curl -s -H "$OP" http://127.0.0.1:18430/v1/akhs/lilith/tools
curl -sf -H "Authorization: Bearer $(cat "$ROOT/secrets/lilith-openai-key")" \
  "$(cat "$ROOT/secrets/lilith-base-url")/models" | python3 -c 'import json,sys;print([m["id"] for m in json.load(sys.stdin)["data"]])'
```

Expected results:

- `Chief-of-Staff guard installed` for `lilith`. `guard failed to load` means the guard
  did not load and she runs read-only. The log line says why (the profile,
  the clone or the log path).
- `/v1/akhs/lilith/tools` answers.
- `/models` lists `lilith`.
- The Romans' startup lines are unchanged.

To roll back, remove `config/lilith.json` and render again. She drops out of
`auto_start`, and her workspace file and state stay in place.

## Rollback

Use this at any step before 11.

```sh
sudo launchctl bootout system/dev.roman-akhs.core
sudo launchctl bootout system/dev.roman-akhs.events 2>/dev/null
# Store: only if the new core wrote something you do not want to keep (it migrates on start).
sqlite3 "$BK/akh.sqlite3" 'pragma integrity_check;' && cp -p "$BK/akh.sqlite3" "$ROOT/state/core/akh.sqlite3"
rm -f "$ROOT/state/core/akh.sqlite3-wal" "$ROOT/state/core/akh.sqlite3-shm"   # never the .lock files
rm -rf "$ROOT/core-config" "$ROOT/config" && cp -Rp "$BK/core-config" "$BK/config" "$ROOT/"   # also removes t3-acp.json
sudo cp "$BK/dev.roman-akhs.core.plist" /Library/LaunchDaemons/
for b in akheventd akhsessd akhrtd akhsbxd akhexecd akh-trigger; do install -m 755 "$BK/$b.pre-phase53" "$ROOT/bin/$b"; done
for f in t3_entrypoint.py daemon_credentials.py observer.py; do cp -p "$BK/$f" "$ROOT/bin/$f"; done
for r in janus minerva vesta; do install -m 700 "$BK/roman-janus" "$ROOT/bin/roman-$r"; done
sudo launchctl bootstrap system /Library/LaunchDaemons/dev.roman-akhs.core.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/dev.roman-akhs.events.plist
launchctl kickstart -k gui/$UID_/dev.roman-akhs.observer
launchctl kickstart -k gui/$UID_/dev.roman-akhs.laya-controller
```

### Rolling back to a previous full-akh build (stable path, same generation)

The rollback above restores the pre-53f stripped core. Once the Roman is
already on a full akhomed built through `configure_core.py` (any deploy
after step 4), rolling back to an *older full-akh* build is narrower: it
only needs to re-point `bin/akhomed` at an older, already-installed
`bin/akhomed-main-<old-sha>`, re-signed with the same identity so its sha256
and signature are consistent, and `core-release.json` rewritten to match.
Two ways to do this:

**A. Rerun `configure_core.py` from the old commit** (preferred — it also
re-renders the workspace TOMLs and reactor seeds consistently for that
commit):

```sh
git -C "$REPO" checkout "$OLD_SHA"   # a clean checkout of that ancestor of origin/main
cd "$REPO" && cargo build --release --locked --features server --bin akhomed
python3 deploy/roman-mini/configure_core.py --root "$ROOT" --binaries "$REPO/target/release" --repo "$REPO"
kill $(pgrep -f bin/akhomed)   # KeepAlive restarts it
```

**B. Re-sign the already-installed versioned copy directly**, when the old
build is still on disk as `bin/akhomed-main-<old-sha>` and rebuilding is not
wanted:

```sh
OLD="$ROOT/bin/akhomed-main-<old-sha>"
TMP="$ROOT/bin/.akhomed.$$.tmp"
cp -p "$OLD" "$TMP"
IDENTITY=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["identity"])' "$ROOT/config/codesign.json")
IDENTIFIER=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("identifier","dev.akh-medu.akhomed"))' "$ROOT/config/codesign.json")
codesign --force --sign "$IDENTITY" --identifier "$IDENTIFIER" --timestamp=none "$TMP"
mv "$TMP" "$ROOT/bin/akhomed"                       # atomic rename, same filesystem
NEW_SHA256=$(shasum -a 256 "$ROOT/bin/akhomed" | awk '{print $1}')
DR=$(codesign -dr - "$ROOT/bin/akhomed" 2>/dev/null | sed 's/^designated => //')
python3 -c '
import json, sys
path, sha, identity, identifier, dr = sys.argv[1:]
release = json.load(open(path))
release["sha256"] = sha
release["codesign"] = {"identity": identity, "identifier": identifier, "designated_requirement": dr}
json.dump(release, open(path, "w"), indent=2)
' "$ROOT/config/core-release.json" "$NEW_SHA256" "$IDENTITY" "$IDENTIFIER" "$DR"
kill $(pgrep -f bin/akhomed)   # KeepAlive restarts it; daemon_credentials.py checks the sha256 it just wrote
```

Either way, `daemon_credentials.py` role `core` refuses to start unless
`bin/akhomed`'s sha256 matches `core-release.json`'s `sha256`, so a partial
rollback (binary swapped, `core-release.json` not rewritten, or vice versa)
fails closed with a clear message instead of silently running the wrong
build.

### The pre-53f rollback (whole generation)

Roll the services back writers first: akhexecd before akhsessd and akhsbxd,
as described in "Wire compatibility" under step 4. If the new core recorded any
delegated `/v1` call, the old core cannot decode that provenance. In that case
restoring `akh.sqlite3` is required, not optional.

Restoring `akh.sqlite3` throws away everything the full core harvested or
learned since step 5. Prefer keeping the store: the old core reads it, unless a
migration it cannot read ran. Check `logs/core.err.log` after the rollback
start. The `roman-janus` copy restores all three role entrypoints, because they
are identical copies of `t3_entrypoint.py`.

## Open questions for the operator

1. **`AKH_REACTOR_ENABLE=1` is process-wide.** It lets `apply_reactor_seeds.py
   --enable` work. It also lets an akh enable a schedule it staged itself with
   `reactor_manage`. There are two options:
   - keep it set: autonomy as intended by ADR 113, bounded by the budget and
     the ADR 105 exit conditions;
   - set it only for step 7, then remove it from `daemon_credentials.py` and
     restart the core.
2. **`report_thread`** is set only on inbox and laya schedules, and posts to
   the shared Roman status thread `01a0ab49-9cc4-7012-8785-1eada26fbd46`.
   Anima-change and monitoring schedules would post after every run. Decide
   whether they should report too, and to which thread.
3. **Grants policy.** `sessions` stays `grants = "optional"` (legacy nine
   tools without a grant). Flip it to `"required"` once akhsessd's
   automations are retired (step 11) and only 53e-minted delegations call
   `/v1`.
4. **Seshat over plaintext HTTP.** `seshat.lab.home/mcp` gets
   `allow_insecure_http = true`, the same trust the akhsessd endpoint has
   today. Moving Seshat behind HTTPS removes the flag.
5. **Observer backpressure.** The observer's provider cooldown and
   "active work suppresses wakes" logic read akhsessd sessions. They do not
   see reactor turns. The reactor's coalescing, the budget and the trip-wire
   now bound the load instead. Accept this, or schedule the observer→akheventd
   Rust source conversion (INVENTORY.md follow-up 1).
6. **No execd sandbox.** `[exec] kind = "none"` keeps the Romans without
   side-effecting tools, as today. Vesta's GitOps and homelab actions stay on
   the `vesta-ops` MCP broker. Provisioning an akhsbxd/akhexecd profile is a
   separate step.
7. **Contradiction policy** (step 10). The Romans keep the default
   `[memory] contradiction_policy = "warn"`. Once the state predicates are
   declared single-valued, the remaining contradictions are on undeclared
   predicates. Decide whether to switch an akh to `replace` (keep the newest
   value) or `reject` (refuse conflicting writes). Either choice also changes
   how hygiene repairs functional predicates.
