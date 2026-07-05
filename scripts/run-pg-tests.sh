#!/usr/bin/env bash
# Run the full Phase-38 Postgres integration suite against a standalone CNPG
# instance, validating the relocatable-Akh workflow end-to-end:
#   - akhomed-store::live_pg     ensure_akh / WAL append / membership /
#                                advisory-lock split-brain guard /
#                                concurrent advisory-locked migrators
#   - snapshot_live              snapshot populates tables; wake reconstructs an
#                                identical akh; degraded wake with a missing table
#   - wal_live                   record → flush writes rows
#   - relocation                 random assert/retract WAL tail recovers exactly
#                                (property test over several seeds)
#
# These tests are #[ignore]'d and gated on AKH_TEST_PG_URL, so a normal
# `cargo test` never touches Postgres. This script wires it all together.
#
# Usage:
#   scripts/run-pg-tests.sh            # bring up PG (if needed), run all, leave PG up
#   AKH_KEEP_PG=0 scripts/run-pg-tests.sh   # tear PG down afterwards
#   AKH_TEST_PG_URL=postgres://… scripts/run-pg-tests.sh   # use an existing PG
set -euo pipefail
cd "$(dirname "$0")/.."

# Bring up the standalone CNPG unless an external URL was supplied.
if [[ -z "${AKH_TEST_PG_URL:-}" ]]; then
  echo ">> starting standalone CloudNativePG…"
  line="$(scripts/pg-standalone.sh up | tail -1)"   # "AKH_TEST_PG_URL=postgres://…"
  export AKH_TEST_PG_URL="${line#AKH_TEST_PG_URL=}"
  STARTED_PG=1
else
  echo ">> using AKH_TEST_PG_URL from environment"
  STARTED_PG=0
fi
echo ">> AKH_TEST_PG_URL=$AKH_TEST_PG_URL"

# --test-threads=1: the live tests share one database and each call
# run_migrations; serial avoids the first-migration race (the locked path is
# proven separately by `concurrent_locked_migrators_do_not_race`).
status=0
echo; echo "== akhomed-store::live_pg =="
cargo test -p akhomed-store --test live_pg -- --ignored --test-threads=1 || status=1

echo; echo "== main crate: snapshot_live + wal_live + relocation =="
cargo test --no-default-features --features server \
  --test snapshot_live --test wal_live --test relocation \
  -- --ignored --test-threads=1 || status=1

if [[ "${AKH_KEEP_PG:-1}" == "0" && "${STARTED_PG:-0}" == "1" ]]; then
  echo; echo ">> tearing down standalone Postgres"
  scripts/pg-standalone.sh down
fi

echo
if [[ "$status" == "0" ]]; then
  echo "ALL POSTGRES INTEGRATION TESTS PASSED"
else
  echo "SOME POSTGRES INTEGRATION TESTS FAILED" >&2
fi
exit "$status"
