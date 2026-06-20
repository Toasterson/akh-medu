#!/usr/bin/env bash
# Stand up the homelab CloudNativePG image as a *standalone* Postgres for
# exercising the Phase-38 relocatable-Akh workflow end-to-end: snapshot / WAL /
# wake / relocate / the split-brain advisory lock.
#
# The CNPG image (ghcr.io/cloudnative-pg/postgresql) is operator-driven — it has
# no docker-entrypoint auto-init like the official `postgres` image (its default
# CMD is just `bash`). So we drive `initdb` + `postgres` ourselves. This keeps
# the test database on the *same image family* as production (CNPG, with pgvector
# baked in); the production cluster additionally carries Apache AGE (homelab
# wrapper image, Anima WI 019dc614) and optionally TimescaleDB — both are
# guarded in the migrations and no-op when absent, so the base image suffices for
# the relocation workflow (AGE projection is Phase 38d, deferred).
#
# Usage:
#   scripts/pg-standalone.sh up      # pull (if needed) + init + start, wait ready
#   scripts/pg-standalone.sh down    # stop + remove container and volume
#   scripts/pg-standalone.sh url     # print the AKH_TEST_PG_URL to use
#   scripts/pg-standalone.sh psql    # open a psql shell inside the container
#   scripts/pg-standalone.sh logs    # tail container logs
#
# Overridable via env: AKH_PG_IMAGE, AKH_PG_CONTAINER, AKH_PG_VOLUME,
# AKH_PG_PORT, AKH_PG_USER, AKH_PG_PASSWORD, AKH_PG_DB.
set -euo pipefail

IMAGE="${AKH_PG_IMAGE:-ghcr.io/cloudnative-pg/postgresql:16}"
NAME="${AKH_PG_CONTAINER:-akh-pg}"
VOLUME="${AKH_PG_VOLUME:-akh-pg-data}"
PORT="${AKH_PG_PORT:-5432}"
PGUSER="${AKH_PG_USER:-akh}"
PGPASS="${AKH_PG_PASSWORD:-akh}"
PGDB="${AKH_PG_DB:-akhmedu}"

url() { echo "postgres://${PGUSER}:${PGPASS}@127.0.0.1:${PORT}/${PGDB}?sslmode=disable"; }

case "${1:-up}" in
  up)
    docker pull "$IMAGE" >/dev/null
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    # The CNPG image runs as uid 26 (postgres) and a named volume mounts
    # root-owned, so init as root, fix ownership, then drop to postgres via
    # `runuser` for every server-touching command (postgres refuses to run as
    # root). Mirrors the official postgres image's gosu dance.
    docker run -d --name "$NAME" -p "${PORT}:5432" \
      -v "${VOLUME}:/var/lib/postgresql/data" \
      --user 0 \
      -e PGUSER="$PGUSER" -e PGPASS="$PGPASS" -e PGDB="$PGDB" \
      --entrypoint bash "$IMAGE" -c '
        set -e
        export PGDATA=/var/lib/postgresql/data/pgdata
        mkdir -p "$PGDATA"
        chown -R postgres:postgres /var/lib/postgresql/data
        chmod 700 "$PGDATA"
        as_pg() { runuser -u postgres -- "$@"; }
        if [ ! -s "$PGDATA/PG_VERSION" ]; then
          PWFILE=$(mktemp); printf "%s" "$PGPASS" > "$PWFILE"; chown postgres "$PWFILE"
          as_pg initdb -D "$PGDATA" -U "$PGUSER" --pwfile="$PWFILE" \
            --auth-host=scram-sha-256 --auth-local=trust
          rm -f "$PWFILE"
          echo "host all all all scram-sha-256" >> "$PGDATA/pg_hba.conf"
          echo "listen_addresses = '\''*'\''"     >> "$PGDATA/postgresql.conf"
          as_pg pg_ctl -D "$PGDATA" -w start
          as_pg psql -U "$PGUSER" -d postgres -v ON_ERROR_STOP=1 \
            -c "CREATE DATABASE \"$PGDB\" OWNER \"$PGUSER\";"
          as_pg pg_ctl -D "$PGDATA" -m fast -w stop
        fi
        exec runuser -u postgres -- postgres -D "$PGDATA"
      ' >/dev/null
    printf "waiting for postgres"
    for _ in $(seq 1 60); do
      if docker exec "$NAME" pg_isready -U "$PGUSER" -d "$PGDB" >/dev/null 2>&1; then
        echo " ready"
        echo "AKH_TEST_PG_URL=$(url)"
        exit 0
      fi
      printf "."; sleep 1
    done
    echo " TIMEOUT"; docker logs "$NAME" 2>&1 | tail -40; exit 1
    ;;
  down)
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker volume rm "$VOLUME" >/dev/null 2>&1 || true
    echo "removed container '$NAME' + volume '$VOLUME'"
    ;;
  url)  url ;;
  psql) docker exec -it "$NAME" psql -U "$PGUSER" -d "$PGDB" ;;
  logs) docker logs --tail "${2:-50}" "$NAME" ;;
  *) echo "usage: $0 {up|down|url|psql|logs}" >&2; exit 2 ;;
esac
