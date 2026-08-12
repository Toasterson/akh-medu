#!/usr/bin/env python3
"""Rename an akh in the Postgres backing store, carrying its data across.

An akh's identity is derived from its workspace name:

    akh_id       = uuid_v5(AKH_NAMESPACE, "akh:{name}")
    workspace_id = uuid_v5(AKH_NAMESPACE, "workspace:{name}")

(see `akh_id_for` / `workspace_id_for` in `src/wal.rs`). So renaming a workspace
mints a *new* identity, and every row keyed to the old id — symbols, facts, the
mutation log, drives, episodes — is orphaned. There is no `akh workspace rename`
verb, and renaming the `.toml` alone silently strands the history.

This rewrites the id across `akhomed_akh` and every table carrying an `akh_id`,
in one transaction, verifying before it commits.

Two design notes worth keeping:

* The table list is **discovered** from `information_schema`, not hardcoded, so
  a table added by a later migration cannot be silently left behind.
* It first re-derives the ids of every existing akh and checks them against the
  live rows. If the namespace math is wrong the run aborts before touching
  anything, rather than scattering rows under bogus ids.

There are no foreign keys to `akhomed_akh(id)` (verified via `pg_constraint`;
a code comment claiming `akh_state` FKs to it is stale), so plain UPDATEs are
sufficient — no deferred constraints, no insert/delete dance.

Run with the akh **not resident** (dormant, holding no advisory lock). Nothing
here stops a running daemon from writing under the old id mid-flight.

Usage:
    export AKH_PG_DSN="postgres://akhomed@host:5432/akhomed?sslmode=require&..."
    python3 scripts/rename-akh.py old-name new-name [old2 new2 ...]
    python3 scripts/rename-akh.py old-name new-name --commit

Without --commit it is a dry run: it does all the work, prints it, and rolls
back. Reversible by running it again with the names swapped.
"""
import os
import sys
import uuid

try:
    import psycopg2
except ImportError:  # pragma: no cover - operator-facing
    sys.exit("psycopg2 is required: pip3 install psycopg2-binary")

# Must match AKH_NAMESPACE in src/wal.rs.
NS = uuid.UUID(int=0x616B685F6D6564755F76350000000001)


def akh_id(name: str) -> uuid.UUID:
    return uuid.uuid5(NS, f"akh:{name}")


def validate_derivation(cur) -> bool:
    """Re-derive every existing akh id and compare against the live rows."""
    cur.execute("select name, id from akhomed_akh order by name")
    rows = cur.fetchall()
    if not rows:
        print("no akhs in this database")
        return False
    ok = True
    for name, ident in rows:
        calc = akh_id(name)
        if str(calc) != str(ident):
            print(f"  MISMATCH {name}: db={ident} derived={calc}")
            ok = False
    print(f"uuid5 derivation reproduces all {len(rows)} existing akh ids"
          if ok else "DERIVATION IS WRONG — refusing to touch anything")
    return ok


def rename(cur, tables, old_name: str, new_name: str) -> int:
    old, new = akh_id(old_name), akh_id(new_name)
    print(f"\n=== {old_name} -> {new_name}\n    {old}  ->  {new}")

    cur.execute("select count(*) from akhomed_akh where id=%s", (str(old),))
    if cur.fetchone()[0] != 1:
        raise SystemExit(f"    ABORT: no akhomed_akh row for {old_name}")
    cur.execute("select count(*) from akhomed_akh where id=%s or name=%s",
                (str(new), new_name))
    if cur.fetchone()[0] != 0:
        raise SystemExit(f"    ABORT: {new_name} already exists")

    moved = 0
    for t in tables:
        cur.execute(f"select count(*) from {t} where akh_id=%s", (str(old),))
        n = cur.fetchone()[0]
        if n:
            print(f"      {t:34s} {n:>7d} rows")
            moved += n
        cur.execute(f"update {t} set akh_id=%s where akh_id=%s", (str(new), str(old)))
    cur.execute("update akhomed_akh set id=%s, name=%s where id=%s",
                (str(new), new_name, str(old)))

    leftover = 0
    for t in tables:
        cur.execute(f"select count(*) from {t} where akh_id=%s", (str(old),))
        leftover += cur.fetchone()[0]
    cur.execute("select count(*) from akhomed_akh where id=%s", (str(old),))
    leftover += cur.fetchone()[0]
    landed = sum(
        (cur.execute(f"select count(*) from {t} where akh_id=%s", (str(new),)) or
         cur.fetchone()[0])
        for t in tables
    )
    if leftover or landed != moved:
        raise SystemExit(
            f"    ABORT: leftover={leftover} landed={landed} expected={moved}")
    print(f"    {moved} child rows + 1 akh row; verified 0 left under the old id")
    return moved


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--commit"]
    commit = "--commit" in sys.argv
    if not args or len(args) % 2:
        return print(__doc__) or 2

    dsn = os.environ.get("AKH_PG_DSN")
    if not dsn:
        return print("set AKH_PG_DSN to the akhomed Postgres DSN") or 2

    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    if not validate_derivation(cur):
        return 1

    cur.execute(
        "select table_name from information_schema.columns "
        "where column_name='akh_id' and table_schema='public' order by table_name"
    )
    tables = [r[0] for r in cur.fetchall()]
    print(f"{len(tables)} tables carry akh_id")

    for old_name, new_name in zip(args[::2], args[1::2]):
        rename(cur, tables, old_name, new_name)

    if commit:
        conn.commit()
        print("\nCOMMITTED")
    else:
        conn.rollback()
        print("\nDRY RUN — rolled back. Re-run with --commit to apply.")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
