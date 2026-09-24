#!/usr/bin/env python3
"""Apply the rendered reactor seeds through akhomed's `/v1/tools/call` (53f step 6).

Calls `reactor_manage` as the core `operator` principal (configure_core.py
scopes it to reactor_manage and kg_query). `add` stages each schedule
disabled; `--enable` then enables them, which akhomed allows only while
AKH_REACTOR_ENABLE=1 is set in its environment (daemon_credentials.py core).
Adding an id that already exists is reported and skipped, so reruns are safe.
`--dry-run` prints the requests without sending them. The bearer token is read
from the token file and never printed.
"""
import argparse
import json
from pathlib import Path
import sys
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full_akh as fa  # noqa: E402


def requests_for(seeds: dict, enable: bool) -> list[dict]:
    """CognitiveToolCall bodies for `seeds` (add, or enable)."""
    calls = seeds["enable"] if enable else seeds["apply"]
    # A fresh request id per run: the core caches results by id, so a reused id
    # would replay an earlier failure instead of retrying.
    return [{"request_id": str(uuid.uuid4()),
             "session_id": "phase53f-cutover", "akh": seeds["akh"], "tool": seeds["tool"],
             "arguments": c} for c in calls]


def post(url: str, token: str, body: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + token,
                                              "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("roles", nargs="*", default=list(fa.ROLES))
    parser.add_argument("--enable", action="store_true", help="enable the staged schedules")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--url", default=f"http://{fa.CORE_LISTEN}/v1/tools/call")
    args = parser.parse_args()
    root = args.root.resolve()
    token = None if args.dry_run else (root / "secrets/operator.key").read_text().strip()
    failures = 0
    for role in args.roles:
        seeds = json.loads((root / f"config/reactor-seeds/{role}.json").read_text())
        for body in requests_for(seeds, args.enable):
            label = f"{role} {body['arguments']['operation']} {body['arguments'].get('id') or body['arguments']['schedule']['id']}"
            if args.dry_run:
                print(json.dumps(body, indent=2))
                continue
            result = post(args.url, token, body)
            ok = result.get("success") or "already exists" in result.get("text", "")
            failures += not ok
            print(("ok     " if ok else "FAILED ") + label + ": " + result.get("text", "")[:300])
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError) as error:
        print("apply_reactor_seeds: " + str(error), file=sys.stderr)
        sys.exit(1)
