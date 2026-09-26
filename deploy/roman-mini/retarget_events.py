#!/usr/bin/env python3
"""Re-target akheventd routes from akhsessd automations to the akhs' reactors (53f step 6).

Installs `config/events.akh-target.json` (rendered by configure_core.py) as
`config/events.json`, keeping the previous file as
`evidence/events-before-phase53.json`. Run it only after every reactor seed is
applied AND enabled: akhomed answers an event nobody subscribes to with
`no_subscriber`, which akheventd records as terminal and never retries.
Does not restart akheventd. `--rollback` restores the saved file.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full_akh as fa  # noqa: E402


def retarget(root: Path) -> list[str]:
    config = root / "config"
    staged = json.loads((config / "events.akh-target.json").read_text())
    current = json.loads((config / "events.json").read_text())
    backup = root / "evidence/events-before-phase53.json"
    if not backup.exists():
        fa.write_atomic(backup, json.dumps(current, indent=2) + "\n")
    for route in staged.get("routes", []):
        if route.get("automation") is not None or route.get("target", {}).get("kind") != "akh":
            raise fa.RenderError(f"staged route {route.get('id')!r} is not an akh target; rerun configure_core.py")
    missing = fa.missing_monitor_files(staged)
    if missing:
        # akheventd reads every endpoint CA and principal token at start; a
        # missing file would take all event routing down, not just monitoring.
        raise fa.RenderError(
            "the staged akheventd config needs files that do not exist: " + ", ".join(missing)
            + ". Create the Grafana service-account token and export the lab CA (PR 5 / D2 in "
            "deploy/roman-mini/INVENTORY.md), rerun configure_core.py, then retarget again")
    fa.write_atomic(config / "events.json", json.dumps(staged, indent=2) + "\n")
    return [f"{r['id']}: {r['source']}/{r['event_type']} -> akh {r['target']['akh']}" for r in staged["routes"]]


def rollback(root: Path) -> None:
    backup = root / "evidence/events-before-phase53.json"
    if not backup.exists():
        raise fa.RenderError(f"{backup} not found; nothing to roll back")
    fa.write_atomic(root / "config/events.json", backup.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    try:
        if args.rollback:
            rollback(args.root.resolve())
            print("Restored the pre-Phase-53 events.json; restart dev.roman-akhs.events.")
        else:
            for line in retarget(args.root.resolve()):
                print(line)
            print("events.json re-targeted; restart dev.roman-akhs.events to load it.")
    except (fa.RenderError, OSError, KeyError) as error:
        print("retarget_events: " + str(error), file=sys.stderr)
        sys.exit(1)
