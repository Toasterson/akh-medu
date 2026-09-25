#!/usr/bin/env python3
"""Render the laya-controller LaunchAgent on the system python3.

`laya_service.py` (the Laya classifier, a resident HTTP model server that
needed torch/transformers) is retired: Phase 53g step-2 plan, PR 2, operator
decision D1. `laya_controller.py` no longer imports it or makes any HTTP call
to it (see its module docstring), so the controller's LaunchAgent no longer
needs the `experiments/typed-decisions/.venv-comparison` interpreter that used
to load those model libraries -- it now runs on `/usr/bin/python3`, the same
system interpreter every other roman-akhs Python LaunchAgent uses.

Renders `launchd/dev.roman-akhs.laya-controller.plist` into `<root>/launchd/`
(it does not install the plist -- copy it to `~/Library/LaunchAgents` and
`launchctl bootstrap` by hand, per INVENTORY.md's Mac-side steps) and copies
`laya_controller.py` into `<root>/bin/`, mirroring `configure_vesta.py` /
`configure_janus_escalation.py`. The write is atomic (`full_akh.write_atomic`)
and the plist is world-readable (0o644), like `configure_core.py`'s core
plist -- it names no secret, only a loopback launchd/plist arrangement.
"""
import plistlib
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full_akh as fa  # noqa: E402

LABEL = 'dev.roman-akhs.laya-controller'

# Read verbatim off the live plist (~/Library/LaunchAgents/dev.roman-akhs.laya-controller.plist)
# on 2026-09-25, minus the venv interpreter in ProgramArguments[0].
ENVIRONMENT = {'TOKENIZERS_PARALLELISM': 'false', 'USE_TF': '0', 'HF_HUB_OFFLINE': '1'}


def render(root):
    root = Path(root).resolve()
    source = Path(__file__).resolve().parent
    (root / 'bin').mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / 'laya_controller.py', root / 'bin' / 'laya_controller.py')
    plist = {
        'Label': LABEL,
        'ProgramArguments': ['/usr/bin/python3', str(root / 'bin/laya_controller.py'), str(root)],
        'RunAtLoad': True,
        'KeepAlive': True,
        'ThrottleInterval': 15,
        'ExitTimeOut': 10,
        'Umask': 63,
        'EnvironmentVariables': dict(ENVIRONMENT),
        'StandardOutPath': str(root / 'logs/laya-controller.out.log'),
        'StandardErrorPath': str(root / 'logs/laya-controller.err.log'),
    }
    path = root / 'launchd' / f'{LABEL}.plist'
    fa.write_atomic(path, plistlib.dumps(plist), mode=0o644)
    return path


if __name__ == '__main__':
    rendered = render(sys.argv[1])
    print('Rendered ' + str(rendered) + ' for the system python3 (laya_service retired)')
