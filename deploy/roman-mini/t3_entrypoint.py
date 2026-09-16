#!/usr/bin/env python3
"""T3's existing Grok ACP driver used solely as a transport for Roman akhs.

No Grok model/API is used. Ignore CLI permission flags: the daemon owns grants.
The executable basename selects an operator-provisioned, individually scoped
credential; project cwd and T3 filesystem capabilities never become agent tools.
"""
import os
from pathlib import Path
import sys
import urllib.request

root = Path(__file__).resolve().parent.parent
role = Path(sys.argv[0]).name.removeprefix('roman-')
if role not in ('janus', 'minerva', 'vesta'):
    raise SystemExit('unknown Roman Akh')
if '--version' in sys.argv:
    print('Roman Akh ACP bridge 0.1.0')
elif 'models' in sys.argv:
    token = (root / 'secrets' / ('t3-' + role + '.key')).read_text().strip()
    request = urllib.request.Request('http://127.0.0.1:18431/v1/sessions', headers={'Authorization': 'Bearer ' + token})
    try:
        with urllib.request.urlopen(request, timeout=5):
            pass
        print('You are logged in to the Roman Akh session daemon.\nDefault model: grok-build\nAvailable models:\n * grok-build (default)\nTransport alias only: configured daemon model is Mistral Large 3 via Ollama Cloud.')
    except Exception:
        raise SystemExit('Roman session daemon not authenticated or unavailable')
else:
    binary = root / 'bin/akh-acp'
    os.execv(binary, [str(binary), '--endpoint', 'ws://127.0.0.1:18431/acp', '--token-file', str(root / 'secrets' / ('t3-' + role + '.key')), '--akh', role, '--workspace-profile', role, '--t3-compat'])
