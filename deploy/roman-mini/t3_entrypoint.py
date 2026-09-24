#!/usr/bin/env python3
"""T3's existing Grok ACP driver used solely as a transport for Roman akhs.

No Grok model/API is used. Ignore CLI permission flags: the daemon owns grants.
The executable basename selects an operator-provisioned, individually scoped
credential; project cwd and T3 filesystem capabilities never become agent tools.
"""
import json
import os
from pathlib import Path
import sys
import urllib.request

root = Path(__file__).resolve().parent.parent
role = Path(sys.argv[0]).name.removeprefix('roman-')
if role not in ('janus', 'minerva', 'vesta'):
    raise SystemExit('unknown Roman Akh')
# Phase 53f cutover (ADR 113, 53d notes): `config/t3-acp.json`, rendered by
# configure_core.py, points this transport at akhomed's own ACP TCP listener via
# `akh-acp --akhomed`. Without that file (or with ROMAN_T3_ACP_ENDPOINT set to a
# ws:// URL) the transport stays on the akhsessd broker. Deleting the file is the
# rollback.
settings_path = root / 'config' / 't3-acp.json'
settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
endpoint = os.environ.get('ROMAN_T3_ACP_ENDPOINT', settings.get('endpoint', 'ws://127.0.0.1:18431/acp'))
health_url = os.environ.get('ROMAN_T3_HEALTH_URL', settings.get('health_url', 'http://127.0.0.1:18431/v1/sessions'))
akhomed = endpoint.startswith('tcp://')
if '--version' in sys.argv:
    print('Roman Akh ACP bridge 0.1.0')
elif 'models' in sys.argv:
    token = (root / 'secrets' / ('t3-' + role + '.key')).read_text().strip()
    request = urllib.request.Request(health_url, headers={'Authorization': 'Bearer ' + token})
    try:
        with urllib.request.urlopen(request, timeout=5):
            pass
        model = 'GLM-5.3 on Ollama Cloud, in-process in akhomed' if akhomed else 'Mistral Large 3 via Ollama Cloud'
        print('You are logged in to the Roman Akh session daemon.\nDefault model: grok-build\nAvailable models:\n * grok-build (default)\nTransport alias only: configured daemon model is ' + model + '.')
    except Exception:
        raise SystemExit('Roman session daemon not authenticated or unavailable')
else:
    binary = Path(settings.get('binary', root / 'bin/akh-acp')) if akhomed else root / 'bin/akh-acp'
    token_file = str(root / 'secrets' / ('t3-' + role + '.key'))
    if akhomed:
        # akhomed has no workspace profiles; `--akhomed` injects the bearer token
        # and the akh selection into initialize / session/new.
        argv = [str(binary), '--akhomed', '--endpoint', endpoint, '--token-file', token_file, '--akh', role, '--t3-compat']
    else:
        argv = [str(binary), '--endpoint', endpoint, '--token-file', token_file, '--akh', role, '--workspace-profile', role, '--t3-compat']
    os.execv(binary, argv)
