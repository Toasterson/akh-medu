#!/usr/bin/env python3
"""Install scoped Roman ACP entrypoints and merge provider entries into T3.

Run before starting/restarting akhsessd. Does not restart T3 or modify threads.
Existing settings and each existing Roman credential are preserved.

Lilith (ADR 116) is added as `roman-lilith` ("Lilith · Chief of Staff") when
`ROOT/config/lilith.json` exists. She is not on akhsessd, so her `t3-lilith`
principal is NOT added to config/sessions.json: configure_core.py renders it
into akhomed's host config. Only her client key (created once, never
overwritten) and provider entry are handled here. Typical run:

    python3 configure_t3.py ROOT ~/.t3/userdata/settings.json
"""
import json
from pathlib import Path
import secrets
import shutil
import sys

root = Path(sys.argv[1]).resolve()
settings = Path(sys.argv[2]).resolve()
source = Path(__file__).resolve().parent
session_path = root / 'config/sessions.json'
session = json.loads(session_path.read_text())
data = json.loads(settings.read_text())
backup = root / 'evidence/t3-settings-before-romans.json'
if not backup.exists():
    backup.write_text(json.dumps(data, indent=2))
    backup.chmod(0o600)
providers = [('janus', 'Janus · Project Manager'), ('minerva', 'Minerva · Product Owner'), ('vesta', 'Vesta · Incident Operations')]
if (root / 'config/lilith.json').exists():
    providers.append(('lilith', 'Lilith · Chief of Staff'))
for role, label in providers:
    token = root / 'secrets' / f't3-{role}.key'
    if not token.exists():
        with token.open('x') as file:
            token.chmod(0o600)
            file.write(secrets.token_urlsafe(48))
    principal = 't3-' + role
    if role != 'lilith' and not any(p['id'] == principal for p in session['principals']):
        session['principals'].append({'id': principal, 'token_file': str(token), 'roles': ['client'], 'akhs': [role]})
    binary = root / 'bin' / ('roman-' + role)
    shutil.copy2(source / 't3_entrypoint.py', binary)
    binary.chmod(0o700)
    data.setdefault('providerInstances', {})['roman-' + role] = {
        'driver': 'grok', 'displayName': label, 'enabled': True,
        'config': {'enabled': True, 'binaryPath': str(binary), 'customModels': []}}
session_path.write_text(json.dumps(session, indent=2) + '\n')
staged = settings.with_suffix('.roman-tmp')
staged.write_text(json.dumps(data, indent=2) + '\n')
staged.chmod(0o600)
staged.replace(settings)
print('Roman T3 provider entries installed; restart only the Roman session daemon if new credentials were added')
