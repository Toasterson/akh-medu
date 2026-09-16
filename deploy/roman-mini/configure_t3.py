#!/usr/bin/env python3
"""Install scoped Roman ACP entrypoints and merge provider entries into T3.

Run before starting/restarting akhsessd. Does not restart T3 or modify threads.
Existing settings and each existing Roman credential are preserved.
"""
import json
import plistlib
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
for role, label in [('janus', 'Janus · Project Manager'), ('minerva', 'Minerva · Product Owner'), ('vesta', 'Vesta · Incident Operations')]:
    token = root / 'secrets' / f't3-{role}.key'
    if not token.exists():
        with token.open('x') as file:
            token.chmod(0o600)
            file.write(secrets.token_urlsafe(48))
    principal = 't3-' + role
    if not any(p['id'] == principal for p in session['principals']):
        session['principals'].append({'id': principal, 'token_file': str(token), 'roles': ['client'], 'akhs': [role]})
    binary = root / 'bin' / ('roman-' + role)
    shutil.copy2(source / 't3_entrypoint.py', binary)
    binary.chmod(0o700)
    data.setdefault('providerInstances', {})['roman-' + role] = {
        'driver': 'grok', 'displayName': label, 'enabled': True,
        'config': {'enabled': True, 'binaryPath': str(binary), 'customModels': []}}
for role in ('janus','minerva','vesta'):
    binary=root/'bin'/('roman-activity-'+role)
    shutil.copy2(source/'t3_activity.py',binary); binary.chmod(0o700)
    data.setdefault('providerInstances',{})['roman-activity-'+role]={
        'driver':'grok','displayName':role.title()+' · Autonomous activity','enabled':True,
        'config':{'enabled':True,'binaryPath':str(binary),'customModels':[]}}
shutil.copy2(source/'t3_activity.py',root/'bin/t3_activity.py')
(root/'launchd/dev.roman-akhs.t3-activity.plist').write_bytes(plistlib.dumps({
    'Label':'dev.roman-akhs.t3-activity','ProgramArguments':['/usr/bin/python3',str(root/'bin/t3_activity.py'),'--service'],
    'RunAtLoad':True,'KeepAlive':True,'ThrottleInterval':30,'Umask':63,
    'EnvironmentVariables':{'HOME':str(Path.home()),'PATH':'/opt/homebrew/bin:/usr/bin:/bin'},
    'StandardOutPath':str(root/'logs/t3-activity.out.log'),'StandardErrorPath':str(root/'logs/t3-activity.err.log')}))
session_path.write_text(json.dumps(session, indent=2) + '\n')
staged = settings.with_suffix('.roman-tmp')
staged.write_text(json.dumps(data, indent=2) + '\n')
staged.chmod(0o600)
staged.replace(settings)
print('Roman T3 provider entries installed; restart only the Roman session daemon if new credentials were added')
